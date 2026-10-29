#include "tc.cuh"
#include "rope.cuh"

struct Meta {
    int pos;
    int m;
    int tokens[16];
};

struct TvParams {
    const void* tmap;
    const uint8_t* sfa;
    const bf16* x;
    const bf16* prenorm_w;
    const float* norm_part;
    void* out;
    void* out2;
    const float* rope_cos;
    const float* rope_sin;
    const Meta* meta;
    float alpha, eps;
    int M, N, K, kpt;
    int ldx, ldo, ldo2, n_split, out_f32;
    int nslices;
    int rope_head, rope_off;
    int partial;
    int group_rows;
    RopeParams kvr;
    int kvr_on;
    int _pad2;
};

extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv_fp8(const __grid_constant__ TvParams p) {
    if (p.kvr_on && blockIdx.x == gridDim.x - 1) {
        pdl_wait();
        pdl_launch();
        const int grp = threadIdx.x >> 6, ng = 2;
        const int step = ng * (int)gridDim.y;
        for (int row = (int)blockIdx.y * ng + grp; row < 16 * p.kvr.rows_per_token; row += step)
            rope_body((const int*)p.meta, p.kvr, row, threadIdx.x & 63, grp);
        return;
    }
    if (threadIdx.x == 0) asm volatile("prefetch.tensormap [%0];" ::"l"(p.tmap) : "memory");
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* sm = align1k(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint32_t crank, cs;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cs));
    const int rt = blockIdx.x;
    const int KS = p.K / cs, KT = KS / 128;
    const int k0 = crank * KS;
    const int M = p.M;
    uint8_t* sA = sm;
    uint8_t* sB = sA + KT * 16384;
    uint8_t* sSFA = sB + KT * 1024;
    uint8_t* sSFB = sSFA + KT * 512;
    bf16* xs = (bf16*)(sSFB + KT * 512);
    bf16* pws = xs + 8 * KS;
    float* red = (float*)(pws + KS);
    float* recv = red + 8 * 128;
    __shared__ __align__(8) uint64_t wbar, xbar, mbar, rbar;
    __shared__ uint32_t tbase;
    __shared__ float rs[8];
    const int total = M * 128;
    const int per = ((total + cs - 1) / cs + 3) & ~3;
    const bool push = cs > 1 && !p.partial;
    if (warp == 0) tmalloc(&tbase, 128);
    if (tid == 0) {
        mb_init_t(&wbar, 1);
        mb_init_t(&xbar, 1);
        mb_init_t(&mbar, 1);
        mb_init_t(&rbar, 1);
        if (push) mb_expect_t(&rbar, (uint32_t)((cs - 1) * per * 4));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (push) asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    tcbefore();
    __syncthreads();
    tcafter();
    const uint32_t tm = tbase;
    if (tid == 0) {
        mb_expect_t(&wbar, KT * 16384 + KT * 512);
        for (int t = 0; t < KT; ++t) tma2d(sA + t * 16384, p.tmap, k0 + t * 128, rt * 128, &wbar);
        bulk1d(sSFA, p.sfa + ((size_t)rt * p.kpt + k0 / 128) * 512, KT * 512, &wbar);
    }
    for (int i = tid; i < KT * 512 / 4; i += 128) ((uint32_t*)sSFB)[i] = 0x7f7f7f7fu;
    pdl_wait();
    pdl_launch();
    if (tid == 0) {
        mb_expect_t(&xbar, M * KS * 2 + (p.prenorm_w ? KS * 2 : 0));
        for (int m = 0; m < M; ++m) bulk1d(xs + m * KS, p.x + (size_t)m * p.ldx + k0, KS * 2, &xbar);
        if (p.prenorm_w) bulk1d(pws, p.prenorm_w + k0, KS * 2, &xbar);
    }
    if (p.prenorm_w) {
        if (p.nslices) {
            if (tid < 8) rs[tid] = rsqrtf((tid < M ? norm_part_sum(p.norm_part, tid, p.nslices) : 0.f) / p.K + p.eps);
        } else {
            for (int m = warp; m < 8; m += 4) {
                float ss = 0.f;
                if (m < M)
                    for (int k = lane * 8; k < p.K; k += 256) {
                        const uint4 q = *(const uint4*)(p.x + (size_t)m * p.ldx + k);
                        const bf16* h = (const bf16*)&q;
#pragma unroll
                        for (int e = 0; e < 8; ++e) { const float v = __bfloat162float(h[e]); ss += v * v; }
                    }
                for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
                if (lane == 0) rs[m] = rsqrtf(ss / p.K + p.eps);
            }
        }
    }
    mb_wait_t(&xbar, 0);
    __syncthreads();
    for (int b = tid; b < M * (KS / 32); b += 128) {
        const int m = b / (KS / 32), kb = b % (KS / 32);
        float v[32];
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 32; i += 8) {
            const uint4 q = *(const uint4*)(xs + m * KS + kb * 32 + i);
            const bf16* h = (const bf16*)&q;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float f = __bfloat162float(h[e]);
                if (p.prenorm_w) f = __bfloat162float(__float2bfloat16(__bfloat162float(pws[kb * 32 + i + e]) * (f * rs[m])));
                v[i + e] = f;
                a = fmaxf(a, fabsf(f));
            }
        }
        const float sc = round_scale_pow2(fmaxf(a, 1e-4f), 1.0f / 448.0f);
        const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
        const int t = kb / 4, kc = kb % 4;
        uint8_t* img = sB + t * 1024 + m * 16;
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            uint32_t w[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int i = h2 * 16 + 4 * j;
                const uint16_t lo = f32x2_to_e4m3x2(v[i] * inv, v[i + 1] * inv), hi = f32x2_to_e4m3x2(v[i + 2] * inv, v[i + 3] * inv);
                w[j] = (uint32_t)lo | ((uint32_t)hi << 16);
            }
            *(uint4*)(img + (kc * 2 + h2) * 128) = make_uint4(w[0], w[1], w[2], w[3]);
        }
        sSFB[t * 512 + m * 16 + kc] = (uint8_t)(__float_as_uint(sc) >> 23);
    }
    tcfence_proxy();
    __syncthreads();
    if (warp == 1) {
        mb_wait_t(&wbar, 0);
        tcafter();
        if (elect_one()) {
            const uint64_t a0 = sdesc_t(sA, 16, 1024, 2), b0 = sdesc_t(sB, 128, 1024, 0);
            const uint64_t sfa0 = sdesc_t(sSFA, 512, 128, 0), sfb0 = sdesc_t(sSFB, 512, 128, 0);
            for (int t = 0; t < KT; ++t) {
                tccpsf(tm + 32 + 4 * t, sfa0 + (uint64_t)((t * 512) >> 4));
                tccpsf(tm + 64 + 4 * t, sfb0 + (uint64_t)((t * 512) >> 4));
            }
            for (int t = 0; t < KT; ++t)
#pragma unroll
                for (int kc = 0; kc < 4; ++kc)
                    tcmma(tm, a0 + (uint64_t)((t * 16384 + kc * 32) >> 4), b0 + (uint64_t)((t * 1024 + kc * 256) >> 4), idesc_e4m3(8, kc), tm + 32 + 4 * t,
                          tm + 64 + 4 * t, (t | kc) != 0);
            tccommit(&mbar);
        }
        __syncwarp();
    }
    mb_wait_t(&mbar, 0);
    tcafter();
    {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                     : "r"(tm + ((uint32_t)(32 * warp) << 16))
                     : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        const int row = 32 * warp + lane;
#pragma unroll
        for (int m = 0; m < 8; ++m) red[m * 128 + row] = __uint_as_float(r[m]);
    }
    tcbefore();
    if (p.partial) {
        __syncthreads();
        for (int i = tid; i < M * 128; i += 128) {
            const int m = i / 128, n = rt * 128 + i % 128;
            ((float*)p.out)[((size_t)m * cs + crank) * p.ldo + n] = red[i] * p.alpha;
        }
        tcbefore();
        __syncthreads();
        tcafter();
        if (warp == 0) tmdealloc(tm, 128);
        return;
    }
    __syncthreads();
    if (push) {
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        const int nv = per / 4;
        for (int i = tid; i < (int)(cs - 1) * nv; i += 128) {
            const int qq = i / nv, c4 = i - qq * nv;
            const uint32_t q = qq < (int)crank ? qq : qq + 1;
            const int o = q * per + c4 * 4;
            const float4 v = o + 3 < total ? *(const float4*)(red + o) : make_float4(0.f, 0.f, 0.f, 0.f);
            uint32_t ra, rb;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"((uint32_t)__cvta_generic_to_shared(recv + crank * per + c4 * 4)), "r"(q));
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"((uint32_t)__cvta_generic_to_shared(&rbar)), "r"(q));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rb) : "memory");
        }
        mb_wait_t(&rbar, 0);
    }
    auto total_of = [&](int i) -> float {
        if (cs == 1) return red[i];
        const int li = i - crank * per;
        float v = 0.f;
        for (uint32_t r = 0; r < cs; ++r) v += r == crank ? red[i] : recv[r * per + li];
        return v;
    };
    for (int i = crank * per + tid; i < min(total, (int)(crank + 1) * per); i += 128) {
        const int m = i / 128, n = rt * 128 + i % 128;
        const float v = total_of(i) * p.alpha;
        if (p.rope_cos && (n % p.rope_head) >= p.rope_off) {
            if (n & 1) continue;
            const float a = __bfloat162float(__float2bfloat16(v)), b = __bfloat162float(__float2bfloat16(total_of(i + 1) * p.alpha));
            const int pos = p.meta->pos + m, f = ((n % p.rope_head) - p.rope_off) / 2;
            const float c = p.rope_cos[(size_t)pos * 32 + f], sn = p.rope_sin[(size_t)pos * 32 + f];
            bf16* o = (bf16*)p.out + (size_t)m * p.ldo + n;
            o[0] = __float2bfloat16(__fmaf_rn(a, c, -__fmul_rn(b, sn)));
            o[1] = __float2bfloat16(__fmaf_rn(a, sn, __fmul_rn(b, c)));
            continue;
        }
        void* out = p.out;
        int ldo = p.ldo, nn = n;
        if (p.n_split && n >= p.n_split) { out = p.out2; ldo = p.ldo2; nn = n - p.n_split; }
        if (p.out_f32) ((float*)out)[(size_t)m * ldo + nn] = v;
        else ((bf16*)out)[(size_t)m * ldo + nn] = __float2bfloat16(v);
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (warp == 0) tmdealloc(tm, 128);
}

constexpr int TVB_KT = 8;
extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv_bf16(const __grid_constant__ TvParams p) {
    if (threadIdx.x == 0) asm volatile("prefetch.tensormap [%0];" ::"l"(p.tmap) : "memory");
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* sm = align1k(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint32_t crank, cs;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cs));
    const int rt = blockIdx.x;
    const int KS = p.K / cs, KB = KS / 64;
    const int k0 = crank * KS;
    const int M = p.M;
    const size_t xoff = p.group_rows ? (size_t)((rt * 128) / p.group_rows) * p.K : 0;
    uint8_t* sA = sm;
    uint8_t* sB = sA + KB * 16384;
    float* red = (float*)(sB + KS * 16);
    __shared__ __align__(8) uint64_t wbar[TVB_KT], mbar;
    __shared__ uint32_t tbase;
    if (warp == 0) tmalloc(&tbase, 32);
    if (tid == 0) {
        for (int t = 0; t < TVB_KT; ++t) mb_init_t(&wbar[t], 1);
        mb_init_t(&mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    const uint32_t tm = tbase;
    if (tid == 0) {
        for (int t = 0; t < KB; ++t) {
            mb_expect_t(&wbar[t], 16384);
            tma2d_ef(sA + t * 16384, p.tmap, k0 + t * 64, rt * 128, &wbar[t]);
        }
    }
    pdl_wait();
    pdl_launch();
    for (int i = tid; i < (KS / 8) * 8; i += 128) {
        const int c = i >> 3, m = i & 7;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (m < M) v = *(const uint4*)(p.x + (size_t)m * p.ldx + xoff + k0 + c * 8);
        *(uint4*)(sB + c * 128 + m * 16) = v;
    }
    tcfence_proxy();
    __syncthreads();
    if (warp == 1) {
        const uint64_t a0 = sdesc_t(sA, 16, 1024, 2), b0 = sdesc_t(sB, 128, 1024, 0);
        const uint32_t id = idesc_bf16(8);
        for (int t = 0; t < KB; ++t) {
            mb_wait_t(&wbar[t], 0);
            tcafter();
            if (elect_one()) {
#pragma unroll
                for (int kc = 0; kc < 4; ++kc)
                    tcmma_f16(tm, a0 + (uint64_t)((t * 16384 + kc * 32) >> 4), b0 + (uint64_t)((t * 1024 + kc * 256) >> 4), id, (t | kc) != 0);
            }
            __syncwarp();
        }
        if (elect_one()) tccommit(&mbar);
        __syncwarp();
    }
    mb_wait_t(&mbar, 0);
    tcafter();
    {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                     : "r"(tm + ((uint32_t)(32 * warp) << 16))
                     : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        const int row = 32 * warp + lane;
#pragma unroll
        for (int m = 0; m < 8; ++m) red[m * 128 + row] = __uint_as_float(r[m]);
    }
    tcbefore();
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    const int total = M * 128;
    const int per = (total + cs - 1) / cs;
    for (int i = crank * per + tid; i < min(total, (int)(crank + 1) * per); i += 128) {
        float v = 0.f;
        for (uint32_t r = 0; r < cs; ++r) {
            uint32_t a = (uint32_t)__cvta_generic_to_shared(red + i), ra;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(r));
            float x;
            asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(x) : "r"(ra) : "memory");
            v += x;
        }
        v *= p.alpha;
        const int m = i / 128, n = rt * 128 + i % 128;
        if (p.out_f32) ((float*)p.out)[(size_t)m * p.ldo + n] = v;
        else ((bf16*)p.out)[(size_t)m * p.ldo + n] = __float2bfloat16(v);
    }
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    tcafter();
    if (warp == 0) tmdealloc(tm, 32);
}
