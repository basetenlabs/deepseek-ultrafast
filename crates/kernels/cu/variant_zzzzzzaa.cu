#include "variant_zzzzzy3.cu"
#include "variant_zcomm_dec.cu"

struct AaPush {
    float* recv[4];
    const uint32_t* state;
    const Meta* meta;
    int cap, rank;
    static constexpr int TPC = 4;
};
struct AaPush8 {
    float* recv[8];
    const uint32_t* state;
    const Meta* meta;
    int cap, rank;
    static constexpr int TPC = 8;
};

enum { AA_PUSH = 4096, AA_POUT = 8192 };

template <class Q>
DEV void aa_body(const Tg2Params& p, const Q& q) {
    using F = TgF<0, V_PART>;
    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* sm = align1k(smem_raw);
    uint32_t crank, cs;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cs));
    const int rt = blockIdx.x;
    const int M = p.M;
    constexpr int TKB = 128;
    const int KS = p.K / (int)cs, KT = KS / TKB, k0 = (int)crank * KS;
    uint8_t* sA = sm;
    uint8_t* sB = sA + KT * 16384;
    uint8_t* sSFA = sB + KT * 1024;
    uint8_t* sSFB = sSFA + KT * 512;
    float* recv = (float*)(sSFB + KT * 512);
    __shared__ __align__(8) uint64_t wbar[TG2_KT], xbar, mbar, rbar, tbar;
    __shared__ uint32_t tbase;
    const bool push = cs > 1;
    const int nunits = 4 * M;
    const int nown = (int)crank < nunits ? (nunits - (int)crank + (int)cs - 1) / (int)cs : 0;
    constexpr uint32_t TMCOLS = 128;
    const bool bload = p.bload;
    const int po0 = warp % (int)cs, pslot0 = warp / (int)cs;
    if (warp == 0) tmalloc(&tbase, TMCOLS);
    if (warp == 0) asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    if (tid == 0) {
        for (int t = 0; t < TG2_KT; ++t) mb_init_t(&wbar[t], 1);
        mb_init_t(&xbar, 1);
        mb_init_t(&mbar, 1);
        mb_init_t(&rbar, 1);
        mb_init_t(&tbar, 4);
        if (push) mb_expect_t(&rbar, (uint32_t)(nown * (int)cs * 128));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (push) asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    const uint32_t tm = tbase;
    if (tid == 0) {
        for (int t = 0; t < KT; ++t) {
            mb_expect_t(&wbar[t], 16384 + (t == 0 ? KT * 512 : 0));
            tma2d_ef(sA + t * 16384, p.tmap, k0 + t * 128, rt * 128, &wbar[t]);
            if (t == 0) bulk1d(sSFA, p.sfa + ((size_t)rt * p.kpt + k0 / 128) * 512, KT * 512, &wbar[0]);
        }
    }
    if (warp == 1) {
        __syncwarp();
        const uint32_t tmu = __shfl_sync(0xffffffffu, tm, 0);
        tg3_wait(s32a(&wbar[0]), 0);
        tcafter();
        const uint64_t sfa0 = sdesc_t(sSFA, 512, 128, 0);
        for (int t = 0; t < KT; ++t) tg3_cp(tmu + 32 + 4 * t, sfa0 + (uint64_t)(t * 32));
    }
    if (push) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    pdl_wait();
    pdl_launch();
    int aa_m = 0;
    constexpr int AJ = 4;
    uint32_t aa_e[AJ] = {0u, 0u, 0u, 0u};
    uint32_t aa_e1[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    aa_m = q.meta->m;
    if (cs == 1) {
#pragma unroll
        for (int m = 0; m < 8; ++m)
            if (m < M) aa_e1[m] = __ldcg(q.state + (size_t)((m * p.N + rt * 128 + 32 * warp) >> 8) * 4);
    }
#pragma unroll
    for (int j = 0; j < AJ; ++j) {
        const int s = ((warp + 3) & 3) + 4 * j;
        if (cs > 1 && s < nown) {
            const int u = (int)crank + s * (int)cs;
            const int sl = ((u >> 2) * p.N + rt * 128 + 32 * (u & 3)) >> 8;
            aa_e[j] = __ldcg(q.state + (size_t)sl * 4);
        }
    }
    if (tid == 0 && !bload) {
        mb_expect_t(&xbar, KT * 1024 + KT * 512);
        bulk1d(sB, p.bimg + (size_t)(k0 / 128) * 1024, KT * 1024, &xbar);
        bulk1d(sSFB, p.bimg + (size_t)(p.K / 128) * 1024 + (size_t)(k0 / 128) * 512, KT * 512, &xbar);
    }
    uint4 bv[4];
    const int bimg_n = bload ? KT * 96 : 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = tid + 128 * j;
        bv[j] = make_uint4(0, 0, 0, 0);
        if (i < bimg_n) {
            const uint4* src = i < KT * 64 ? (const uint4*)(p.bimg + (size_t)(k0 / 128) * 1024) + i
                                           : (const uint4*)(p.bimg + (size_t)(p.K / 128) * 1024 + (size_t)(k0 / 128) * 512) + (i - KT * 64);
            bv[j] = __ldcg(src);
        }
    }
    if (bimg_n) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + 128 * j;
            if (i < bimg_n) {
                if (i < KT * 64) ((uint4*)sB)[i] = bv[j];
                else ((uint4*)sSFB)[i - KT * 64] = bv[j];
            }
        }
        tcfence_proxy();
        __syncthreads();
    }
    if (warp == 1) {
        __syncwarp();
        const uint32_t tmu = __shfl_sync(0xffffffffu, tm, 0);
        const uint32_t wb = s32a(&wbar[0]);
        if (!bload) tg3_wait(s32a(&xbar), 0);
        tcafter();
        const uint64_t sfb0 = sdesc_t(sSFB, 512, 128, 0);
        for (int t = 0; t < KT; ++t) tg3_cp(tmu + 64 + 4 * t, sfb0 + (uint64_t)(t * 32));
        const uint64_t a0 = sdesc_t(sA, 16, 1024, 2), b0 = sdesc_t(sB, 128, 1024, 0);
        for (int t = 0; t < KT; ++t) {
            tg3_wait(wb + 8u * (uint32_t)t, 0);
            tcafter();
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) {
                const uint64_t ad = a0 + (uint64_t)(t * 1024 + kc * 2), bd = b0 + (uint64_t)(t * 64 + kc * 16);
                const uint32_t acc = (t | kc) != 0;
                tg3_mma_f8(tmu, ad, bd, idesc_e4m3(8, kc), tmu + 32 + 4 * t, tmu + 64 + 4 * t, acc);
            }
        }
        tg3_commit(s32a(&mbar));
    }
    __syncwarp();
    tg3_wait(s32a(&mbar), 0);
    tcafter();
    float acc[8];
    {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                     : "r"(tm + ((uint32_t)(32 * warp) << 16))
                     : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
        for (int m = 0; m < 8; ++m) acc[m] = __uint_as_float(r[m]);
    }
    tcbefore();
    tg3_arrive(s32a(&tbar));
    if (cs != 1) {
        const uint32_t rcv = s32a(recv), rbs = s32a(&rbar);
        int o = po0, slot = pslot0;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < M) tg3_st_async1(tg3_mapa(rcv + (uint32_t)(((slot * (int)cs + (int)crank) * 32 + lane) * 4), (uint32_t)o), acc[m] * p.alpha, tg3_mapa(rbs, (uint32_t)o));
            o += 4;
            if (o >= (int)cs) { o -= (int)cs; ++slot; }
            if (o >= (int)cs) { o -= (int)cs; ++slot; }
        }
    }
    __syncwarp();
    if (warp == 0) {
        tg3_wait(s32a(&tbar), 0);
        tcafter();
        tmdealloc(tm, TMCOLS);
    }
    if (cs == 1) {
        __shared__ __align__(16) float pst[4][8][32];
#pragma unroll
        for (int m = 0; m < 8; ++m)
            if (m < M) pst[warp][m][lane] = acc[m] * p.alpha;
        __syncwarp();
        const int m = lane >> 2, f0 = (lane & 3) * 8;
        if (m < M && m < aa_m) {
            const float4 a0 = *(const float4*)&pst[warp][m][f0], a1 = *(const float4*)&pst[warp][m][f0 + 4];
            const uint32_t w0 = pk_bf2(a0.x, a0.y), w1 = pk_bf2(a0.z, a0.w), w2 = pk_bf2(a1.x, a1.y), w3 = pk_bf2(a1.z, a1.w);
            const int i = m * p.N + rt * 128 + 32 * warp + f0;
            uint32_t e = aa_e1[0];
#pragma unroll
            for (int k = 1; k < 8; ++k)
                if (k == m) e = aa_e1[k];
            const size_t o = (size_t)comm_stage(comm_epoch_next(e)) * Q::TPC * (size_t)q.cap + (size_t)q.rank * q.cap + (size_t)(i / 256) * 256 + (i % 256) / 2;
#pragma unroll
            for (int r = 0; r < Q::TPC; ++r)
                asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"((uint32_t*)q.recv[r] + o), "r"(w0), "r"(w1), "r"(w2), "r"(w3) : "memory");
        }
    } else {
        size_t aa_o[AJ] = {0, 0, 0, 0};
        asm volatile("" : "+r"(aa_m), "+r"(aa_e[0]), "+r"(aa_e[1]), "+r"(aa_e[2]), "+r"(aa_e[3]) :: "memory");
#pragma unroll
        for (int j = 0; j < AJ; ++j) {
            const int u = (int)crank + (((warp + 3) & 3) + 4 * j) * (int)cs;
            const int i = (u >> 2) * p.N + rt * 128 + 32 * (u & 3) + lane;
            aa_o[j] = (size_t)comm_stage(comm_epoch_next(aa_e[j])) * Q::TPC * (size_t)q.cap + (size_t)q.rank * q.cap + (size_t)(i / 256) * 256 + (i % 256) / 2;
        }
        tg3_wait(s32a(&rbar), 0);
        const int s0 = (warp + 3) & 3;
        float v[AJ];
#pragma unroll
        for (int j = 0; j < AJ; ++j) {
            const int s = s0 + 4 * j;
            float t[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) t[r] = s < nown && r < (int)cs ? recv[(s * (int)cs + r) * 32 + lane] : 0.f;
            float a = 0.f;
#pragma unroll
            for (int r = 0; r < 4; ++r) a += t[r];
            v[j] = a;
        }
        uint32_t w0[AJ], w1[AJ];
#pragma unroll
        for (int j = 0; j < AJ; ++j) w0[j] = pk_bf2(v[j], __shfl_down_sync(0xffffffffu, v[j], 1));
#pragma unroll
        for (int j = 0; j < AJ; ++j) w1[j] = __shfl_down_sync(0xffffffffu, w0[j], 2);
#pragma unroll
        for (int j = 0; j < AJ; ++j) {
            const int s = s0 + 4 * j;
            if (s < nown && ((int)crank + s * (int)cs) >> 2 < aa_m && (lane & 3) == 0) {
#pragma unroll
                for (int r = 0; r < Q::TPC; ++r)
                    asm volatile("st.relaxed.sys.global.v2.u32 [%0], {%1,%2};" ::"l"((uint32_t*)q.recv[r] + aa_o[j]), "r"(w0[j]), "r"(w1[j]) : "memory");
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv3yp_part(const __grid_constant__ Tg2Params p, const __grid_constant__ AaPush q) {
    aa_body(p, q);
}
extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv3yp8_part(const __grid_constant__ Tg2Params p, const __grid_constant__ AaPush8 q) {
    aa_body(p, q);
}

template <int TPC = 4>
DEV void comm_dec_pp_body(const Meta* meta, const NvlsParams& q) {
    const CommParams& p = q.c;
    __shared__ uint32_t s_epoch, s_l0, s_l1;
    __shared__ float red[2][2];
    const int tid = threadIdx.x, b = blockIdx.x, npt = p.n_per_token;
    const int i4 = b * blockDim.x + tid, i = i4 * 4, m = i / npt, c = i % npt;
    pdl_wait();
    pdl_launch();
    if (tid == 0) {
        const uint4 st = *(const uint4*)(p.state + b * 4);
        s_epoch = comm_epoch_next(st.x);
        s_l0 = st.y;
        s_l1 = st.z;
    }
    const int M = meta->m;
    const int n4 = M * npt / 4;
    __syncthreads();
    const uint32_t e = s_epoch;
    const int stage = comm_stage(e), prev = comm_prev_stage(stage);
    const size_t slot = (size_t)TPC * p.cap;
    float* const mine = p.recv[p.rank];
    const bool act = i4 < n4;
    const size_t wi = (size_t)(i / 256) * 256 + (i % 256) / 2;
    {
        float* pb = mine + prev * slot;
        for (int j = s_l0 + tid; j < (int)s_l1; j += blockDim.x)
#pragma unroll
            for (int s = 0; s < TPC; ++s) *(float4*)(pb + (size_t)s * p.cap + (size_t)j * 4) = make_float4(-0.f, -0.f, -0.f, -0.f);
        if (tid == 0) {
            const int lo = b * blockDim.x, hi = min(n4, lo + (int)blockDim.x);
            *(uint4*)(p.state + b * 4) = make_uint4(e, (uint32_t)lo, (uint32_t)(hi > lo ? hi : lo), 0u);
        }
    }
    if (act) {
        HcIn hin;
        hc_in_load(p, m, c, npt, hin);
        float pr[4];
        const bool hp = p.e.pre != nullptr;
        if (hp) {
            const float4 t = *(const float4*)(p.e.pre + m * HC);
            pr[0] = t.x; pr[1] = t.y; pr[2] = t.z; pr[3] = t.w;
        }
        const float* rb = mine + stage * slot;
        float4 parts[TPC];
        {
            uint2 w[TPC];
#pragma unroll
            for (int s = 0; s < TPC; ++s) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(w[s].x), "=r"(w[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
            unsigned miss = 0;
#pragma unroll
            for (int s = 0; s < TPC; ++s) if (!(arr_bf2(w[s].x) && arr_bf2(w[s].y))) miss |= 1u << s;
            while (miss) {
#pragma unroll
                for (int s = 0; s < TPC; ++s)
                    if (miss & (1u << s)) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(w[s].x), "=r"(w[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
#pragma unroll
                for (int s = 0; s < TPC; ++s) if ((miss & (1u << s)) && arr_bf2(w[s].x) && arr_bf2(w[s].y)) miss &= ~(1u << s);
            }
#pragma unroll
            for (int s = 0; s < TPC; ++s) parts[s] = make_float4(__uint_as_float(w[s].x << 16), __uint_as_float(w[s].x & 0xffff0000u), __uint_as_float(w[s].y << 16), __uint_as_float(w[s].y & 0xffff0000u));
        }
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < TPC; ++s) {
            float vv[4] = {parts[s].x, parts[s].y, parts[s].z, parts[s].w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                vv[k] = __bfloat162float(__float2bfloat16(vv[k]));
                acc[k] += vv[k];
            }
        }
        const int dim = npt;
        float hn[4][4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float x = __bfloat162float(__float2bfloat16(acc[k]));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s2v = 0.f;
#pragma unroll
                for (int h = 0; h < 4; ++h) s2v += hin.cb[h][j] * hin.rr[h][k];
                hn[j][k] = __bfloat162float(__float2bfloat16(hin.po[j] * x + s2v));
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint2 o;
            o.x = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][0])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][1])) << 16);
            o.y = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][2])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][3])) << 16);
            *(uint2*)((bf16*)p.out + ((size_t)m * 4 + j) * dim + c) = o;
        }
        float xr[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float y = 0.f;
            if (hp) {
#pragma unroll
                for (int h = 0; h < HC; ++h) y += pr[h] * hn[h][k];
            } else {
                y = hn[0][k];
            }
            xr[k] = __bfloat162float(__float2bfloat16(y));
        }
        *(__nv_bfloat162*)(p.e.x_raw + (size_t)m * p.e.dim + c) = __floats2bfloat162_rn(xr[0], xr[1]);
        *(__nv_bfloat162*)(p.e.x_raw + (size_t)m * p.e.dim + c + 2) = __floats2bfloat162_rn(xr[2], xr[3]);
        float sh = 0.f, sx = 0.f;
#pragma unroll
        for (int h = 0; h < HC; ++h)
#pragma unroll
            for (int k = 0; k < 4; ++k) sh += hn[h][k] * hn[h][k];
#pragma unroll
        for (int k = 0; k < 4; ++k) sx += xr[k] * xr[k];
        __syncwarp();
        sh = warp_sum(sh);
        sx = warp_sum(sx);
        if ((tid & 31) == 0) { red[tid >> 5][0] = sh; red[tid >> 5][1] = sx; }
        __syncthreads();
        if (tid < 2) p.e.part[((size_t)m * (dim / SLICE) + c / SLICE) * NP + MIX + tid] = red[0][tid] + red[1][tid];
    }
}
extern "C" __global__ void __launch_bounds__(64) comm_dec_attn_pp(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_pp_body(meta, q); }
extern "C" __global__ void __launch_bounds__(64) comm_dec8_attn_pp(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_pp_body<8>(meta, q); }
