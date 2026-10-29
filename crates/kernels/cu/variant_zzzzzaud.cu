#include "common.cuh"

struct Meta {
    int pos;
    int m;
    int tokens[16];
};

DEV float bf(bf16 v) { return __bfloat162float(v); }
DEV bf16 tobf(float v) { return __float2bfloat16(v); }

DEV void ldsm_x4(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
DEV void ldsm_x2(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
DEV void ldsm_x2_t(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"((uint32_t)__cvta_generic_to_shared(p)));
}

constexpr int AH = 16, AD = 512, ALD = AD + 8;

struct AttnFParams {
    const bf16* q;
    const bf16* win;
    const bf16* comp;
    const int* topk;
    int ktop;
    float scale;
    const float* sink;
    bf16* out;
    const float* cos;
    const float* sin;
    int R;
    int fixed_window;
    int nh;
    int pre;
    int win_floor;
};

DEV uint32_t cl_rank_a() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEV uint32_t cl_size_a() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r)); return r; }
DEV uint32_t sm_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sm_addr(dst)),
                 "l"(src), "r"(bytes), "r"(sm_addr(bar))
                 : "memory");
}

DEV int swz(int h, int dd) { return dd ^ ((h & 3) << 3); }

DEV bool au_alias(int R, int cs, int nh) { return (AH + R) * ALD * 2 + AH * R * 4 >= cs * (AH * 2 + nh * (AD / cs)) * 4; }

DEV void bulk_g2s_elect(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile(
        "{ .reg .pred P; elect.sync _|P, 0xffffffff;\n"
        "@P cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3]; }" ::"r"(sm_addr(dst)),
        "l"(src), "r"(bytes), "r"(sm_addr(bar))
        : "memory");
}
DEV void bulk_s2peer_elect(uint32_t rcv_local, uint32_t rbar_local, uint32_t q, uint32_t src, uint32_t bytes) {
    asm volatile(
        "{ .reg .pred P; .reg .b32 rd, rb; elect.sync _|P, 0xffffffff;\n"
        "mapa.shared::cluster.u32 rd, %0, %2;\n"
        "mapa.shared::cluster.u32 rb, %1, %2;\n"
        "@P cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [rd], [%3], %4, [rb]; }" ::"r"(rcv_local),
        "r"(rbar_local), "r"(q), "r"(src), "r"(bytes)
        : "memory");
}

DEV void attn_aud_body(const Meta* meta, const AttnFParams& p) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int R = p.R;
    bf16* qs = (bf16*)smem;
    bf16* kv = qs + AH * ALD;
    float* sc = (float*)(kv + R * ALD);
    bf16* ps = (bf16*)(sc + AH * R);
    const uint32_t cr = cl_rank_a(), cs = cl_size_a();
    const int DSp = AD / cs, slice = AH * 2 + p.nh * DSp;
    const bool alias = au_alias(R, (int)cs, p.nh);
    float* src = alias ? (float*)smem : (float*)(ps + AH * (R + 8));
    int* valid = alias ? (int*)(ps + AH * (R + 8)) : (int*)(src + cs * slice);
    float* rcv = (float*)(valid + ((R + 3) & ~3));
    __shared__ __align__(8) float2 stl[AH];
    __shared__ __align__(8) uint64_t rbar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sm_addr(&rbar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        const uint32_t bytes = cs * slice * 4;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sm_addr(&rbar)), "r"(bytes) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    const int m = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
    const int s0 = cr * R;
    const int nslots = 128 + p.ktop;
    __shared__ __align__(8) uint64_t bar;
    __shared__ int nvalid, npre;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(sm_addr(&bar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        nvalid = 0;
        npre = 0;
    }
    __syncthreads();
    const int pos0 = meta->pos, Mm = meta->m;
    const bool live = m < Mm;
    {
        const int pos = pos0 + m, wend = p.fixed_window ? pos0 - 1 : pos;
        auto early_row = [&](int i) -> const bf16* {
            const int s = s0 + i;
            if (s < 128) {
                const int ps_ = wend - 127 + s;
                if (ps_ < p.win_floor) return nullptr;
                return ((p.pre & 1) && ps_ < pos0) ? p.win + (size_t)ps_ * AD : (const bf16*)1;
            }
            if (s >= nslots) return nullptr;
            if (!(p.pre & 2)) return (const bf16*)1;
            const int j = p.topk[m * p.ktop + (s - 128)];
            return j >= 0 ? p.comp + (size_t)j * AD : nullptr;
        };
        for (int i = tid; i < R; i += 128) {
            const bf16* row = live && p.pre ? early_row(i) : (const bf16*)1;
            if (row == (const bf16*)1) { valid[i] = -1; continue; }
            valid[i] = row != nullptr;
            if (row) atomicAdd(&npre, 1);
            else for (int c = 0; c < AD; c += 8) *(uint4*)(kv + i * ALD + c) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (tid == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sm_addr(&bar)), "r"((uint32_t)npre * AD * 2) : "memory");
        __syncthreads();
        for (int i = tid; i < R; i += 128)
            if (valid[i] == 1) {
                const bf16* row = early_row(i);
                bulk_g2s(kv + i * ALD, row, AD * 2, &bar);
            }
    }
    float rc0 = 1.f, rs0 = 0.f, rc1 = 1.f, rs1 = 0.f;
    if (live) {
        const int d = cr * DSp + (tid % (DSp / 4)) * 4;
        if (d >= AD - 64) {
            const size_t o = (size_t)(pos0 + m) * 32 + (d - (AD - 64)) / 2;
            rc0 = p.cos[o]; rs0 = p.sin[o]; rc1 = p.cos[o + 1]; rs1 = p.sin[o + 1];
        }
    }
    pdl_wait();
    pdl_launch();
    if (!live) {
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        return;
    }
    const int pos = pos0 + m;
    const int wend = p.fixed_window ? pos0 - 1 : pos;
    for (int i = tid; i < R; i += 128) {
        if (valid[i] != -1) continue;
        const int s = s0 + i;
        const bf16* row = nullptr;
        if (s < 128) { if (wend - 127 + s >= p.win_floor) row = p.win + (size_t)(wend - 127 + s) * AD; }
        else if (s < nslots) { const int j = p.topk[m * p.ktop + (s - 128)]; if (j >= 0) row = p.comp + (size_t)j * AD; }
        valid[i] = row != nullptr ? 2 : 0;
        if (row) atomicAdd(&nvalid, 1);
        else for (int c = 0; c < AD; c += 8) *(uint4*)(kv + i * ALD + c) = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    if (tid == 0) {
        const uint32_t bytes = (uint32_t)nvalid * AD * 2 + p.nh * AD * 2;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sm_addr(&bar)), "r"(bytes) : "memory");
    }
    __syncthreads();
    if (warp == 0) {
        const bf16* qsrc = p.q + (size_t)m * p.nh * AD;
        for (int h = 0; h < p.nh; ++h) bulk_g2s_elect(qs + h * ALD, qsrc + (size_t)h * AD, AD * 2, &bar);
    }

    for (int i = tid; i < R; i += 128) {
        if (valid[i] != 2) continue;
        const int s = s0 + i;
        const bf16* row = s < 128 ? p.win + (size_t)(wend - 127 + s) * AD : p.comp + (size_t)p.topk[m * p.ktop + (s - 128)] * AD;
        bulk_g2s(kv + i * ALD, row, AD * 2, &bar);
    }
    asm volatile("{ .reg .pred P; W_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W_%=; }" ::"r"(sm_addr(&bar)) : "memory");
    const int g = lane >> 2, t = lane & 3;
    for (int nt = warp; nt < R / 8; nt += 4) {
        float c0[4] = {0, 0, 0, 0};
        for (int k0 = 0; k0 < AD; k0 += 16) {
            uint32_t a[4], b[2];
            ldsm_x4(a, qs + (lane % 16) * ALD + k0 + (lane / 16) * 8);
            ldsm_x2(b, kv + (nt * 8 + (lane % 8)) * ALD + k0 + ((lane / 8) % 2) * 8);
            mma_bf16(c0, a, b[0], b[1]);
        }
        const int n0 = nt * 8 + 2 * t;
        sc[g * R + n0] = valid[n0] ? c0[0] * p.scale : -INFINITY;
        sc[g * R + n0 + 1] = valid[n0 + 1] ? c0[1] * p.scale : -INFINITY;
        sc[(g + 8) * R + n0] = valid[n0] ? c0[2] * p.scale : -INFINITY;
        sc[(g + 8) * R + n0 + 1] = valid[n0 + 1] ? c0[3] * p.scale : -INFINITY;
    }

    __syncthreads();
    {
        const int h = tid / 8, j = tid % 8;
        float mx = -INFINITY;
        for (int n = j; n < R; n += 8) mx = fmaxf(mx, sc[h * R + n]);
        for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        float sum = 0.f;
        for (int n = j; n < R; n += 8) {
            const float e = mx == -INFINITY ? 0.f : expf(sc[h * R + n] - mx);
            ps[h * (R + 8) + n] = tobf(e);
            sum += e;
        }
        for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
        if (alias) {
            if (j == 0) stl[h] = make_float2(mx, sum);
        } else {
            for (int q = j; q < (int)cs; q += 8) *(float2*)(src + q * slice + h * 2) = make_float2(mx, sum);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
    }
    __syncthreads();
    {
        float acc[16][4];
        for (int i = 0; i < 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
        for (int k0 = 0; k0 < R; k0 += 16) {
            uint32_t a[4];
            ldsm_x4(a, ps + (lane % 16) * (R + 8) + k0 + (lane / 16) * 8);
            for (int nt = 0; nt < 16; ++nt) {
                uint32_t b[2];
                ldsm_x2_t(b, kv + (k0 + (lane % 16)) * ALD + warp * 128 + nt * 8);
                mma_bf16(acc[nt], a, b[0], b[1]);
            }
        }
        const int lg = __ffs(DSp) - 1;
        if (alias) __syncthreads();
        for (int nt = 0; nt < 16; ++nt) {
            const int d = warp * 128 + nt * 8 + 2 * t, q = d >> lg, dd = d & (DSp - 1);
            float* blk = src + q * slice + AH * 2;
            if (g < p.nh) *(float2*)(blk + g * DSp + swz(g, dd)) = make_float2(acc[nt][0], acc[nt][1]);
            if (g + 8 < p.nh) *(float2*)(blk + (g + 8) * DSp + swz(g + 8, dd)) = make_float2(acc[nt][2], acc[nt][3]);
        }
    }
    if (DSp <= 128) {
        const int lg = __ffs(DSp) - 1;
        const int q0 = (warp * 128) >> lg, q1 = (warp * 128 + 128) >> lg;
        if (alias)
            for (int q = q0; q < q1; ++q) src[q * slice + lane] = ((const float*)stl)[lane];
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        const uint32_t rl = sm_addr(rcv + cr * slice), bl = sm_addr(&rbar);
        for (int q = q0; q < q1; ++q) bulk_s2peer_elect(rl, bl, (uint32_t)q, sm_addr(src + q * slice), (uint32_t)slice * 4);
        asm volatile("{ .reg .pred P; W_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W_%=; }" ::"r"(sm_addr(&rbar)) : "memory");
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    } else {
        if (alias) {
            const int h = tid / 8, j = tid % 8;
            for (int q = j; q < (int)cs; q += 8) *(float2*)(src + q * slice + h * 2) = stl[h];
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (tid < (int)cs) {
            const uint32_t q = tid;
            uint32_t rd, rb;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd) : "r"(sm_addr(rcv + cr * slice)), "r"(q));
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(sm_addr(&rbar)), "r"(q));
            asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(rd),
                         "r"(sm_addr(src + q * slice)), "r"((uint32_t)slice * 4), "r"(rb)
                         : "memory");
        }
        asm volatile("{ .reg .pred P; W_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W_%=; }" ::"r"(sm_addr(&rbar)) : "memory");
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    }
    const int DS = DSp, d0 = cr * DS;
    __shared__ float wgt[16][16];
    __shared__ float invL[16];
    {
        const int h = tid / 8, r0 = (tid % 8) * 2;
        float mx[2], sm[2];
        for (int q = 0; q < 2; ++q) {
            const int r = r0 + q;
            if (r < (int)cs) { mx[q] = rcv[r * slice + h * 2]; sm[q] = rcv[r * slice + h * 2 + 1]; }
            else { mx[q] = -INFINITY; sm[q] = 0.f; }
        }
        const float sink = h < p.nh ? p.sink[h] : 0.f;
        float M = fmaxf(fmaxf(mx[0], mx[1]), sink);
        for (int o = 1; o < 8; o <<= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
        float L = 0.f;
        for (int q = 0; q < 2; ++q) {
            const int r = r0 + q;
            const float f = mx[q] == -INFINITY ? 0.f : expf(mx[q] - M);
            if (r < 16) wgt[h][r] = f;
            L += f * sm[q];
        }
        for (int o = 1; o < 8; o <<= 1) L += __shfl_xor_sync(0xffffffff, L, o);
        if ((tid % 8) == 0) invL[h] = 1.f / (L + expf(sink - M));
    }
    __syncthreads();
    for (int i = tid; i < p.nh * DS / 4; i += 128) {
        const int h = i / (DS / 4), dl = (i % (DS / 4)) * 4, d = d0 + dl;
        const int off = AH * 2 + h * DS + swz(h, dl);
        float a[4] = {0.f, 0.f, 0.f, 0.f};
        for (int r = 0; r < (int)cs; ++r) {
            const float4 v = *(const float4*)(rcv + (size_t)r * slice + off);
            const float f = wgt[h][r];
            a[0] += f * v.x; a[1] += f * v.y; a[2] += f * v.z; a[3] += f * v.w;
        }
        for (int k = 0; k < 4; ++k) a[k] = bf(tobf(a[k] * invL[h]));
        if (d >= AD - 64) {
            const float x0 = a[0] * rc0 + a[1] * rs0, x1 = a[1] * rc0 - a[0] * rs0;
            const float x2 = a[2] * rc1 + a[3] * rs1, x3 = a[3] * rc1 - a[2] * rs1;
            a[0] = x0; a[1] = x1; a[2] = x2; a[3] = x3;
        }
        const __nv_bfloat162 o01 = __halves2bfloat162(tobf(a[0]), tobf(a[1])), o23 = __halves2bfloat162(tobf(a[2]), tobf(a[3]));
        *(uint2*)(p.out + ((size_t)m * p.nh + h) * AD + d) = make_uint2(*(const uint32_t*)&o01, *(const uint32_t*)&o23);
    }
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

extern "C" __global__ void __launch_bounds__(128) attn_aud_mqx(const Meta* meta, const __grid_constant__ AttnFParams p) { attn_aud_body(meta, p); }
