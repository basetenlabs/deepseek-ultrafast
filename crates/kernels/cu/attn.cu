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

constexpr int AH = 16, AD = 512, ACH = 64, ALD = AD + 8;

struct IdxParams {
    const bf16* q;
    const bf16* w;
    int ldw;
    const bf16* kc;
    float* scores;
    int ld;
    int ratio;
    int tmax;
    const uint8_t* cand;
    int ld_cand, cand_block;
    int bf16_ref;
};

extern "C" __global__ void __launch_bounds__(128) idx_score(const Meta* meta, const IdxParams p) {
    __shared__ __align__(16) bf16 qs[32][136];
    __shared__ __align__(16) bf16 ks[128][136];
    __shared__ float ws[32];
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.y;
    if (m >= meta->m) return;
    const int T = (meta->pos + m + 1) / p.ratio;
    const int tmax = (meta->pos + meta->m) / p.ratio;
    if ((int)blockIdx.x * 128 >= tmax) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < 32 * 16; i += 128) {
        const int h = i / 16, c = (i % 16) * 8;
        *(uint4*)&qs[h][c] = *(const uint4*)(p.q + ((size_t)m * 32 + h) * 128 + c);
    }
    if (tid < 32) ws[tid] = bf(p.w[(size_t)m * p.ldw + tid]);
    for (int t0 = blockIdx.x * 128; t0 < tmax; t0 += gridDim.x * 128) {
    __syncthreads();
    for (int i = tid; i < 128 * 16; i += 128) {
        const int r = i / 16, c = (i % 16) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (t0 + r < T) v = *(const uint4*)(p.kc + (size_t)(t0 + r) * 128 + c);
        *(uint4*)&ks[r][c] = v;
    }
    __syncthreads();
    float acc[2][4][4];
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 4; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
    for (int k0 = 0; k0 < 128; k0 += 16) {
        uint32_t a0[4], a1[4];
        ldsm_x4(a0, &qs[lane % 16][k0 + (lane / 16) * 8]);
        ldsm_x4(a1, &qs[16 + lane % 16][k0 + (lane / 16) * 8]);
        for (int nt = 0; nt < 4; ++nt) {
            uint32_t b[2];
            ldsm_x2(b, &ks[warp * 32 + nt * 8 + (lane % 8)][k0 + ((lane / 8) % 2) * 8]);
            mma_bf16(acc[0][nt], a0, b[0], b[1]);
            mma_bf16(acc[1][nt], a1, b[0], b[1]);
        }
    }
    const int g = lane >> 2, t = lane & 3;
    auto rb = [](float v) { return __bfloat162float(__float2bfloat16(v)); };
    for (int nt = 0; nt < 4; ++nt) {
        float s0 = 0.f, s1 = 0.f;
        if (p.bf16_ref) {
            float g0[4], g1[4];
            for (int q = 0; q < 4; ++q) {
                const int mt = q / 2, hf = q % 2, h = 8 * q + g;
                const float a0 = acc[mt][nt][hf * 2], a1 = acc[mt][nt][hf * 2 + 1];
                g0[q] = rb(fmaxf(rb(a0), 0.f) * ws[h]);
                g1[q] = rb(fmaxf(rb(a1), 0.f) * ws[h]);
                for (int o = 4; o < 32; o <<= 1) {
                    g0[q] += __shfl_xor_sync(0xffffffff, g0[q], o);
                    g1[q] += __shfl_xor_sync(0xffffffff, g1[q], o);
                }
                g0[q] = rb(g0[q]);
                g1[q] = rb(g1[q]);
            }
            s0 = rb(rb(rb(g0[0] + g0[1]) + g0[2]) + g0[3]);
            s1 = rb(rb(rb(g1[0] + g1[1]) + g1[2]) + g1[3]);
        } else {
            for (int mt = 0; mt < 2; ++mt) {
                const int h0 = mt * 16 + g, h1 = h0 + 8;
                s0 += fmaxf(acc[mt][nt][0], 0.f) * ws[h0] + fmaxf(acc[mt][nt][2], 0.f) * ws[h1];
                s1 += fmaxf(acc[mt][nt][1], 0.f) * ws[h0] + fmaxf(acc[mt][nt][3], 0.f) * ws[h1];
            }
            for (int o = 4; o < 32; o <<= 1) {
                s0 += __shfl_xor_sync(0xffffffff, s0, o);
                s1 += __shfl_xor_sync(0xffffffff, s1, o);
            }
        }
        if (g == 0) {
            const int pos0 = t0 + warp * 32 + nt * 8 + 2 * t;
            float v[2] = {s0, s1};
            for (int i = 0; i < 2; ++i) {
                const int pp = pos0 + i;
                if (pp >= tmax) continue;
                bool ok = pp < T;
                if (ok && p.cand) ok = p.cand[(size_t)m * p.ld_cand + pp / p.cand_block] != 0;
                p.scores[(size_t)m * p.ld + pp] = ok ? v[i] : -INFINITY;
            }
        }
    }
    }
}

extern "C" __global__ void block_max(const Meta* meta, const float* scores, int ld, int ratio, int bs, float* out,
                                     int ldo) {
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.y;
    if (m >= meta->m) return;
    const int T = (meta->pos + m + 1) / ratio;
    const int nb = (T + bs - 1) / bs;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
        float mx = -INFINITY;
        for (int i = b * bs; i < min(T, (b + 1) * bs); ++i) mx = fmaxf(mx, scores[(size_t)m * ld + i]);
        if (b == (T - 1) / bs) mx = INFINITY;
        out[(size_t)m * ldo + b] = mx;
    }
}

DEV uint32_t fkey(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

struct TopkParams {
    const float* x;
    int ld;
    int ratio;
    int block;
    int k;
    int* out;
    uint8_t* mask;
    int ld_mask;
    int drop_neg_inf;
};

DEV void topk_select_row(const TopkParams& p, int m, int n) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_need;
    __shared__ int wsum[32];
    __shared__ int s_base, s_eqbase;
    const float* x = p.x + (size_t)m * p.ld;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int* out = p.out + (size_t)m * p.k;
    const int k = min(p.k, n);
    if (tid == 0) { s_prefix = 0; s_need = k; }
    __syncthreads();
    uint32_t prefix_mask = 0;
    for (int shift = 24; shift >= 0 && k < n; shift -= 8) {
        for (int i = tid; i < 256; i += (int)blockDim.x) hist[i] = 0;
        __syncthreads();
        const uint32_t pre = s_prefix;
        for (int i = tid; i < n; i += (int)blockDim.x) {
            const uint32_t key = fkey(x[i]);
            if ((key & prefix_mask) == pre) atomicAdd(&hist[(key >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t need = s_need, cum = 0;
            int d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= need) break;
                cum += hist[d];
            }
            s_prefix = pre | ((uint32_t)d << shift);
            s_need = need - cum;
        }
        prefix_mask |= 255u << shift;
        __syncthreads();
    }
    const bool all = k >= n;
    const uint32_t kth = s_prefix;
    const int need_eq = s_need;
    if (tid == 0) { s_base = 0; s_eqbase = 0; }
    __syncthreads();
    for (int i0 = 0; i0 < n; i0 += (int)blockDim.x) {
        const int i = i0 + tid;
        bool gt = false, eq = false;
        if (i < n) {
            if (all) gt = true;
            else {
                const uint32_t key = fkey(x[i]);
                gt = key > kth;
                eq = key == kth;
            }
        }
        const uint32_t eb = __ballot_sync(0xffffffff, eq);
        if (lane == 0) wsum[warp] = __popc(eb);
        __syncthreads();
        int eqoff = s_eqbase;
        for (int w = 0; w < warp; ++w) eqoff += wsum[w];
        eqoff += __popc(eb & ((1u << lane) - 1));
        __syncthreads();
        if (tid == (int)blockDim.x - 1) s_eqbase = eqoff + (eq ? 1 : 0);
        bool take = gt || (eq && eqoff < need_eq);
        if (take && p.drop_neg_inf && x[i] == -INFINITY) take = false;
        const uint32_t tb = __ballot_sync(0xffffffff, take);
        if (lane == 0) wsum[warp] = __popc(tb);
        __syncthreads();
        int off = s_base;
        for (int w = 0; w < warp; ++w) off += wsum[w];
        off += __popc(tb & ((1u << lane) - 1));
        if (take) {
            out[off] = i;
            if (p.mask) p.mask[(size_t)m * p.ld_mask + i] = 1;
        }
        __syncthreads();
        if (tid == (int)blockDim.x - 1) s_base = off + (take ? 1 : 0);
        __syncthreads();
    }
    const int got = s_base;
    for (int i = got + tid; i < p.k; i += (int)blockDim.x) out[i] = -1;
}

extern "C" __global__ void __launch_bounds__(1024) topk_select(const Meta* meta, const TopkParams p) {
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.x;
    if (m >= meta->m) return;
    int n = (meta->pos + m + 1) / p.ratio;
    if (p.block > 1) n = (n + p.block - 1) / p.block;
    if (p.mask)
        for (int i = threadIdx.x; i < p.ld_mask; i += 1024) p.mask[(size_t)m * p.ld_mask + i] = 0;
    __syncthreads();
    topk_select_row(p, m, n);
}

extern "C" __global__ void __launch_bounds__(128) compress_pool(const Meta* meta, const float* kvh,
                                                               const float* sch, int ratio, const bf16* w, float eps,
                                                               bf16* latent, int dim) {
    __shared__ float sh[4];
    __shared__ bf16 tmp[512];
    constexpr int RMAX = 4, DPT = 4;
    const int m = blockIdx.x;
    const int pos0 = meta->pos, pos = pos0 + m;
    const bool act = m < meta->m && (pos + 1) % ratio == 0;
    const int g0 = pos + 1 - ratio;
    float sv[DPT][RMAX], kv[DPT][RMAX], wv[DPT];
#pragma unroll
    for (int k = 0; k < DPT; ++k) {
        const int d = threadIdx.x + k * 128;
        wv[k] = act && d < dim ? bf(w[d]) : 0.f;
#pragma unroll
        for (int i = 0; i < RMAX; ++i) {
            const bool on = act && d < dim && i < ratio && g0 + i < pos0;
            sv[k][i] = on ? sch[(size_t)(g0 + i) * dim + d] : -INFINITY;
            kv[k][i] = on ? kvh[(size_t)(g0 + i) * dim + d] : 0.f;
        }
    }
    pdl_wait();
    pdl_launch();
    if (!act) return;
#pragma unroll
    for (int k = 0; k < DPT; ++k)
#pragma unroll
        for (int i = 0; i < RMAX; ++i) {
            const int d = threadIdx.x + k * 128;
            if (d < dim && i < ratio && g0 + i >= pos0) {
                sv[k][i] = sch[(size_t)(g0 + i) * dim + d];
                kv[k][i] = kvh[(size_t)(g0 + i) * dim + d];
            }
        }
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < DPT; ++k) {
        const int d = threadIdx.x + k * 128;
        if (d < dim) {
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < RMAX; ++i)
                if (i < ratio) mx = fmaxf(mx, sv[k][i]);
            float den = 0.f, num = 0.f;
#pragma unroll
            for (int i = 0; i < RMAX; ++i)
                if (i < ratio) den += expf(sv[k][i] - mx);
#pragma unroll
            for (int i = 0; i < RMAX; ++i)
                if (i < ratio) num += kv[k][i] * (expf(sv[k][i] - mx) / den);
            const bf16 b = tobf(num);
            tmp[d] = b;
            ss += bf(b) * bf(b);
        }
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    ss = sh[0] + sh[1] + sh[2] + sh[3];
    const float r = rsqrtf(ss / dim + eps);
#pragma unroll
    for (int k = 0; k < DPT; ++k) {
        const int d = threadIdx.x + k * 128;
        if (d < dim) latent[(size_t)m * dim + d] = tobf(wv[k] * (bf(tmp[d]) * r));
    }
}

extern "C" __global__ void store_hist_f32(const Meta* meta, const float* src, int ld, float* hist, int n) {
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.y;
    if (m >= meta->m) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hist[(size_t)(meta->pos + m) * n + i] = src[(size_t)m * ld + i];
}

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

DEV void attn_fused_body(const Meta* meta, const AttnFParams& p) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int R = p.R;
    const int hg = blockIdx.z * AH, nh = min(AH, p.nh - hg), nhs = p.nh;
    bf16* qs = (bf16*)smem;
    bf16* kv = qs + AH * ALD;
    float* sc = (float*)(kv + R * ALD);
    bf16* ps = (bf16*)(sc + AH * R);
    float* op = (float*)(ps + AH * (R + 8));
    float* ml = op + nh * AD;
    int* valid = (int*)(ml + AH * 2);
    const uint32_t cr = cl_rank_a(), cs = cl_size_a();
    const int DSp = AD / cs;
    float* rml = (float*)(valid + ((R + 3) & ~3));
    float* rop = rml + cs * AH * 2;
    __shared__ __align__(8) uint64_t rbar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sm_addr(&rbar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        const uint32_t bytes = cs * (AH * 2 + nh * DSp) * 4;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sm_addr(&rbar)), "r"(bytes) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    const int m = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
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
        const uint32_t bytes = (uint32_t)nvalid * AD * 2 + nh * AD * 2;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sm_addr(&bar)), "r"(bytes) : "memory");
        for (int h = 0; h < nh; ++h) bulk_g2s(qs + h * ALD, p.q + ((size_t)m * nhs + hg + h) * AD, AD * 2, &bar);
    }
    __syncthreads();
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
        if (j == 0) { ml[h * 2] = mx; ml[h * 2 + 1] = sum; }
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
        for (int nt = 0; nt < 16; ++nt) {
            const int d = warp * 128 + nt * 8 + 2 * t;
            if (g < nh) *(float2*)(op + g * AD + d) = make_float2(acc[nt][0], acc[nt][1]);
            if (g + 8 < nh) *(float2*)(op + (g + 8) * AD + d) = make_float2(acc[nt][2], acc[nt][3]);
        }
    }
    {
        __syncthreads();
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        const int nv_ml = AH * 2 / 4, nv_op = nh * DSp / 4, nv = nv_ml + nv_op;
        for (int i = threadIdx.x; i < (int)cs * nv; i += 128) {
            const int q = i / nv, j = i - q * nv;
            float4 v;
            uint32_t dst;
            if (j < nv_ml) {
                v = *(const float4*)(ml + j * 4);
                dst = (uint32_t)__cvta_generic_to_shared(rml + ((size_t)cr * AH * 2 + j * 4));
            } else {
                const int jj = j - nv_ml, h = jj / (DSp / 4), d4 = jj - h * (DSp / 4);
                v = *(const float4*)(op + h * AD + q * DSp + d4 * 4);
                dst = (uint32_t)__cvta_generic_to_shared(rop + ((size_t)cr * nh * DSp + h * DSp + d4 * 4));
            }
            uint32_t ra, rb;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(dst), "r"((uint32_t)q));
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(sm_addr(&rbar)), "r"((uint32_t)q));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(ra), "f"(v.x), "f"(v.y),
                         "f"(v.z), "f"(v.w), "r"(rb)
                         : "memory");
        }
        asm volatile("{ .reg .pred P; W_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W_%=; }" ::"r"(sm_addr(&rbar)) : "memory");
    }
    const int DS = DSp, d0 = cr * DS;
    __shared__ float wgt[16][16];
    __shared__ float invL[16];
    {
        const int h = tid / 8, r0 = (tid % 8) * 2;
        float mx[2], sm[2];
        for (int q = 0; q < 2; ++q) {
            const int r = r0 + q;
            if (r < (int)cs) { mx[q] = rml[(r * AH + h) * 2]; sm[q] = rml[(r * AH + h) * 2 + 1]; }
            else { mx[q] = -INFINITY; sm[q] = 0.f; }
        }
        const float sink = h < nh ? p.sink[hg + h] : 0.f;
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
    for (int i = tid; i < nh * DS / 4; i += 128) {
        const int h = i / (DS / 4), dl = (i % (DS / 4)) * 4, d = d0 + dl;
        float a[4] = {0.f, 0.f, 0.f, 0.f};
        for (int r = 0; r < (int)cs; ++r) {
            const float4 v = *(const float4*)(rop + ((size_t)r * nh * DS + h * DS + dl));
            const float f = wgt[h][r];
            a[0] += f * v.x; a[1] += f * v.y; a[2] += f * v.z; a[3] += f * v.w;
        }
        for (int k = 0; k < 4; ++k) a[k] = bf(tobf(a[k] * invL[h]));
        if (d >= AD - 64) {
            for (int k = 0; k < 4; k += 2) {
                const int f = (d + k - (AD - 64)) / 2;
                const float c = p.cos[(size_t)pos * 32 + f], sn = p.sin[(size_t)pos * 32 + f];
                const float x0 = a[k] * c + a[k + 1] * sn, x1 = a[k + 1] * c - a[k] * sn;
                a[k] = x0; a[k + 1] = x1;
            }
        }
        bf16* o = p.out + ((size_t)m * nhs + hg + h) * AD + d;
        for (int k = 0; k < 4; ++k) o[k] = tobf(a[k]);
    }
}
extern "C" __global__ void __launch_bounds__(128) attn_fused(const Meta* meta, const __grid_constant__ AttnFParams p) { attn_fused_body(meta, p); }
extern "C" __global__ void __launch_bounds__(128) attn_fused_hg(const Meta* meta, const __grid_constant__ AttnFParams p) { attn_fused_body(meta, p); }

DEV void cp_async16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm_addr(dst)), "l"(src), "r"(ok ? 16 : 0) : "memory");
}
DEV void attn_prefill_body(const Meta* meta, const AttnFParams& p) {
    constexpr int PC = 16, NB = 2;
    extern __shared__ __align__(16) uint8_t smem[];
    bf16* qs = (bf16*)smem;
    bf16* kvb = qs;
    float* sc = (float*)(kvb + NB * PC * ALD);
    bf16* ps = (bf16*)(sc + 4 * AH * PC);
    __shared__ __align__(8) uint64_t bar[NB];
    __shared__ int valid[NB][PC];
    __shared__ float alpha_s[AH], m_s[AH], l_s[AH];
    __shared__ int tk_s[1024];
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.x;
    if (m >= meta->m) return;
    const int hg = blockIdx.z * AH, nh = min(AH, p.nh - hg), nhs = p.nh;
    const int pos = meta->pos + m, wend = pos;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nslots = 128 + p.ktop, nch = (nslots + PC - 1) / PC;
    for (int i = tid; i < p.ktop; i += 128) tk_s[i] = p.topk[m * p.ktop + i];
    if (tid == 0) {
        for (int i = 0; i < NB; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sm_addr(&bar[i])), "r"(128));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (tid < AH) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
    __syncthreads();
    auto issue = [&](int c) {
        const int b = c % NB;
        constexpr int RW = PC / 4;
        const bf16* rp = nullptr;
        if (lane < RW) {
            const int s = c * PC + warp * RW + lane;
            if (s < 128) { if (wend - 127 + s >= p.win_floor) rp = p.win + (size_t)(wend - 127 + s) * AD; }
            else if (s < nslots) { const int j = tk_s[s - 128]; if (j >= 0) rp = p.comp + (size_t)j * AD; }
            valid[b][warp * RW + lane] = rp != nullptr;
        }
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            const bf16* r = (const bf16*)__shfl_sync(0xffffffffu, (unsigned long long)rp, i);
            bf16* dst = kvb + ((size_t)b * PC + warp * RW + i) * ALD + lane * 8;
            const bf16* src = r ? r + lane * 8 : p.q;
            cp_async16(dst, src, r != nullptr);
            cp_async16(dst + 256, src + (r ? 256 : 0), r != nullptr);
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(sm_addr(&bar[b])) : "memory");
    };
    for (int c = 0; c < NB - 1 && c < nch; ++c) issue(c);
    __syncthreads();
    const int g = lane >> 2, t = lane & 3;
    float acc[16][4];
    for (int i = 0; i < 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    uint32_t qa[8][4];
    const bf16* q0 = p.q + ((size_t)m * nhs + hg + g) * AD + warp * 128 + 2 * t;
    const bf16* q1 = p.q + ((size_t)m * nhs + hg + g + 8) * AD + warp * 128 + 2 * t;
    const bool h0 = g < nh, h1 = g + 8 < nh;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        qa[kk][0] = h0 ? *(const uint32_t*)(q0 + kk * 16) : 0u;
        qa[kk][1] = h1 ? *(const uint32_t*)(q1 + kk * 16) : 0u;
        qa[kk][2] = h0 ? *(const uint32_t*)(q0 + kk * 16 + 8) : 0u;
        qa[kk][3] = h1 ? *(const uint32_t*)(q1 + kk * 16 + 8) : 0u;
    }
    for (int c = 0; c < nch; ++c) {
        const int b = c % NB;
        if (c + NB - 1 < nch) issue(c + NB - 1);
        asm volatile("{ .reg .pred P; W_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W_%=; }" ::"r"(sm_addr(&bar[b])), "r"((c / NB) & 1) : "memory");

        const bf16* kv = kvb + (size_t)b * PC * ALD;
        float s4[PC / 8][4];
#pragma unroll
        for (int nt = 0; nt < PC / 8; ++nt) s4[nt][0] = s4[nt][1] = s4[nt][2] = s4[nt][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
#pragma unroll
            for (int nt = 0; nt < PC / 8; ++nt) {
                uint32_t bb[2];
                ldsm_x2(bb, kv + (nt * 8 + (lane % 8)) * ALD + warp * 128 + kk * 16 + ((lane / 8) % 2) * 8);
                mma_bf16(s4[nt], qa[kk], bb[0], bb[1]);
            }
        float* sp = sc + warp * AH * PC;
#pragma unroll
        for (int nt = 0; nt < PC / 8; ++nt) {
            const int n0 = nt * 8 + 2 * t;
            *(float2*)(sp + g * PC + n0) = make_float2(s4[nt][0], s4[nt][1]);
            *(float2*)(sp + (g + 8) * PC + n0) = make_float2(s4[nt][2], s4[nt][3]);
        }
        __syncthreads();
        {
            const int h = tid / 8, j = tid % 8;
            constexpr int SPT = PC / 8;
            float sv[SPT];
#pragma unroll
            for (int q = 0; q < SPT; ++q) {
                const int n = j + 8 * q;
                const float v = ((sc[h * PC + n] + sc[AH * PC + h * PC + n]) + sc[2 * AH * PC + h * PC + n]) + sc[3 * AH * PC + h * PC + n];
                sv[q] = valid[b][n] ? v * p.scale : -INFINITY;
            }
            const float mo = m_s[h];
            float mx = -INFINITY;
#pragma unroll
            for (int q = 0; q < SPT; ++q) mx = fmaxf(mx, sv[q]);
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
            const float mn = fmaxf(mo, mx);
            float sum = 0.f;
#pragma unroll
            for (int q = 0; q < SPT; ++q) {
                const float e = mn == -INFINITY ? 0.f : expf(sv[q] - mn);
                ps[h * (PC + 8) + j + 8 * q] = tobf(e);
                sum += e;
            }
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
            __syncwarp();
            if (j == 0) {
                const float al = mo == -INFINITY ? 0.f : expf(mo - mn);
                alpha_s[h] = al;
                l_s[h] = l_s[h] * al + sum;
                m_s[h] = mn;
            }
        }
        __syncthreads();
        {
            const float a0 = alpha_s[g], a1 = alpha_s[g + 8];
            for (int nt = 0; nt < 16; ++nt) { acc[nt][0] *= a0; acc[nt][1] *= a0; acc[nt][2] *= a1; acc[nt][3] *= a1; }
            for (int k0 = 0; k0 < PC; k0 += 16) {
                uint32_t a[4];
                ldsm_x4(a, ps + (lane % 16) * (PC + 8) + k0 + (lane / 16) * 8);
                for (int nt = 0; nt < 16; ++nt) {
                    uint32_t bb[2];
                    ldsm_x2_t(bb, kv + (k0 + (lane % 16)) * ALD + warp * 128 + nt * 8);
                    mma_bf16(acc[nt], a, bb[0], bb[1]);
                }
            }
        }
        __syncthreads();
    }
    for (int half = 0; half < 2; ++half) {
        const int h = g + 8 * half;
        if (h >= nh) continue;
        const float sink = p.sink[hg + h], mo = m_s[h], M = fmaxf(mo, sink);
        const float f = mo == -INFINITY ? 0.f : expf(mo - M);
        const float invL = 1.f / (l_s[h] * f + expf(sink - M));
        for (int nt = 0; nt < 16; ++nt) {
            const int d = warp * 128 + nt * 8 + 2 * t;
            float a[2] = {bf(tobf(acc[nt][2 * half] * f * invL)), bf(tobf(acc[nt][2 * half + 1] * f * invL))};
            if (d >= AD - 64) {
                const int fi = (d - (AD - 64)) / 2;
                const float cs = p.cos[(size_t)pos * 32 + fi], sn = p.sin[(size_t)pos * 32 + fi];
                const float x0 = a[0] * cs + a[1] * sn, x1 = a[1] * cs - a[0] * sn;
                a[0] = x0; a[1] = x1;
            }
            bf16* o = p.out + ((size_t)m * nhs + hg + h) * AD + d;
            o[0] = tobf(a[0]);
            o[1] = tobf(a[1]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) attn_prefill_ks16(const Meta* meta, const __grid_constant__ AttnFParams p) { attn_prefill_body(meta, p); }

constexpr int TK_S = 4096;
struct TopkPartParams {
    TopkParams t;
    int nsplit;
    int cap;
    float* cv;
    int* ci;
    int* cn;
    uint32_t* kth;
};

DEV void radix_kth(const uint32_t* keys, int n, int k, uint32_t* hist, uint32_t* s_prefix, uint32_t* s_need) {
    const int tid = threadIdx.x;
    if (tid == 0) { *s_prefix = 0; *s_need = k; }
    __syncthreads();
    uint32_t prefix_mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const uint32_t pre = *s_prefix;
        for (int i = tid; i < n; i += blockDim.x) {
            const uint32_t key = keys[i];
            if ((key & prefix_mask) == pre) atomicAdd(&hist[(key >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid < 32) {
            const int lane = tid;
            uint32_t c[8], sum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { c[j] = hist[255 - 8 * lane - j]; sum += c[j]; }
            uint32_t incl = sum;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            const uint32_t need = *s_need, excl = incl - sum;
            const unsigned hit = __ballot_sync(0xffffffffu, incl >= need);
            const int L = __ffs(hit) - 1;
            __syncwarp();
            if (lane == L) {
                uint32_t cum = excl;
                int d = 255 - 8 * lane;
                for (int j = 0; j < 7; ++j, --d) {
                    if (cum + c[j] >= need) break;
                    cum += c[j];
                }
                *s_prefix = pre | ((uint32_t)d << shift);
                *s_need = need - cum;
            }
        }
        prefix_mask |= 255u << shift;
        __syncthreads();
    }
}

DEV int topk_n(const Meta* meta, const TopkParams& t, int m) {
    int n = (meta->pos + m + 1) / t.ratio;
    if (t.block > 1) n = (n + t.block - 1) / t.block;
    return n;
}

extern "C" __global__ void __launch_bounds__(512) topk_kth(const Meta* meta, const __grid_constant__ TopkPartParams p) {
    __shared__ uint32_t keys[TK_S];
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_need;
    pdl_wait();
    pdl_launch();
    const int s = blockIdx.x, m = blockIdx.y;
    if (m >= meta->m) return;
    const TopkParams& t = p.t;
    const int n = topk_n(meta, t, m);
    const int S = (n + p.nsplit - 1) / p.nsplit;
    const int b0 = s * S, b1 = min(n, b0 + S), ns = max(0, b1 - b0);
    if (t.mask) {
        const int L = (t.ld_mask + gridDim.x - 1) / gridDim.x;
        for (int i = s * L + threadIdx.x; i < min(t.ld_mask, (s + 1) * L); i += blockDim.x) t.mask[(size_t)m * t.ld_mask + i] = 0;
    }
    if (ns <= t.k) {
        if (threadIdx.x == 0) p.kth[(size_t)m * p.nsplit + s] = 0u;
        return;
    }
    const float* x = t.x + (size_t)m * t.ld + b0;
    for (int i = threadIdx.x; i < ns; i += blockDim.x) keys[i] = fkey(x[i]);
    __syncthreads();
    radix_kth(keys, ns, t.k, hist, &s_prefix, &s_need);
    if (threadIdx.x == 0) p.kth[(size_t)m * p.nsplit + s] = s_prefix;
}

extern "C" __global__ void __launch_bounds__(512) topk_part(const Meta* meta, const __grid_constant__ TopkPartParams p) {
    __shared__ int wsum[32], s_base, s_eqbase;
    __shared__ uint32_t s_L;
    pdl_wait();
    pdl_launch();
    const int s = blockIdx.x, m = blockIdx.y;
    if (m >= meta->m) return;
    const TopkParams& t = p.t;
    const int n = topk_n(meta, t, m);
    const int S = (n + p.nsplit - 1) / p.nsplit;
    const int b0 = s * S, b1 = min(n, b0 + S), ns = max(0, b1 - b0);
    if (threadIdx.x < 32) {
        uint32_t L = 0;
        for (int j = threadIdx.x; j < p.nsplit; j += 32) L = max(L, p.kth[(size_t)m * p.nsplit + j]);
        L = __reduce_max_sync(0xffffffffu, L);
        if (threadIdx.x == 0) s_L = L;
    }
    __syncthreads();
    const uint32_t L = s_L;
    const float* x = t.x + (size_t)m * t.ld + b0;
    float* cv = p.cv + ((size_t)m * p.nsplit + s) * p.cap;
    int* ci = p.ci + ((size_t)m * p.nsplit + s) * p.cap;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) s_base = 0;
    __syncthreads();
    for (int i0 = 0; i0 < ns; i0 += blockDim.x) {
        const int i = i0 + tid;
        const float v = i < ns ? x[i] : 0.f;
        const int base = s_base;
        bool take = i < ns && fkey(v) >= L;
        const uint32_t tb = __ballot_sync(0xffffffff, take);
        if (lane == 0) wsum[warp] = __popc(tb);
        __syncthreads();
        int off = base;
        for (int w = 0; w < warp; ++w) off += wsum[w];
        off += __popc(tb & ((1u << lane) - 1));
        if (take && off < p.cap) { cv[off] = v; ci[off] = b0 + i; }
        __syncthreads();
        if (tid == blockDim.x - 1) s_base = off + (take ? 1 : 0);
        __syncthreads();
        if (s_base >= p.cap) break;
    }
    if (tid == 0) p.cn[(size_t)m * p.nsplit + s] = min(s_base, p.cap);
}

extern "C" __global__ void __launch_bounds__(1024) topk_merge(const Meta* meta, const __grid_constant__ TopkPartParams p) {
    extern __shared__ uint32_t mkeys[];
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_need;
    __shared__ int wsum[32], s_base, s_eqbase, s_off[129];
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.x;
    if (m >= meta->m) return;
    const TopkParams& t = p.t;
    int* midx = (int*)(mkeys + p.nsplit * p.cap);
    if (threadIdx.x < 64) {
        const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
        const int sl = threadIdx.x;
        const int c = sl < p.nsplit ? p.cn[(size_t)m * p.nsplit + sl] : 0;
        int incl = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += v;
        }
        if (lane == 31) wsum[w] = incl;
        s_off[sl + 1] = incl;
        if (sl == 0) s_off[0] = 0;
    }
    __syncthreads();
    if (threadIdx.x >= 32 && threadIdx.x < 64) s_off[threadIdx.x + 1] += wsum[0];
    __syncthreads();
    const int nc = s_off[p.nsplit];
    for (int j = threadIdx.x; j < nc; j += blockDim.x) {
        int lo = 0, hi = p.nsplit - 1;
        while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_off[mid] <= j) lo = mid; else hi = mid - 1; }
        const int i = j - s_off[lo];
        const size_t o = ((size_t)m * p.nsplit + lo) * p.cap + i;
        mkeys[j] = fkey(p.cv[o]);
        midx[j] = p.ci[o];
    }
    __syncthreads();
    const int n = topk_n(meta, t, m);
    const int k = min(t.k, n);
    const bool all = k >= nc;
    if (!all) radix_kth(mkeys, nc, k, hist, &s_prefix, &s_need);
    const uint32_t kth = all ? 0u : s_prefix;
    const int need_eq = all ? 0 : (int)s_need;
    int* out = t.out + (size_t)m * t.k;
    const uint32_t ninf = fkey(-INFINITY);
    if (threadIdx.x == 0) { s_base = 0; s_eqbase = 0; }
    __syncthreads();
    {
        const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        for (int i0 = 0; i0 < nc; i0 += blockDim.x) {
            const int i = i0 + tid;
            bool gt = false, eq = false;
            if (i < nc) {
                if (all) gt = true;
                else { const uint32_t key = mkeys[i]; gt = key > kth; eq = key == kth; }
            }
            const uint32_t eb = __ballot_sync(0xffffffff, eq);
            if (lane == 0) wsum[warp] = __popc(eb);
            __syncthreads();
            int eqoff = s_eqbase;
            for (int w = 0; w < warp; ++w) eqoff += wsum[w];
            eqoff += __popc(eb & ((1u << lane) - 1));
            __syncthreads();
            if (tid == (int)blockDim.x - 1) s_eqbase = eqoff + (eq ? 1 : 0);
            bool take = gt || (eq && eqoff < need_eq);
            if (take && t.drop_neg_inf && mkeys[i] == ninf) take = false;
            const uint32_t tb = __ballot_sync(0xffffffff, take);
            if (lane == 0) wsum[warp] = __popc(tb);
            __syncthreads();
            int off = s_base;
            for (int w = 0; w < warp; ++w) off += wsum[w];
            off += __popc(tb & ((1u << lane) - 1));
            if (take) {
                out[off] = midx[i];
                if (t.mask) t.mask[(size_t)m * t.ld_mask + midx[i]] = 1;
            }
            __syncthreads();
            if (tid == (int)blockDim.x - 1) s_base = off + (take ? 1 : 0);
            __syncthreads();
        }
    }
    const int gotf = s_base;
    for (int i = gotf + threadIdx.x; i < t.k; i += blockDim.x) out[i] = -1;
}

constexpr int TF_T = 512, TF_CAP = 4096, TF_SAMP = 4096, TF_KMAX = 2048;

DEV void topk_prefill_fallback(const TopkParams& p, int m, int n);

extern "C" __global__ void __launch_bounds__(TF_T, 4) topk_fast(const Meta* meta, const TopkParams p) {
    extern __shared__ __align__(16) uint32_t tfs[];
    uint32_t* ck = tfs;
    int* ci = (int*)(tfs + TF_CAP);
    int* sel = (int*)(tfs + 2 * TF_CAP);
    __shared__ uint32_t hist[256], s_prefix, s_need;
    __shared__ int s_cnt, s_nsel;
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.x;
    if (m >= meta->m) return;
    const int tid = threadIdx.x, lane = tid & 31;
    int n = (meta->pos + m + 1) / p.ratio;
    if (p.block > 1) n = (n + p.block - 1) / p.block;
    const float* x = p.x + (size_t)m * p.ld;
    if (p.mask)
        for (int i = tid; i < p.ld_mask; i += TF_T) p.mask[(size_t)m * p.ld_mask + i] = 0;
    const int k = min(p.k, n);
    uint32_t T = 0u;
    if (n > TF_CAP) {
        for (int j = tid; j < TF_SAMP; j += TF_T) ck[j] = fkey(x[(int)((long long)j * n / TF_SAMP)]);
        __syncthreads();
        const int r = min(TF_SAMP, (int)((long long)k * TF_SAMP * 3 / (2 * (long long)n)) + 8);
        radix_kth(ck, TF_SAMP, r, hist, &s_prefix, &s_need);
        T = s_prefix;
    }
    if (tid == 0) { s_cnt = 0; s_nsel = 0; }
    __syncthreads();
    constexpr int U = 4;
    const unsigned lt = (1u << lane) - 1;
    const int n4 = n & ~3;
    const int wb = (tid & ~31) * 4;
    bool overflow = false;
    for (int bw = wb; bw < n && !overflow; bw += TF_T * 4 * U) {
        const int b0 = bw + lane * 4;
        float4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i0 = b0 + u * TF_T * 4;
            if (i0 + 3 < n4 + (n4 == n ? 0 : 0) && i0 + 4 <= n) v[u] = *(const float4*)(x + i0);
            else if (i0 < n) { v[u].x = x[i0]; v[u].y = i0 + 1 < n ? x[i0 + 1] : -INFINITY; v[u].z = i0 + 2 < n ? x[i0 + 2] : -INFINITY; v[u].w = i0 + 3 < n ? x[i0 + 3] : -INFINITY; }
            else v[u] = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i0 = b0 + u * TF_T * 4;
            const float e4[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const uint32_t key = fkey(e4[e]);
                const bool take = i0 + e < n && key >= T;
                const unsigned b = __ballot_sync(0xffffffffu, take);
                if (b == 0u) continue;
                int base = 0;
                if (lane == __ffs(b) - 1) base = atomicAdd(&s_cnt, __popc(b));
                base = __shfl_sync(0xffffffffu, base, __ffs(b) - 1);
                overflow |= base + __popc(b) > TF_CAP;
                const int pos = base + __popc(b & lt);
                if (take && pos < TF_CAP) { ck[pos] = key; ci[pos] = i0 + e; }
            }
        }
    }
    __syncthreads();
    const int cnt = s_cnt;
    if (cnt < k || cnt > TF_CAP) {
        topk_prefill_fallback(p, m, n);
        return;
    }
    uint32_t kth = 0u;
    int need_eq = 0;
    if (k < cnt) {
        radix_kth(ck, cnt, k, hist, &s_prefix, &s_need);
        kth = s_prefix;
        need_eq = (int)s_need;
    }
    const uint32_t ninf = fkey(-INFINITY);
    for (int j = tid; j < cnt; j += TF_T) {
        const uint32_t key = ck[j];
        bool take = k >= cnt || key > kth;
        if (!take && key == kth) {
            const int mi = ci[j];
            int rk = 0;
            for (int q = 0; q < cnt && rk < need_eq; ++q) rk += ck[q] == kth && ci[q] < mi;
            take = rk < need_eq;
        }
        if (take && p.drop_neg_inf && key == ninf) take = false;
        if (take) sel[atomicAdd(&s_nsel, 1)] = ci[j];
    }
    __syncthreads();
    const int ns = s_nsel;
    int* out = p.out + (size_t)m * p.k;
    for (int j = tid; j < ns; j += TF_T) {
        const int v = sel[j];
        int r = 0;
        for (int q = 0; q < ns; ++q) r += sel[q] < v;
        out[r] = v;
        if (p.mask) p.mask[(size_t)m * p.ld_mask + v] = 1;
    }
    for (int i = ns + tid; i < p.k; i += TF_T) out[i] = -1;
}

DEV uint32_t dsm_ldu(const uint32_t* local, uint32_t rank) {
    return *(const volatile uint32_t*)__cluster_map_shared_rank((void*)local, rank);
}
DEV void dsm_stu(uint32_t* local, uint32_t rank, uint32_t v) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(local), r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(r), "r"(v) : "memory");
}
DEV uint32_t topk_cl_key(const float* x, const uint32_t* keys, int i, bool cached, bool drop_neg_inf) {
    if (cached) return keys[i];
    const uint32_t key = fkey(x[i]);
    return drop_neg_inf && key == fkey(-INFINITY) ? 0u : key;
}
DEV uint4 topk_load4(const float* x, int i, bool drop_neg_inf) {
    const float4 v = *(const float4*)(x + i);
    uint4 k = {fkey(v.x), fkey(v.y), fkey(v.z), fkey(v.w)};
    const uint32_t ni = fkey(-INFINITY);
    if (drop_neg_inf) {
        if (k.x == ni) k.x = 0;
        if (k.y == ni) k.y = 0;
        if (k.z == ni) k.z = 0;
        if (k.w == ni) k.w = 0;
    }
    return k;
}
DEV void topk_first_bin(uint32_t* hist, uint32_t key) {
    const int bin = key ? (int)(key >> 20) : -1;
    const int bl = __shfl_sync(0xffffffffu, bin, 0);
    if (__all_sync(0xffffffffu, bin == bl)) {
        if ((threadIdx.x & 31) == 0 && bl >= 0) atomicAdd(&hist[bl], 32u);
    } else if (bin >= 0) {
        atomicAdd(&hist[bin], 1u);
    }
}
DEV uint32_t topk_warp_scan(uint32_t v) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t u = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += u;
    }
    return v;
}
DEV uint4 topk_prefill_load4(const float* x, int i, int n) {
    if (i + 3 < n) return topk_load4(x, i, false);
    return make_uint4(i < n ? fkey(x[i]) : 0u, i + 1 < n ? fkey(x[i + 1]) : 0u,
                      i + 2 < n ? fkey(x[i + 2]) : 0u, i + 3 < n ? fkey(x[i + 3]) : 0u);
}
DEV void topk_prefill_fallback(const TopkParams& p, int m, int n) {
    constexpr int NW = TF_T / 32;
    __shared__ uint32_t hist[256], prefix, need, wg[NW], we[NW], total;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* x = p.x + (size_t)m * p.ld;
    const int k = min(p.k, n);
    if (tid == 0) { prefix = 0; need = k; }
    __syncthreads();
    uint32_t pmask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const uint32_t pre = prefix;
        for (int base = 0; base < n; base += TF_T * 4) {
            const int i = base + tid * 4;
            const uint4 v = topk_prefill_load4(x, i, n);
            const uint32_t keys[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int bin = i + j < n && (keys[j] & pmask) == pre ? (int)((keys[j] >> shift) & 255) : 256;
                const unsigned peers = __match_any_sync(0xffffffffu, bin);
                if (bin < 256 && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
            }
        }
        __syncthreads();
        if (warp == 0) {
            uint32_t c[8], sum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { c[j] = hist[255 - 8 * lane - j]; sum += c[j]; }
            const uint32_t incl = topk_warp_scan(sum), rank = need;
            const unsigned hit = __ballot_sync(0xffffffffu, incl >= rank);
            __syncwarp();
            if (lane == __ffs(hit) - 1) {
                uint32_t cum = incl - sum;
                int d = 255 - 8 * lane;
#pragma unroll
                for (int j = 0; j < 7; ++j, --d) {
                    if (cum + c[j] >= rank) break;
                    cum += c[j];
                }
                prefix = pre | ((uint32_t)d << shift);
                need = rank - cum;
            }
        }
        pmask |= 255u << shift;
        __syncthreads();
    }
    const uint32_t kth = prefix, neq = need, ni = fkey(-INFINITY);
    const int span = ((n + NW * 128 - 1) / (NW * 128)) * 128;
    const int w0 = warp * span, w1 = min(n, w0 + span);
    uint32_t ng = 0, ne = 0;
    for (int base = w0; base < w1; base += 128) {
        const int i = base + lane * 4;
        const uint4 v = topk_prefill_load4(x, i, w1);
        const uint32_t keys[4] = {v.x, v.y, v.z, v.w};
        uint32_t g = 0, e = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const bool valid = i + j < w1 && !(p.drop_neg_inf && keys[j] == ni);
            g += valid && keys[j] > kth;
            e += valid && keys[j] == kth;
        }
        ng += __reduce_add_sync(0xffffffffu, g);
        ne += __reduce_add_sync(0xffffffffu, e);
    }
    if (lane == 0) { wg[warp] = ng; we[warp] = ne; }
    __syncthreads();
    if (warp == 0) {
        const uint32_t g = lane < NW ? wg[lane] : 0u, e = lane < NW ? we[lane] : 0u;
        const uint32_t gi = topk_warp_scan(g), ei = topk_warp_scan(e);
        if (lane < NW) { wg[lane] = gi - g; we[lane] = ei - e; }
        if (lane == 31) total = gi + min(neq, ei);
    }
    __syncthreads();
    uint32_t eqb = we[warp], off = wg[warp] + min(neq, eqb);
    const uint32_t end = off + ng + min(ne, neq - min(neq, eqb));
    int* out = p.out + (size_t)m * p.k;
    for (int base = w0; base < w1 && off < end; base += 128) {
        const int i = base + lane * 4;
        const uint4 v = topk_prefill_load4(x, i, w1);
        const uint32_t keys[4] = {v.x, v.y, v.z, v.w};
        uint32_t eq = 0, take = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const bool valid = i + j < w1 && !(p.drop_neg_inf && keys[j] == ni);
            eq |= (uint32_t)(valid && keys[j] == kth) << j;
            take |= (uint32_t)(valid && keys[j] > kth) << j;
        }
        const uint32_t ei = topk_warp_scan(__popc(eq)), before = eqb + ei - __popc(eq);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if ((eq & (1u << j)) && before + __popc(eq & ((1u << j) - 1)) < neq) take |= 1u << j;
        const uint32_t ti = topk_warp_scan(__popc(take)), dst = off + ti - __popc(take);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (take & (1u << j)) {
                out[dst + __popc(take & ((1u << j) - 1))] = i + j;
                if (p.mask) p.mask[(size_t)m * p.ld_mask + i + j] = 1;
            }
        }
        off += __shfl_sync(0xffffffffu, ti, 31);
        eqb += __shfl_sync(0xffffffffu, ei, 31);
    }
    for (int i = total + tid; i < p.k; i += TF_T) out[i] = -1;
}

constexpr int TKC_NB = 4096;
template <int C, int TKC_T>
DEV void topk_cl_body(const Meta* meta, const TopkParams& t) {
    constexpr int NW = TKC_T / 32;
    static_assert(TKC_NB / C <= TKC_T, "the digit resolve holds one bin per thread");
    extern __shared__ __align__(16) uint32_t tkeys[];
    __shared__ uint32_t hist[TKC_NB], tot[TKC_NB / C], rsum[C], xg[C], xe[C], wg[NW], we[NW];
    __shared__ uint32_t s_sum, s_bin, s_need, s_cnt;
    __shared__ __align__(8) uint64_t lbar;
    const uint32_t c = cl_rank_a();
    const int m = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    for (int i = tid; i < TKC_NB; i += TKC_T) hist[i] = 0;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sm_addr(&lbar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    pdl_wait();
    pdl_launch();
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (m >= meta->m) return;
    const int n = topk_n(meta, t, m);
    const int S = ((n + C - 1) / C + 3) & ~3;
    const int b0 = (int)c * S, b1 = min(n, b0 + S), ns = max(0, b1 - b0);
    uint32_t cache_bytes;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(cache_bytes));
    const bool cached = (uint32_t)S * 4 <= cache_bytes;
    const float* x = t.x + (size_t)m * t.ld + b0;
    if (cached && tid == 0) {
        const uint32_t bytes = (uint32_t)((ns + 3) & ~3) * 4;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sm_addr(&lbar)), "r"(bytes) : "memory");
        if (bytes) bulk_g2s(tkeys, x, bytes, &lbar);
    }
    if (t.mask) {
        uint8_t* mk = t.mask + (size_t)m * t.ld_mask;
        const int e1 = c == C - 1 ? t.ld_mask : min(t.ld_mask, b0 + S);
        for (int i = b0 + tid; i < e1; i += TKC_T) mk[i] = 0;
    }
    const uint32_t ninf = fkey(-INFINITY);
    if (cached) {
        asm volatile("{ .reg .pred P; W_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W_%=; }" ::"r"(sm_addr(&lbar)) : "memory");
    }
    int loaded = 0;
    if (!cached) {
        for (; loaded + 4 * TKC_T <= ns; loaded += 4 * TKC_T) {
            const uint4 k = topk_load4(x, loaded + 4 * tid, t.drop_neg_inf);
            topk_first_bin(hist, k.x);
            topk_first_bin(hist, k.y);
            topk_first_bin(hist, k.z);
            topk_first_bin(hist, k.w);
        }
    }
    for (int i0 = loaded; i0 < ns; i0 += TKC_T) {
        const int i = i0 + tid;
        uint32_t key = 0;
        if (i < ns) {
            key = fkey(cached ? __uint_as_float(tkeys[i]) : x[i]);
            if (t.drop_neg_inf && key == ninf) key = 0;
            if (cached) tkeys[i] = key;
        }
        topk_first_bin(hist, key);
    }
    uint32_t pre = 0, pmask = 0, need = (uint32_t)t.k, eqall = 0;
    for (int pass = 0;; ++pass) {
        const int shift = pass == 0 ? 20 : pass == 1 ? 8 : 0, nb = pass < 2 ? TKC_NB : 256;
        if (pass > 0) {
            for (int i = tid; i < nb; i += TKC_T) hist[i] = 0;
            __syncthreads();
            int loaded = 0;
            if (!cached) {
                for (; loaded + 4 * TKC_T <= ns; loaded += 4 * TKC_T) {
                    const uint4 k = topk_load4(x, loaded + 4 * tid, t.drop_neg_inf);
                    if (k.x && (k.x & pmask) == pre) atomicAdd(&hist[(k.x >> shift) & (nb - 1)], 1u);
                    if (k.y && (k.y & pmask) == pre) atomicAdd(&hist[(k.y >> shift) & (nb - 1)], 1u);
                    if (k.z && (k.z & pmask) == pre) atomicAdd(&hist[(k.z >> shift) & (nb - 1)], 1u);
                    if (k.w && (k.w & pmask) == pre) atomicAdd(&hist[(k.w >> shift) & (nb - 1)], 1u);
                }
            }
            for (int i = loaded + tid; i < ns; i += TKC_T) {
                const uint32_t key = topk_cl_key(x, tkeys, i, cached, t.drop_neg_inf);
                if (key && (key & pmask) == pre) atomicAdd(&hist[(key >> shift) & (nb - 1)], 1u);
            }
        }
        if (tid == 0) s_sum = 0;
        __syncthreads();
        asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
        const int per = nb / C;
        uint32_t s = 0;
        for (int i = tid; i < per; i += TKC_T) {
            uint32_t v = 0;
#pragma unroll
            for (int r = 0; r < C; ++r) v += dsm_ldu(&hist[(int)c * per + i], r);
            tot[i] = v;
            s += v;
        }
        s = __reduce_add_sync(0xffffffffu, s);
        if (lane == 0 && s) atomicAdd(&s_sum, s);
        __syncthreads();
        if (tid < C) dsm_stu(&rsum[c], tid, s_sum);
        asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
        uint32_t above = 0;
        int cr = -1;
        for (int r = C - 1; r >= 0; --r) {
            if (above + rsum[r] >= need) { cr = r; break; }
            above += rsum[r];
        }
        if (cr < 0) { eqall = 1; break; }
        need -= above;
        {
            const int d = per - 1 - tid;
            const uint32_t v = d >= 0 ? dsm_ldu(&tot[d], cr) : 0u;
            uint32_t incl = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const uint32_t u = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += u;
            }
            if (lane == 31) wg[warp] = incl;
            __syncthreads();
            const uint32_t wv = lane < warp && lane * 32 < per ? wg[lane] : 0u;
            incl += __reduce_add_sync(0xffffffffu, wv);
            if (d >= 0 && incl >= need && incl - v < need) {
                s_bin = (uint32_t)(cr * per + d);
                s_need = need - (incl - v);
                s_cnt = v;
            }
        }
        __syncthreads();
        pre |= s_bin << shift;
        pmask |= (uint32_t)(nb - 1) << shift;
        need = s_need;
        const uint32_t cnt = s_cnt;
        if (need == cnt || shift == 0) { eqall = need == cnt; break; }
    }
    const int CH = ((ns + NW * 32 - 1) / (NW * 32)) * 32, w0 = warp * CH, w1 = min(ns, w0 + CH);
    uint32_t ng = 0, ne = 0;
    int counted = w0;
    if (!cached) {
        for (; counted + 128 <= w1; counted += 128) {
            const uint4 k = topk_load4(x, counted + 4 * lane, t.drop_neg_inf);
            const uint32_t g = (k.x && (k.x & pmask) > pre) + (k.y && (k.y & pmask) > pre)
                             + (k.z && (k.z & pmask) > pre) + (k.w && (k.w & pmask) > pre);
            const uint32_t e = (k.x && (k.x & pmask) == pre) + (k.y && (k.y & pmask) == pre)
                             + (k.z && (k.z & pmask) == pre) + (k.w && (k.w & pmask) == pre);
            ng += __reduce_add_sync(0xffffffffu, g);
            ne += __reduce_add_sync(0xffffffffu, e);
        }
    }
    for (int i0 = counted; i0 < w1; i0 += 32) {
        const int i = i0 + lane;
        const uint32_t key = i < w1 ? topk_cl_key(x, tkeys, i, cached, t.drop_neg_inf) : 0u, kp = key & pmask;
        ng += __popc(__ballot_sync(0xffffffffu, key && kp > pre));
        ne += __popc(__ballot_sync(0xffffffffu, key && kp == pre));
    }
    __syncthreads();
    if (lane == 0) { wg[warp] = ng; we[warp] = ne; }
    __syncthreads();
    if (warp == 0) {
        const uint32_t g = lane < NW ? wg[lane] : 0u, e = lane < NW ? we[lane] : 0u;
        uint32_t gi = g, ei = e;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t u = __shfl_up_sync(0xffffffffu, gi, o), v = __shfl_up_sync(0xffffffffu, ei, o);
            if (lane >= o) { gi += u; ei += v; }
        }
        if (lane < NW) { wg[lane] = gi - g; we[lane] = ei - e; }
        const uint32_t gt = __shfl_sync(0xffffffffu, gi, 31), et = __shfl_sync(0xffffffffu, ei, 31);
        if (lane < C) { dsm_stu(&xg[c], lane, gt); dsm_stu(&xe[c], lane, et); }
    }
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    const uint32_t nq = eqall ? 0xffffffffu : need;
    uint32_t gb = 0, eb0 = 0, gall = 0, eall = 0;
#pragma unroll
    for (int r = 0; r < C; ++r) {
        const uint32_t g = xg[r], e = xe[r];
        if (r < (int)c) { gb += g; eb0 += e; }
        gall += g;
        eall += e;
    }
    const uint32_t total = gall + min(nq, eall);
    uint32_t eqb = eb0 + we[warp];
    uint32_t off = gb + wg[warp] + min(nq, eqb);
    int* out = t.out + (size_t)m * t.k;
    uint8_t* mk = t.mask ? t.mask + (size_t)m * t.ld_mask : nullptr;
    int written = w0;
    if (!cached) {
        const uint32_t warp_end = off + ng + min(ne, nq - min(nq, eqb));
        for (; written + 128 <= w1 && off < warp_end; written += 128) {
            const uint4 k = topk_load4(x, written + 4 * lane, t.drop_neg_inf);
            const uint32_t ebits = (uint32_t)(k.x && (k.x & pmask) == pre)
                | ((uint32_t)(k.y && (k.y & pmask) == pre) << 1)
                | ((uint32_t)(k.z && (k.z & pmask) == pre) << 2)
                | ((uint32_t)(k.w && (k.w & pmask) == pre) << 3);
            uint32_t take = (uint32_t)(k.x && (k.x & pmask) > pre)
                | ((uint32_t)(k.y && (k.y & pmask) > pre) << 1)
                | ((uint32_t)(k.z && (k.z & pmask) > pre) << 2)
                | ((uint32_t)(k.w && (k.w & pmask) > pre) << 3);
            const uint32_t ei = topk_warp_scan(__popc(ebits));
            const uint32_t before = eqb + ei - __popc(ebits);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if ((ebits & (1u << j)) && before + __popc(ebits & ((1u << j) - 1)) < nq) take |= 1u << j;
            const uint32_t ni = topk_warp_scan(__popc(take));
            const uint32_t base = off + ni - __popc(take);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (take & (1u << j)) {
                    const int id = b0 + written + 4 * lane + j;
                    out[base + __popc(take & ((1u << j) - 1))] = id;
                    if (mk) mk[id] = 1;
                }
            }
            off += __shfl_sync(0xffffffffu, ni, 31);
            eqb += __shfl_sync(0xffffffffu, ei, 31);
        }
        if (off == warp_end) written = w1;
    }
    for (int i0 = written; i0 < w1; i0 += 32) {
        const int i = i0 + lane;
        const uint32_t key = i < w1 ? topk_cl_key(x, tkeys, i, cached, t.drop_neg_inf) : 0u, kp = key & pmask;
        const bool gt = key && kp > pre, eq = key && kp == pre;
        const uint32_t lt = (1u << lane) - 1, eb = __ballot_sync(0xffffffffu, eq);
        const bool take = gt || (eq && eqb + __popc(eb & lt) < nq);
        const uint32_t tb = __ballot_sync(0xffffffffu, take);
        if (take) {
            out[off + __popc(tb & lt)] = b0 + i;
            if (mk) mk[b0 + i] = 1;
        }
        off += __popc(tb);
        eqb += __popc(eb);
    }
    if (c == C - 1)
        for (int i = (int)total + tid; i < t.k; i += TKC_T) out[i] = -1;
}
#define TOPK_CL(C, T)                                                                                   \
    extern "C" __global__ void __launch_bounds__(T, 1) topk_cl_##C##_##T(const Meta* meta, const __grid_constant__ TopkParams t) { \
        topk_cl_body<C, T>(meta, t);                                                                     \
    }
TOPK_CL(8, 1024)
