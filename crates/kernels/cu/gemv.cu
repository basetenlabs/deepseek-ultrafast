#include "common.cuh"
#include "rope.cuh"
#include "sample.cuh"

enum { FMT_FP8 = 0, FMT_FP4 = 1, FMT_BF16 = 2, FMT_FP8_BF16ACT = 3 };
#define MOE_W2X 4
enum { MODE_STORE = 0, MODE_SWIGLU = 1, MODE_ATOMIC = 2, MODE_SSP = 3, MODE_PARTIAL = 4 };

#define MOE_MAX_SLOTS 97
struct MoePlan {
    int n_slots;
    int expert[MOE_MAX_SLOTS];
    int cnt[MOE_MAX_SLOTS];
    int tok[MOE_MAX_SLOTS][16];
    int pair[MOE_MAX_SLOTS][16];
    float w[MOE_MAX_SLOTS][16];
};

struct GemvParams {
    const uint8_t* w;
    const uint8_t* ws;
    const bf16* x;
    void* out;
    int M, N, K, ldx, ldo;
    int out_f32;
    int x_group_rows;
    float alpha;
    int stage_steps;
    const MoePlan* plan;
    unsigned long long w_stride;
    int mode;
    float swiglu_limit;
    const bf16* prenorm_w;
    float prenorm_eps;
    const float* norm_part;
    int norm_nslices;
    int ksplit_slots;
    const uint16_t* xq;
    const float* xsq;
    uint16_t* oq;
    float* osq;
    unsigned int* sched;
    void* out2;
    int ldo2;
    int n_split;
    const float* rope_cos;
    const float* rope_sin;
    const int* rope_pos;
    int rope_head;
    int rope_off;
    const uint8_t* w_sh;
    const float* r_scores;
    const float* r_bias;
    int* r_ids;
    float* r_w;
    int r_nexp, r_topk;
    float r_scale;
    int r_nsum;
    RopeParams kvr;
    int kvr_on;
    const int* out_pos;
    int iq_head, iq_off;
    const SampParams* sp;
    float* spart;
    int sp_vloc, sp_off;
    int xshift;
    int _padx;
#ifdef GEMV_XB
    uint8_t* xb;
#endif
};
constexpr int STEP8 = 4096 + 16;

template <int FMT> struct Fmt;
template <> struct Fmt<FMT_FP8> { static constexpr int TK = 32, STEP = 2048; };
template <> struct Fmt<FMT_FP4> { static constexpr int TK = 64, STEP = 2048 + 128; };
template <> struct Fmt<FMT_BF16> { static constexpr int TK = 16, STEP = 2048; };
template <> struct Fmt<FMT_FP8_BF16ACT> { static constexpr int TK = 32, STEP = 2048; };

constexpr int STAGES = 4;

DEV uint32_t smem_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(b)), "r"(count));
}
DEV void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(b)), "r"(bytes) : "memory");
}
DEV void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(b)) : "memory");
}
DEV void mbar_wait(uint64_t* b, uint32_t parity) {
    asm volatile(
        "{ .reg .pred p; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra WAIT_%=; }" ::"r"(
            smem_addr(b)),
        "r"(parity)
        : "memory");
}
DEV void tma_load_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
            smem_addr(dst)),
        "l"(src), "r"(bytes), "r"(smem_addr(bar)), "l"(0x12F0000000000000ull)
        : "memory");
}
DEV void tma_load_1d_en(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
            smem_addr(dst)),
        "l"(src), "r"(bytes), "r"(smem_addr(bar)), "l"(0x1000000000000000ull)
        : "memory");
}
DEV uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEV uint32_t cluster_size() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r)); return r; }
DEV void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DEV float ld_dsmem_f32(const float* local, uint32_t rank) {
    uint32_t a = smem_addr(local), r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r) : "memory");
    return v;
}

#include "quantblocks.cuh"
#include "route.cuh"
#include "hcepi.cuh"
DEV float2 bf16x2_to_f2(uint32_t w) {
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

template <int FMT, int NT, int PH>
__device__ __forceinline__ void gemv_body(const GemvParams& p) {
    if (p.kvr_on && blockIdx.x == gridDim.x - 1) {
        const int grp = threadIdx.x >> 6, ng = min(4, (int)(blockDim.x >> 6));
        if (grp >= ng) return;
        const int step = ng * (int)gridDim.y;
        for (int row = (int)blockIdx.y * ng + grp; row < 16 * p.kvr.rows_per_token; row += step)
            rope_body(p.rope_pos, p.kvr, row, threadIdx.x & 63, grp);
        return;
    }
    constexpr int CONSUMERS = 4 * PH;
    constexpr int TK = Fmt<FMT>::TK, STEP = Fmt<FMT>::STEP;
    constexpr int MT = 8 * NT;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int cs = (int)cluster_size(), crank = (int)cluster_rank();
    const int slot = blockIdx.z;
    const MoePlan* plan = p.plan;
    if (plan) {
        pdl_wait();
        if (slot >= plan->n_slots) return;
    }
    const int M = plan ? plan->cnt[slot] : p.M;
    const uint8_t* wbase = p.w + (plan ? (size_t)plan->expert[slot] * p.w_stride : 0);
    auto xrow = [&](int m) -> int { return plan ? (p.mode == MODE_SWIGLU ? plan->tok[slot][m] : plan->pair[slot][m]) : m; };
    const int quad = blockIdx.x;
    const int KS = p.K / cs, k0 = crank * KS;
    const int nsteps = KS / TK;
    const int SS = p.stage_steps;
    const int nst = (nsteps + SS - 1) / SS;

    uint8_t* ring = smem;
    uint64_t* full = (uint64_t*)(ring + (size_t)STAGES * SS * STEP);
    uint64_t* empty = full + STAGES;
    const int LDS = (FMT == FMT_FP8 || FMT == FMT_FP4) ? KS + 32 : KS + 8;
    uint16_t* xs = (uint16_t*)(empty + STAGES);
    float* sa = (float*)(xs + MT * LDS);
    const int nkb = KS / 32;
    float* red = sa + MT * nkb;

    bf16* pws = (bf16*)(red + PH * MT * 64);
    __shared__ __align__(8) uint64_t xbar;
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CONSUMERS); }
        mbar_init(&xbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    const uint8_t* wsrc = wbase + ((size_t)quad * (p.K / TK) + k0 / TK) * STEP;

    if (warp == CONSUMERS) {
        if (lane == 0) {
            for (int st = 0; st < nst; ++st) {
                const int s = st % STAGES;
                if (st >= STAGES) mbar_wait(&empty[s], ((st / STAGES) - 1) & 1);
                const int steps = min(SS, nsteps - st * SS);
                const uint32_t bytes = (uint32_t)steps * STEP;
                mbar_expect_tx(&full[s], bytes);
                tma_load_1d(ring + (size_t)s * SS * STEP, wsrc + (size_t)st * SS * STEP, bytes, &full[s]);
            }
        }
        pdl_wait();
        pdl_launch();
    } else {
        __shared__ uint8_t wss[4][256];
        if (FMT == FMT_FP8 || FMT == FMT_FP8_BF16ACT) {
            const int nkb_ = KS / 32;
            for (int i = tid; i < 2 * nkb_; i += CONSUMERS * 32) {
                const int sr = i / nkb_, kb = i % nkb_;
                wss[sr][kb] = p.ws[(size_t)(quad * 2 + sr) * (p.K / 32) + k0 / 32 + kb];
            }
        }
        pdl_wait();
        pdl_launch();
        const int row_base = quad * 64;
        const int xoff = p.x_group_rows ? (row_base / p.x_group_rows) * p.K : 0;
        const int nthr = CONSUMERS * 32;
        if (tid == 0) {
            mbar_expect_tx(&xbar, (uint32_t)(M + (p.prenorm_w ? 1 : 0)) * KS * 2);
            for (int m = 0; m < M; ++m) tma_load_1d(xs + m * LDS, p.x + (size_t)xrow(m) * p.ldx + xoff + k0, KS * 2, &xbar);
            if (p.prenorm_w) tma_load_1d(pws, p.prenorm_w + k0, KS * 2, &xbar);
        }
        const float dqmul = __uint_as_float((uint32_t)(127 - p.xshift) << 23), wmul = __uint_as_float((uint32_t)(127 + p.xshift) << 23);
        float* rs = red;
        if (p.prenorm_w && p.norm_part) {
            for (int m = tid; m < MT; m += CONSUMERS * 32) {
                float ss = 0.f;
                if (m < M)
                    ss = norm_part_sum(p.norm_part, xrow(m), p.norm_nslices);
                rs[m] = rsqrtf(ss / p.K + p.prenorm_eps);
            }
            asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        } else if (p.prenorm_w) {
            if (cs == 1) mbar_wait(&xbar, 0);
            for (int m = warp; m < MT; m += CONSUMERS) {
                float ss = 0.f;
                if (m < M) {
                    const bf16* xr = cs == 1 ? (const bf16*)(xs + m * LDS) : p.x + (size_t)xrow(m) * p.ldx + xoff;
#pragma unroll 4
                    for (int k = lane * 8; k < p.K; k += 256) {
                        const uint4 q = *(const uint4*)(xr + k);
                        const bf16* h = (const bf16*)&q;
#pragma unroll
                        for (int e = 0; e < 8; ++e) { const float v = __bfloat162float(h[e]); ss += v * v; }
                    }
                }
                ss = warp_sum(ss);
                if (lane == 0) rs[m] = rsqrtf(ss / p.K + p.prenorm_eps);
            }
            asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        }
        mbar_wait(&xbar, 0);
        auto prenorm = [&](int m, int kl, float v) -> float {
            if (!p.prenorm_w) return v;
            return __bfloat162float(__float2bfloat16(__bfloat162float(pws[kl]) * (v * rs[m])));
        };
        if (FMT == FMT_BF16 || FMT == FMT_FP8_BF16ACT) {
            for (int i = tid; i < MT * (KS / 8); i += nthr) {
                int m = i / (KS / 8), c = (i % (KS / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (m < M) {
                    v = *(const uint4*)(xs + m * LDS + c);
                    if (p.prenorm_w) {
                        bf16* h = (bf16*)&v;
#pragma unroll
                        for (int e = 0; e < 8; ++e) h[e] = __float2bfloat16(prenorm(m, c + e, __bfloat162float(h[e])));
                    }
                }
                *(uint4*)(xs + m * LDS + c) = v;
            }
            if (p.oq && blockIdx.x == 0 && !plan) {
                asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
                quant_blocks8(warp, CONSUMERS, lane, MT, M, nkb, p.oq + k0, p.K + 32, p.osq + (k0 / 32) * MT, [&](int m, int kb, int c, float* v) {
                    const uint4 q = *(const uint4*)(xs + m * LDS + kb * 32 + 8 * c);
                    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f = bf16x2_to_f2(w[e]);
                        v[2 * e] = f.x;
                        v[2 * e + 1] = f.y;
                    }
                });
            }
#ifdef GEMV_XB
            if (p.xb && blockIdx.x == 0 && !plan) {
                asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
                uint8_t* sfb = p.xb + (size_t)(p.K / 16) * 128;
                for (int i = tid; i < 8 * (KS / 32); i += nthr) {
                    const int m = i / (KS / 32), kb = i % (KS / 32), kg = k0 / 32 + kb;
                    float v[32];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint4 q = m < MT ? *(const uint4*)(xs + m * LDS + kb * 32 + 8 * j) : make_uint4(0, 0, 0, 0);
                        const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float2 f = bf16x2_to_f2(w[e]);
                            v[8 * j + 2 * e] = f.x;
                            v[8 * j + 2 * e + 1] = f.y;
                        }
                    }
                    float a = 0.f;
#pragma unroll
                    for (int j = 0; j < 32; ++j) a = fmaxf(a, fabsf(v[j]));
                    const float sc = round_scale_pow2(fmaxf(a, 1e-4f), 1.0f / 448.0f);
                    const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        uint32_t w[4];
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int j = h * 16 + q * 4;
                            w[q] = (uint32_t)f32x2_to_e4m3x2(v[j] * inv, v[j + 1] * inv) | ((uint32_t)f32x2_to_e4m3x2(v[j + 2] * inv, v[j + 3] * inv) << 16);
                        }
                        *(uint4*)(p.xb + (size_t)(kg * 2 + h) * 128 + m * 16) = make_uint4(w[0], w[1], w[2], w[3]);
                    }
                    sfb[(size_t)(kg / 4) * 512 + m * 16 + (kg % 4)] = (uint8_t)((__float_as_uint(sc) >> 23) & 0xff);
                }
            }
#endif
        } else {
            quant_blocks8(warp, CONSUMERS, lane, MT, M, nkb, xs, LDS, sa, [&](int m, int kb, int c, float* v) {
                const int kl = kb * 32 + 8 * c;
                const uint4 q = *(const uint4*)(xs + m * LDS + kl);
                const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f = bf16x2_to_f2(w[e]);
                    v[2 * e] = prenorm(m, kl + 2 * e, f.x);
                    v[2 * e + 1] = prenorm(m, kl + 2 * e + 1, f.y);
                }
            }, dqmul);
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));

        const int g = lane >> 2, t = lane & 3;
        const int rg = warp % 4, ph = warp / 4;
        const int row0 = quad * 64 + rg * 16;
        float acc[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
        const uint8_t* wsc = &wss[rg / 2][0];

        for (int st = 0; st < nst; ++st) {
            const int s = st % STAGES;
            mbar_wait(&full[s], (st / STAGES) & 1);
            const int steps = min(SS, nsteps - st * SS);
            const uint8_t* base = ring + (size_t)s * SS * STEP;
            constexpr int B = 1;
            const int nj = (steps - ph + PH - 1) / PH;
            for (int i0 = 0; i0 < nj; i0 += B) {
                uint4 vv[B];
#pragma unroll
                for (int i = 0; i < B; ++i) vv[i] = *(const uint4*)(base + (size_t)min(ph + (i0 + i) * PH, steps - 1) * STEP + rg * 512 + lane * 16);
#pragma unroll
                for (int i = 0; i < B; ++i) {
                const bool ok = i0 + i < nj;
                const int j = min(ph + (i0 + i) * PH, steps - 1);
                const uint4 v = vv[i];
                const int kl = (st * SS + j) * TK;
                if (FMT == FMT_FP8) {
                    const float sw = e8m0_to_f32(wsc[kl / 32]) * wmul;
                    const uint32_t a0[4] = {e4m3x2_to_f16x2(v.x & 0xffff), e4m3x2_to_f16x2(v.z & 0xffff),
                                            e4m3x2_to_f16x2(v.x >> 16), e4m3x2_to_f16x2(v.z >> 16)};
                    const uint32_t a1[4] = {e4m3x2_to_f16x2(v.y & 0xffff), e4m3x2_to_f16x2(v.w & 0xffff),
                                            e4m3x2_to_f16x2(v.y >> 16), e4m3x2_to_f16x2(v.w >> 16)};
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        const uint4 xb = *(const uint4*)(xs + (n * 8 + g) * LDS + kl + 8 * t);
                        float c[4] = {0.f, 0.f, 0.f, 0.f};
                        mma_f16(c, a0, xb.x, xb.y);
                        mma_f16(c, a1, xb.z, xb.w);
                        const float n0 = fmaf(c[0], sw, acc[n][0]), n1 = fmaf(c[1], sw, acc[n][1]);
                        const float n2 = fmaf(c[2], sw, acc[n][2]), n3 = fmaf(c[3], sw, acc[n][3]);
                        if (ok) { acc[n][0] = n0; acc[n][1] = n1; acc[n][2] = n2; acc[n][3] = n3; }
                    }
                } else if (FMT == FMT_FP4) {
                    const uint32_t sc = *(const uint32_t*)(base + (size_t)j * STEP + 2048 + rg * 32 + g * 4);
                    const float sg[2] = {e8m0_to_f32(sc & 0xff) * wmul, e8m0_to_f32((sc >> 8) & 0xff) * wmul};
                    const float sh[2] = {e8m0_to_f32((sc >> 16) & 0xff) * wmul, e8m0_to_f32(sc >> 24) * wmul};
                    const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float c[NT][4];
#pragma unroll
                        for (int n = 0; n < NT; ++n) c[n][0] = c[n][1] = c[n][2] = c[n][3] = 0.f;
                        uint4 xb[NT];
#pragma unroll
                        for (int n = 0; n < NT; ++n) xb[n] = *(const uint4*)(xs + (n * 8 + g) * LDS + kl + h * 32 + 8 * t);
                        uint32_t a[4];
                        cvt4_e2m1(wv[h * 2], a);
#pragma unroll
                        for (int n = 0; n < NT; ++n) mma_f16(c[n], a, xb[n].x, xb[n].y);
                        cvt4_e2m1(wv[h * 2 + 1], a);
#pragma unroll
                        for (int n = 0; n < NT; ++n) mma_f16(c[n], a, xb[n].z, xb[n].w);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            const float n0 = fmaf(c[n][0], sg[h], acc[n][0]), n1 = fmaf(c[n][1], sg[h], acc[n][1]);
                            const float n2 = fmaf(c[n][2], sh[h], acc[n][2]), n3 = fmaf(c[n][3], sh[h], acc[n][3]);
                            if (ok) { acc[n][0] = n0; acc[n][1] = n1; acc[n][2] = n2; acc[n][3] = n3; }
                        }
                    }
                } else if (FMT == FMT_FP8_BF16ACT) {
                    const float sw = e8m0_to_f32(wsc[kl / 32]);
                    const uint32_t a0[4] = {e4m3x2_to_bf16x2(v.x & 0xffff), e4m3x2_to_bf16x2(v.z & 0xffff),
                                            e4m3x2_to_bf16x2(v.x >> 16), e4m3x2_to_bf16x2(v.z >> 16)};
                    const uint32_t a1[4] = {e4m3x2_to_bf16x2(v.y & 0xffff), e4m3x2_to_bf16x2(v.w & 0xffff),
                                            e4m3x2_to_bf16x2(v.y >> 16), e4m3x2_to_bf16x2(v.w >> 16)};
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        const uint16_t* xb = xs + (n * 8 + g) * LDS + kl;
                        float c[4] = {0.f, 0.f, 0.f, 0.f};
                        mma_bf16(c, a0, *(const uint32_t*)(xb + 2 * t), *(const uint32_t*)(xb + 8 + 2 * t));
                        mma_bf16(c, a1, *(const uint32_t*)(xb + 16 + 2 * t), *(const uint32_t*)(xb + 24 + 2 * t));
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const float nq = fmaf(c[q], sw, acc[n][q]);
                            if (ok) acc[n][q] = nq;
                        }
                    }
                } else {
                    const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        const uint16_t* xb = xs + (n * 8 + g) * LDS + kl;
                        float tq[4] = {acc[n][0], acc[n][1], acc[n][2], acc[n][3]};
                        mma_bf16(tq, a, *(const uint32_t*)(xb + 2 * t), *(const uint32_t*)(xb + 8 + 2 * t));
                        if (ok) { acc[n][0] = tq[0]; acc[n][1] = tq[1]; acc[n][2] = tq[2]; acc[n][3] = tq[3]; }
                    }
                }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty[s]);
        }
        float* rp = red + (size_t)ph * MT * 64;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int m0 = n * 8 + 2 * t, r = rg * 16 + g;
            rp[m0 * 64 + r] = acc[n][0];
            rp[(m0 + 1) * 64 + r] = acc[n][1];
            rp[m0 * 64 + r + 8] = acc[n][2];
            rp[(m0 + 1) * 64 + r + 8] = acc[n][3];
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        for (int i = tid; i < MT * 64; i += CONSUMERS * 32) {
            float v = red[i];
#pragma unroll
            for (int q = 1; q < PH; ++q) v += red[(size_t)q * MT * 64 + i];
            red[i] = v;
        }
    }

    if (p.mode == MODE_PARTIAL) {
        if (warp < CONSUMERS) {
            asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
            for (int i = tid; i < M * 64; i += CONSUMERS * 32) {
                const int m = i / 64, r = quad * 64 + i % 64;
                ((float*)p.out)[((size_t)m * cs + crank) * p.ldo + r] = red[i] * p.alpha;
            }
        }
        return;
    }
    auto total_of = [&](int i) -> float {
        if (cs == 1) return red[i];
        float t[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) t[r] = r < cs ? ld_dsmem_f32(red + i, (uint32_t)r) : 0.f;
        float v = 0.f;
#pragma unroll
        for (int r = 0; r < 16; ++r) v += t[r];
        return v;
    };
    if (cs == 1) __syncthreads();
    else cluster_sync();
    if (p.mode == MODE_SWIGLU) {
        const int items = M * 32;
        const int per = (items + cs - 1) / cs;
        const float L = p.swiglu_limit;
        for (int i = crank * per + tid; i < min(items, (crank + 1) * per); i += blockDim.x) {
            const int m = i / 32, j = i % 32;
            float gte = __bfloat162float(__float2bfloat16(total_of(m * 64 + j)));
            float up = __bfloat162float(__float2bfloat16(total_of(m * 64 + 32 + j)));
            if (L > 0.f) { up = fminf(fmaxf(up, -L), L); gte = fminf(gte, L); }
            float a = gte / (1.0f + expf(-gte)) * up;
            const int orow = plan ? plan->pair[slot][m] : m;
            if (plan) a *= plan->w[slot][m];
            ((bf16*)p.out)[(size_t)orow * p.ldo + quad * 32 + j] = __float2bfloat16(a);
        }
    } else {
        const int total = M * 64;
        const int per = (total + cs - 1) / cs;
        for (int i = crank * per + tid; i < min(total, (crank + 1) * per); i += blockDim.x) {
            const int m = i / 64, r = quad * 64 + i % 64;
            const float v = total_of(i) * p.alpha;
            if (p.mode == MODE_ATOMIC) {
                const int orow = plan ? plan->tok[slot][m] : m;
                atomicAdd((float*)p.out + (size_t)orow * p.ldo + r, v);
            } else if (p.iq_head && r >= p.n_split) {
                const int rr = r - p.n_split, d = rr % p.iq_head;
                float x = __bfloat162float(__float2bfloat16(v));
                const float pt = __shfl_xor_sync(0xffffffffu, x, 1);
                if (d >= p.iq_off) {
                    const int pos = p.rope_pos[0] + m, f = (d - p.iq_off) / 2;
                    const float c = p.rope_cos[(size_t)pos * 32 + f], sn = p.rope_sin[(size_t)pos * 32 + f];
                    x = (r & 1) ? __fmaf_rn(pt, sn, __fmul_rn(x, c)) : __fmaf_rn(x, c, -__fmul_rn(pt, sn));
                    x = __bfloat162float(__float2bfloat16(x));
                }
                float amax = fabsf(x);
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                amax = fmaxf(amax, 6.f * 1.1754943508222875e-38f);
                const float sc = round_scale_pow2(amax, 1.0f / 6.0f);
                x = fp4_round(fminf(fmaxf(__fdiv_rn(x, sc), -6.f), 6.f)) * sc;
                ((bf16*)p.out2)[(size_t)m * p.ldo2 + rr] = __float2bfloat16(x);
            } else if (p.rope_cos && (!p.n_split || r < p.n_split) && (r % p.rope_head) >= p.rope_off) {
                if (r & 1) continue;
                const float a = __bfloat162float(__float2bfloat16(v)), b = __bfloat162float(__float2bfloat16(total_of(i + 1) * p.alpha));
                const int pos = p.rope_pos[0] + m, f = ((r % p.rope_head) - p.rope_off) / 2;
                const float c = p.rope_cos[(size_t)pos * 32 + f], sn = p.rope_sin[(size_t)pos * 32 + f];
                bf16* o = (bf16*)p.out + (size_t)m * p.ldo + r;
                o[0] = __float2bfloat16(__fmaf_rn(a, c, -__fmul_rn(b, sn)));
                o[1] = __float2bfloat16(__fmaf_rn(a, sn, __fmul_rn(b, c)));
            } else {
                const int orow = plan ? plan->pair[slot][m] : m + (p.out_pos ? p.out_pos[0] : 0);
                void* out = p.out;
                int ldo = p.ldo, rr = r;
                if (p.n_split && r >= p.n_split) { out = p.out2; ldo = p.ldo2; rr = r - p.n_split; }
                if (p.mode == MODE_SSP) ((float*)out)[(size_t)orow * ldo + rr] = sqrt_softplus(v);
                else if (p.out_f32) ((float*)out)[(size_t)orow * ldo + rr] = v;
                else ((bf16*)out)[(size_t)orow * ldo + rr] = __float2bfloat16(v);
            }
        }
    }
    if (cs > 1) cluster_sync();
}

#ifndef GEMV_XB
#define ENTRY(name, F, NT, PH)                                                                                  \
    extern "C" __global__ void __launch_bounds__((4 * PH + 1) * 32, GEMV_MIN_CTAS_PER_SM(PH)) name(const __grid_constant__ GemvParams p) { \
        gemv_body<F, NT, PH>(p);                                                                                  \
    }
#define ENTRIES(fmt, F)                                                                    \
    ENTRY(gemv_##fmt##_n1_p1, F, 1, 1) ENTRY(gemv_##fmt##_n2_p1, F, 2, 1)                  \
    ENTRY(gemv_##fmt##_n1_p2, F, 1, 2) ENTRY(gemv_##fmt##_n2_p2, F, 2, 2)                  \
    ENTRY(gemv_##fmt##_n1_p4, F, 1, 4) ENTRY(gemv_##fmt##_n2_p4, F, 2, 4)

template <int PH>
__device__ __forceinline__ void gemv_bf16_persist_body(const GemvParams& p) {
    constexpr int CONSUMERS = 4 * PH, TK = 16, STEP = 2048, MT = 8;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int M = p.M, K = p.K, nquads = p.N / 64;
    const int nsteps = K / TK, SS = p.stage_steps, nst = (nsteps + SS - 1) / SS;
    uint8_t* ring = smem;
    uint64_t* full = (uint64_t*)(ring + (size_t)STAGES * SS * STEP);
    uint64_t* empty = full + STAGES;
    const int LDS = K + 8;
    uint16_t* xs = (uint16_t*)(empty + STAGES);
    float* red = (float*)(xs + MT * LDS);
    bf16* pws = (bf16*)(red + PH * MT * 64);
    __shared__ __align__(8) uint64_t xbar;
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CONSUMERS); }
        mbar_init(&xbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == CONSUMERS) {
        if (lane == 0) {
            int g = 0;
            for (int q = blockIdx.x; q < nquads; q += gridDim.x) {
                const uint8_t* wsrc = p.w + (size_t)q * nsteps * STEP;
                for (int st = 0; st < nst; ++st, ++g) {
                    const int s = g % STAGES;
                    if (g >= STAGES) mbar_wait(&empty[s], ((g / STAGES) - 1) & 1);
                    const uint32_t bytes = (uint32_t)min(SS, nsteps - st * SS) * STEP;
                    mbar_expect_tx(&full[s], bytes);
                    tma_load_1d(ring + (size_t)s * SS * STEP, wsrc + (size_t)st * SS * STEP, bytes, &full[s]);
                }
            }
        }
        pdl_wait();
        pdl_launch();
        return;
    }
    pdl_wait();
    pdl_launch();
    if (tid == 0) {
        mbar_expect_tx(&xbar, (uint32_t)(M + (p.prenorm_w ? 1 : 0)) * K * 2);
        for (int m = 0; m < M; ++m) tma_load_1d(xs + m * LDS, p.x + (size_t)m * p.ldx, K * 2, &xbar);
        if (p.prenorm_w) tma_load_1d(pws, p.prenorm_w, K * 2, &xbar);
    }
    float* rs = red;
    if (p.prenorm_w && p.norm_part) {
        for (int m = tid; m < MT; m += CONSUMERS * 32) {
            float ss = 0.f;
            if (m < M) ss = norm_part_sum(p.norm_part, m, p.norm_nslices);
            rs[m] = rsqrtf(ss / p.K + p.prenorm_eps);
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    }
    mbar_wait(&xbar, 0);
    for (int i = tid; i < MT * (K / 8); i += CONSUMERS * 32) {
        const int m = i / (K / 8), c = (i % (K / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (m < M) {
            v = *(const uint4*)(xs + m * LDS + c);
            if (p.prenorm_w) {
                bf16* h = (bf16*)&v;
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    h[e] = __float2bfloat16(__bfloat162float(__float2bfloat16(__bfloat162float(pws[c + e]) * (__bfloat162float(h[e]) * rs[m]))));
            }
        }
        *(uint4*)(xs + m * LDS + c) = v;
    }
    asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    const int g8 = lane >> 2, t = lane & 3;
    const int rg = warp % 4, ph = warp / 4;
    int g = 0;
    for (int q = blockIdx.x; q < nquads; q += gridDim.x) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int st = 0; st < nst; ++st, ++g) {
            const int s = g % STAGES;
            mbar_wait(&full[s], (g / STAGES) & 1);
            const int steps = min(SS, nsteps - st * SS);
            const uint8_t* base = ring + (size_t)s * SS * STEP;
            for (int j = ph; j < steps; j += PH) {
                const uint4 v = *(const uint4*)(base + (size_t)j * STEP + rg * 512 + lane * 16);
                const int kl = (st * SS + j) * TK;
                const uint32_t a[4] = {v.x, v.y, v.z, v.w};
                const uint16_t* xb = xs + g8 * LDS + kl;
                float tq[4] = {acc[0], acc[1], acc[2], acc[3]};
                mma_bf16(tq, a, *(const uint32_t*)(xb + 2 * t), *(const uint32_t*)(xb + 8 + 2 * t));
                acc[0] = tq[0]; acc[1] = tq[1]; acc[2] = tq[2]; acc[3] = tq[3];
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty[s]);
        }
        float* rp = red + (size_t)ph * MT * 64;
        {
            const int m0 = 2 * t, r = rg * 16 + g8;
            rp[m0 * 64 + r] = acc[0];
            rp[(m0 + 1) * 64 + r] = acc[1];
            rp[m0 * 64 + r + 8] = acc[2];
            rp[(m0 + 1) * 64 + r + 8] = acc[3];
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        for (int i = tid; i < MT * 64; i += CONSUMERS * 32) {
            float v = red[i];
#pragma unroll
            for (int k = 1; k < PH; ++k) v += red[(size_t)k * MT * 64 + i];
            red[i] = v;
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        for (int i = tid; i < M * 64; i += CONSUMERS * 32) {
            const int m = i / 64, r = q * 64 + i % 64;
            ((float*)p.out)[(size_t)m * p.ldo + r] = red[i] * p.alpha;
        }
        if (p.sp && warp < M) {
            __syncwarp();
            const int m = warp;
            const float inv_t = 1.f / p.sp->temp;
            float v[2];
            int id[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = q * 64 + lane + 32 * h;
                v[h] = r < p.sp_vloc ? red[m * 64 + lane + 32 * h] * p.alpha * inv_t : -INFINITY;
                id[h] = r + p.sp_off;
            }
            const float v0 = v[0], v1 = v[1];
            float* rec = p.spart + ((size_t)m * nquads + q) * 20;
            float mx = -INFINITY;
#pragma unroll 1
            for (int k = 0; k < 9; ++k) {
                const int hb = v[1] > v[0] ? 1 : 0;
                const float lv = hb ? v[1] : v[0];
                const int li = hb ? id[1] : id[0];
                const unsigned u = __float_as_uint(lv);
                const unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                const unsigned kmax = __reduce_max_sync(0xffffffffu, key);
                const int bi = (int)__reduce_min_sync(0xffffffffu, key == kmax ? (unsigned)li : 0x7fffffffu);
                const float bv = __uint_as_float((kmax & 0x80000000u) ? (kmax & 0x7fffffffu) : ~kmax);
                if (k == 0) mx = bv;
                if (lane == 0) {
                    if (k < 8) { rec[3 + k] = bv; rec[11 + k] = __int_as_float(bi); } else rec[2] = bv;
                }
                v[0] = id[0] == bi ? -INFINITY : v[0];
                v[1] = id[1] == bi ? -INFINITY : v[1];
            }
            const float se = warp_sum((v0 == -INFINITY ? 0.f : expf(v0 - mx)) + (v1 == -INFINITY ? 0.f : expf(v1 - mx)));
            if (lane == 0) { rec[0] = mx; rec[1] = se; }
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    }
}
#ifndef GEMV_NO_ENTRIES
extern "C" __global__ void __launch_bounds__(17 * 32) gemv_bf16_persist_p4(const __grid_constant__ GemvParams p) { gemv_bf16_persist_body<4>(p); }

ENTRIES(fp8, FMT_FP8)
ENTRIES(fp4, FMT_FP4)
ENTRIES(bf16, FMT_BF16)
ENTRIES(fp8b, FMT_FP8_BF16ACT)
#endif

struct SPlan {
    int n_slots, P;
    int sel[96];
    float selw[96];
    short expert[MOE_MAX_SLOTS];
    unsigned char cnt[MOE_MAX_SLOTS];
    unsigned char pair[MOE_MAX_SLOTS][16];
    unsigned char isf[96];
    float bias[384];
    unsigned emask[384][3];
    unsigned fmask[3];
};

DEV void route_plan(const GemvParams& p, SPlan& sp, MoePlan* gp, int M, int shared, const float* scores, int nsum
) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int topk = p.r_topk, P = M * topk;
    const bool g0 = blockIdx.x == 0;
    __syncthreads();
    if (warp < M)
        route_token(warp, lane, scores, sp.bias, p.r_nexp, topk, p.r_scale, sp.sel + warp * topk, sp.selw + warp * topk,
                    g0 ? p.r_ids : nullptr, p.r_w, nsum);
    __syncthreads();
    int first = -1, rank = 0, tot = 0, e = 0;
    if (tid < P) {
        e = sp.sel[tid];
        atomicOr(&sp.emask[e][tid >> 5], 1u << (tid & 31));
    }
    __syncthreads();
    bool isf = false;
    if (tid < P) {
        const unsigned m0 = sp.emask[e][0], m1 = sp.emask[e][1], m2 = sp.emask[e][2];
        const unsigned bl = (1u << (tid & 31)) - 1;
        const int w = tid >> 5;
        rank = (w > 0 ? __popc(m0) : __popc(m0 & bl)) * 1 + (w == 1 ? __popc(m1 & bl) : (w > 1 ? __popc(m1) : 0)) + (w == 2 ? __popc(m2 & bl) : 0);
        tot = __popc(m0) + __popc(m1) + __popc(m2);
        first = m0 ? __ffs(m0) - 1 : (m1 ? 32 + __ffs(m1) - 1 : 64 + __ffs(m2) - 1);
        isf = first == tid;
    }
    if (warp < 3) {
        const unsigned b = __ballot_sync(0xffffffffu, isf);
        if (lane == 0) sp.fmask[warp] = b;
    }
    __syncthreads();
    if (tid < P) {
        const int fw = first >> 5;
        const unsigned fb = (1u << (first & 31)) - 1;
        int sl = shared + __popc(sp.fmask[fw] & fb);
        for (int q = 0; q < fw; ++q) sl += __popc(sp.fmask[q]);
        sp.pair[sl][rank] = (unsigned char)tid;
        if (isf) { sp.expert[sl] = (short)e; sp.cnt[sl] = (unsigned char)tot; }
        if (g0) {
            gp->tok[sl][rank] = tid / topk;
            gp->pair[sl][rank] = tid;
            gp->w[sl][rank] = sp.selw[tid];
            if (isf) { gp->expert[sl] = e; gp->cnt[sl] = tot; }
        }
    }
    if (tid == 0) {
        const int n = shared + __popc(sp.fmask[0]) + __popc(sp.fmask[1]) + __popc(sp.fmask[2]);
        sp.n_slots = n;
        sp.P = P;
        if (g0) gp->n_slots = n;
    }
    if (shared && tid < M) {
        if (tid == 0) {
            sp.expert[0] = -1;
            sp.cnt[0] = (unsigned char)M;
            if (g0) { gp->expert[0] = -1; gp->cnt[0] = M; }
        }
        sp.pair[0][tid] = (unsigned char)(P + tid);
        if (g0) { gp->tok[0][tid] = tid; gp->pair[0][tid] = P + tid; gp->w[0][tid] = 1.0f; }
    }
    __syncthreads();
}

struct W2Params {
    const uint8_t* w;
    unsigned long long w_stride;
    const uint8_t* w_sh;
    float* out;
    int ldo, N, K;
    float alpha;
    unsigned* done;
    int sq_ld;
    int qps;
    int w2x;
    int flags;
};

struct GateParams {
    const uint8_t* w;
    float* part;
    unsigned* ctr;
    uint16_t* xq;
    float* xsq;
    int nks;
    int quads;
    int ss, ph;
    unsigned* mixf;
    const bf16* dots_h;
    const float* dots_fn;
    unsigned* dots_ctr;
    int dots_z;
    int _pad2;
    HcMix mx;
};

template <int CONSUMERS>
__device__ __noinline__ void moe_dots_item(const GateParams& gt, int M, int slice, int z, const float* fns, float* red4, uint64_t* gbar) {
    const int tid = threadIdx.x;
    const int nj = MIX / gt.dots_z, jz = z * nj, dim = gt.mx.dim, nslices = gt.mx.nslices;
    const int grp = tid >> 6, t = tid & 63;
    const int d = slice * SLICE + t * 4, dl = t * 4;
    constexpr int RW = 65;
    mbar_wait(gbar, 0);
    if (tid < 256) {
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int m = m0 + grp;
            if (m < M) {
                float h4[HC][4];
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const uint2 v = __ldcg((const uint2*)(gt.dots_h + ((size_t)m * HC + c) * dim + d));
                    const bf16* b = (const bf16*)&v;
#pragma unroll
                    for (int k = 0; k < 4; ++k) h4[c][k] = __bfloat162float(b[k]);
                }
#pragma unroll
                for (int j = 0; j < MIX; ++j) {
                    if (j < jz || j >= jz + nj) continue;
                    float s = 0.f;
#pragma unroll
                    for (int c = 0; c < HC; ++c) {
                        const float4 f = *(const float4*)(fns + ((size_t)(j - jz) * HC + c) * SLICE + dl);
                        s += h4[c][0] * f.x + h4[c][1] * f.y + h4[c][2] * f.z + h4[c][3] * f.w;
                    }
                    red4[(grp * MIX + j) * RW + t] = s;
                }
            }
            asm volatile("bar.sync 2, 256;");
            if (t >= jz && t < jz + nj && m < M) {
                float s = 0.f;
                for (int q = 0; q < 64; ++q) s += red4[(grp * MIX + t) * RW + q];
                ((float*)gt.mx.part)[((size_t)m * nslices + slice) * NP + t] = s;
            }
            asm volatile("bar.sync 2, 256;");
        }
    }
    asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    if (tid == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(gt.dots_ctr) : "memory");
}

__device__ __noinline__ void moe_mix_item(const GateParams& gt, int M, int nd, float* red) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) {
        for (;;) {
            unsigned d;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(d) : "l"(gt.dots_ctr) : "memory");
            if (d >= (unsigned)nd) break;
            __nanosleep(64);
        }
    }
    __syncthreads();
    if (warp < M) {
        hc_mix_token<true>(gt.mx, warp, lane, red + warp * (MIX + 1), gt.mx.pre + warp * HC, gt.mx.post + warp * HC, gt.mx.comb + warp * HC * HC);
        if (gt.mixf) {
            __syncwarp();
            if (lane == 0) {
                __threadfence();
                asm volatile("st.release.gpu.global.u32 [%0], 1;" ::"l"(gt.mixf + warp) : "memory");
            }
        }
    }
}

DEV void moe_mixf_arm(const GateParams& gt, int M) {
    if (gt.mixf && (int)threadIdx.x < M) {
        gt.mixf[threadIdx.x] = 0u;
        __threadfence();
    }
}

template <int CONSUMERS, int MT>
__device__ __forceinline__ void moe_gate_item(const GemvParams& p, const GateParams& gt, bool xcta, int gq, int gks, int KSg, int ntg, int SS16, int nst_g, int stage_bytes,
                                           uint16_t* xg, const bf16* gw, const float* rsr, const uint8_t* ring, uint64_t* full, uint64_t* empty, uint64_t* gbar,
                                           float* red) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int LDSg = KSg + 8;
    mbar_wait(gbar, 0);
    for (int i = tid; i < p.M * (KSg / 8); i += CONSUMERS * 32) {
        const int m = i / (KSg / 8), c = (i - m * (KSg / 8)) * 8;
        uint4 v = *(const uint4*)(xg + m * LDSg + c);
        const uint4 gv = *(const uint4*)(gw + c);
        bf16* h = (bf16*)&v;
        const bf16* gh = (const bf16*)&gv;
#pragma unroll
        for (int e = 0; e < 8; ++e) h[e] = __float2bfloat16(__bfloat162float(gh[e]) * (__bfloat162float(h[e]) * rsr[m]));
        *(uint4*)(xg + m * LDSg + c) = v;
    }
    asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    if (xcta) {
        quant_blocks8(warp, CONSUMERS, lane, MT, p.M, KSg / 32, gt.xq + gks * KSg, p.K + 32, gt.xsq + (gks * KSg / 32) * MT, [&](int m, int kb, int c, float* v) {
            const uint4 q4 = *(const uint4*)(xg + m * LDSg + kb * 32 + 8 * c);
            const uint32_t w[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = bf16x2_to_f2(w[e]);
                v[2 * e] = f.x;
                v[2 * e + 1] = f.y;
            }
        });
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        if (tid == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(gt.ctr + 1) : "memory");
        return;
    }
    const int g8 = lane >> 2, t4 = lane & 3, rg = warp % 4, ph = warp / 4;
    const int gss = gt.ss, gpm = gt.ph - 1;
    const bool strided = (gss & gpm) == 0;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int jr = 0;
    for (int st = 0; st < nst_g; ++st) {
        mbar_wait(&full[st], 0);
        const uint8_t* base = ring + (size_t)st * stage_bytes;
        const int tiles = min(SS16, ntg - st * SS16);
        const int J0 = st * SS16;
        auto tile = [&](int j) {
            const uint4 v = *(const uint4*)(base + (size_t)j * 2048 + rg * 512 + lane * 16);
            const uint32_t a[4] = {v.x, v.y, v.z, v.w};
            const uint16_t* xb = xg + g8 * LDSg + (J0 + j) * 16;
            mma_bf16(acc, a, *(const uint32_t*)(xb + 2 * t4), *(const uint32_t*)(xb + 8 + 2 * t4));
        };
        if (ph <= gpm) {
            if (strided) {
                const int step = gpm + 1;
                int j = (ph - J0) & gpm;
                for (; j + step < tiles; j += 2 * step) {
                    const uint4 v0 = *(const uint4*)(base + (size_t)j * 2048 + rg * 512 + lane * 16);
                    const uint4 v1 = *(const uint4*)(base + (size_t)(j + step) * 2048 + rg * 512 + lane * 16);
                    const uint16_t* xb0 = xg + g8 * LDSg + (J0 + j) * 16;
                    const uint16_t* xb1 = xb0 + step * 16;
                    const uint32_t x00 = *(const uint32_t*)(xb0 + 2 * t4), x01 = *(const uint32_t*)(xb0 + 8 + 2 * t4);
                    const uint32_t x10 = *(const uint32_t*)(xb1 + 2 * t4), x11 = *(const uint32_t*)(xb1 + 8 + 2 * t4);
                    const uint32_t a0[4] = {v0.x, v0.y, v0.z, v0.w}, a1[4] = {v1.x, v1.y, v1.z, v1.w};
                    mma_bf16(acc, a0, x00, x01);
                    mma_bf16(acc, a1, x10, x11);
                }
                if (j < tiles) tile(j);
            } else {
                for (int j = 0; j < tiles; ++j) {
                    if ((jr & gpm) == ph) tile(j);
                    if (++jr == gss) jr = 0;
                }
            }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[st]);
    }
    float* rp = red + (size_t)ph * MT * 64;
    {
        const int m0 = 2 * t4, r = rg * 16 + g8;
        rp[m0 * 64 + r] = acc[0];
        rp[(m0 + 1) * 64 + r] = acc[1];
        rp[m0 * 64 + r + 8] = acc[2];
        rp[(m0 + 1) * 64 + r + 8] = acc[3];
    }
    asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    for (int i = tid; i < p.M * 64; i += CONSUMERS * 32) {
        float v = red[i];
        for (int qq = 1; qq < gt.ph; ++qq) v += red[(size_t)qq * MT * 64 + i];
        const int m = i >> 6, r = i & 63;
        gt.part[((size_t)m * gt.nks + gks) * p.r_nexp + gq * 64 + r] = v;
    }
    asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    if (tid == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(gt.ctr) : "memory");
}

template <int MT>
__device__ __forceinline__ void moe_gate_scores(const GemvParams& p, const GateParams& gt, int ng, int LDS, int nkb, uint16_t* xs, float* sa, uint64_t* xbar, float* s_scores) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        for (;;) {
            unsigned d;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(d) : "l"(gt.ctr + 1) : "memory");
            if (d >= (unsigned)gt.nks) break;
            __nanosleep(64);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
        const uint32_t xb = (uint32_t)p.M * LDS * 2, sb = (uint32_t)nkb * MT * 4;
        mbar_expect_tx(xbar, xb + sb);
        tma_load_1d(xs, gt.xq, xb, xbar);
        tma_load_1d(sa, gt.xsq, sb, xbar);
        for (;;) {
            unsigned d;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(d) : "l"(gt.ctr) : "memory");
            if (d >= (unsigned)ng) break;
            __nanosleep(64);
        }
    }
    __syncthreads();
    const int nexp = p.r_nexp;
    for (int i = tid; i < p.M * (nexp / 4); i += blockDim.x) {
        const int m = i / (nexp / 4), e = (i - m * (nexp / 4)) * 4;
        const float4* pp = (const float4*)(gt.part + (size_t)m * gt.nks * nexp + e);
        float4 pr[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) pr[r] = r < gt.nks ? __ldcg(pp + (size_t)r * (nexp / 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int r = 0; r < 8; ++r) { v.x += pr[r].x; v.y += pr[r].y; v.z += pr[r].z; v.w += pr[r].w; }
        *(float4*)(s_scores + m * nexp + e) = make_float4(sqrt_softplus(v.x), sqrt_softplus(v.y), sqrt_softplus(v.z), sqrt_softplus(v.w));
    }
}

template <int NT, int MODE, int PH, bool FUSED = false, bool GATE = false>
__device__ __forceinline__ void moe_body(const GemvParams& p, int S, const W2Params* q = nullptr, const GateParams* gt = nullptr) {
    constexpr int CONSUMERS = 4 * PH;
    constexpr int TK = 64, STEP = 2048 + 128, MT = 8 * NT;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int quads = p.N / 64;
    const MoePlan* plan = p.plan;
    const int SS = p.stage_steps;
    const int nsteps = p.K / TK;
    const int nst = (nsteps + SS - 1) / SS;
    const int SS8 = max(1, SS * STEP / STEP8), nst8 = (nsteps + SS8 - 1) / SS8;
    constexpr int W2X = FUSED ? MOE_W2X : 0, NSLOT = STAGES + W2X;
    const int nslot = FUSED && q->w2x ? 8 : 4, nslot_lg = FUSED && q->w2x ? 3 : 2;
    uint8_t* ring = smem;
    uint64_t* full = (uint64_t*)(ring + (size_t)STAGES * SS * STEP);
    uint64_t* empty = full + NSLOT;
    const int LDS = p.K + 32;
    uint16_t* xs = (uint16_t*)(empty + NSLOT);
    const int nkb = p.K / 32;
    float* sa = (float*)(xs + MT * LDS);
    float* red = sa + MT * nkb;
    __shared__ __align__(8) uint64_t xbar;
    const int LDS2 = FUSED ? q->K + 32 : 0, sq_ld1 = FUSED ? q->sq_ld : 0;
    uint16_t* xs2 = (uint16_t*)(red + 4 * MT * 64);
    float* sa2 = (float*)(xs2 + 2 * MT * LDS2);
    __shared__ __align__(8) uint64_t x2full[2], x2empty[2];
    __shared__ __align__(8) uint64_t epi_bar, epi_ack;
    __shared__ int s_pend[2];
    __shared__ __align__(8) uint64_t xsfree;
    __shared__ __align__(8) uint64_t gbar;
    if (tid == 0) {
        for (int s = 0; s < NSLOT; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CONSUMERS); }
        mbar_init(&xbar, 1);
        mbar_init(&gbar, 1);
        if (FUSED) for (int b = 0; b < 2; ++b) { mbar_init(&x2full[b], 1); mbar_init(&x2empty[b], CONSUMERS); }
        if (FUSED) { mbar_init(&epi_bar, CONSUMERS); mbar_init(&epi_ack, 1); mbar_init(&xsfree, CONSUMERS); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    __shared__ SPlan sp;
    constexpr bool gate = GATE;
    const int ng = gate ? gt->quads * gt->nks : 0, nitg = gate ? ng + gt->nks : 0;
    const bool gcta = gate && (int)blockIdx.x < nitg;
    const bool xcta = gate && (int)blockIdx.x >= ng && (int)blockIdx.x < nitg;
    const int gq = gcta ? (int)blockIdx.x / gt->nks : 0, gks = gcta ? (int)blockIdx.x % gt->nks : 0;
    const bool dots = gate && gt->dots_fn != nullptr;
    const int nd = dots ? gt->mx.nslices * gt->dots_z : 0;
    const bool dcta = dots && (int)blockIdx.x >= nitg && (int)blockIdx.x < nitg + nd;
    const bool mcta = dots && (int)blockIdx.x == nitg + nd;
    const int dslice = dcta ? ((int)blockIdx.x - nitg) % gt->mx.nslices : 0, dz = dcta ? ((int)blockIdx.x - nitg) / gt->mx.nslices : 0;
    const int KSg = gate ? p.K / gt->nks : 0, LDSg = KSg + 8, ntg = KSg / 16;
    const int SS16 = SS * STEP / 2048;
    const int nst_g = gcta && !xcta ? (ntg + SS16 - 1) / SS16 : 0;
    const bool fr = MODE == MODE_SWIGLU && (p.r_scores != nullptr || gate);
    auto pl_nslots = [&]() { return fr ? sp.n_slots : plan->n_slots; };
    auto pl_expert = [&](int sl) { return fr ? (int)sp.expert[sl] : plan->expert[sl]; };
    auto pl_cnt = [&](int sl) { return fr ? (int)sp.cnt[sl] : plan->cnt[sl]; };
    auto pl_pair = [&](int sl, int k) { return fr ? (int)sp.pair[sl][k] : plan->pair[sl][k]; };
    auto pl_tok = [&](int sl, int k) {
        if (!fr) return plan->tok[sl][k];
        const int pr = sp.pair[sl][k];
        return pr < sp.P ? pr / p.r_topk : pr - sp.P;
    };
    auto pl_w = [&](int sl, int k) {
        if (!fr) return plan->w[sl][k];
        const int pr = sp.pair[sl][k];
        return pr < sp.P ? sp.selw[pr] : 1.0f;
    };
    const bool dyn = p.sched != nullptr;
    __shared__ int sitem[STAGES + MOE_W2X];
    auto slot_of = [&](int ring_k, int w2r) -> int { return (w2r < 0 || (w2r & (nslot - 1)) < STAGES) ? ring_k % STAGES : w2r & (nslot - 1); };
    auto uses_of = [&](int slot, int ring_k, int w2r) -> int { return slot < STAGES ? (ring_k - slot + STAGES - 1) / STAGES : (w2r - slot + nslot - 1) >> nslot_lg; };
    const int w2sz = !FUSED ? 0 : q->qps == 2 ? max(2 * (q->K / 64) * STEP, q->K / 64 * STEP8) : max(q->K / 64 * STEP, (q->K / 64 + 1) / 2 * STEP8);
    auto slot_base = [&](int slot) -> uint8_t* { return slot < STAGES ? ring + (size_t)slot * SS * STEP : (uint8_t*)xs + (size_t)(slot - STAGES) * w2sz; };
    int pg = 0, nxt = 0, st0 = 0;
    auto issue = [&](int it, int st, int lim) {
        const int sl = it / quads, quad = it % quads;
        const int ex = it < quads && p.w_sh ? -1 : pl_expert(sl);
        const bool shx = ex < 0;
        const int stb = shx ? STEP8 : STEP, ssx = shx ? SS8 : SS, nstx = shx ? nst8 : nst;
        const uint8_t* src = shx ? p.w_sh + (size_t)quad * nsteps * STEP8 : p.w + (size_t)ex * p.w_stride + (size_t)quad * nsteps * STEP;
        for (; st < nstx && pg < lim; ++st, ++pg) {
            const int s = pg % STAGES;
            if (pg >= STAGES) mbar_wait(&empty[s], ((pg / STAGES) - 1) & 1);
            sitem[s] = it;
            const int steps = min(ssx, nsteps - st * ssx);
            mbar_expect_tx(&full[s], (uint32_t)steps * stb);
            tma_load_1d(ring + (size_t)s * SS * STEP, src + (size_t)st * ssx * stb, (uint32_t)steps * stb, &full[s]);
        }
        return st;
    };
    const int nst_sh = nst8;
    if (gcta && !xcta && warp == CONSUMERS && lane == 0) {
        const uint8_t* src = gt->w + ((size_t)gq * (p.K / 16) + (size_t)gks * ntg) * 2048;
        for (int st = 0; st < nst_g; ++st, ++pg) {
            sitem[pg] = -2;
            const int tiles = min(SS16, ntg - st * SS16);
            mbar_expect_tx(&full[pg], (uint32_t)tiles * 2048);
            tma_load_1d(ring + (size_t)pg * SS * STEP, src + (size_t)st * SS16 * 2048, (uint32_t)tiles * 2048, &full[pg]);
        }
    }
    if (dcta && warp == CONSUMERS && lane == 0) {
        const int nj = MIX / gt->dots_z, jz = dz * nj;
        mbar_expect_tx(&gbar, (uint32_t)(nj * HC * SLICE * 4));
        float* fns = (float*)xs;
        for (int j = jz; j < jz + nj; ++j)
            for (int c = 0; c < HC; ++c)
                tma_load_1d(fns + ((size_t)(j - jz) * HC + c) * SLICE, gt->dots_fn + (size_t)j * HC * gt->mx.dim + (size_t)c * gt->mx.dim + dslice * SLICE, SLICE * 4, &gbar);
    }
    const bool pre = fr && dyn && p.w_sh && !(gate && (gcta || dcta || mcta));
    if (pre && warp == CONSUMERS && lane == 0) {
        nxt = (int)atomicAdd(p.sched, 1u);
        while (nxt < quads && pg < STAGES) {
            st0 = issue(nxt, 0, STAGES);
            if (st0 < nst_sh) break;
            st0 = 0;
            nxt = (int)atomicAdd(p.sched, 1u);
        }
    }
    if (fr) {
        for (int e = tid; e < p.r_nexp; e += blockDim.x) sp.bias[e] = p.r_bias[e];
        for (int i = tid; i < 384 * 3; i += blockDim.x) (&sp.emask[0][0])[i] = 0u;
        if (tid < 3) sp.fmask[tid] = 0u;
    }
    if (mcta) moe_mixf_arm(*gt, p.M);
    pdl_wait();
    pdl_launch();
    bf16* pw = (bf16*)(red + 4 * MT * 64);
    if (MODE == MODE_SWIGLU && warp == CONSUMERS && lane == 0) {
        if (p.xq) {
            const uint32_t xb = (uint32_t)p.M * LDS * 2, sb = (uint32_t)nkb * MT * 4;
            mbar_expect_tx(&xbar, xb + sb);
            tma_load_1d(xs, p.xq, xb, &xbar);
            tma_load_1d(sa, p.xsq, sb, &xbar);
        } else {
            if (gcta) {
                mbar_expect_tx(&gbar, (uint32_t)(p.M + 1) * KSg * 2);
                for (int m = 0; m < p.M; ++m) tma_load_1d(pw + m * LDSg, p.x + (size_t)m * p.ldx + gks * KSg, KSg * 2, &gbar);
                tma_load_1d(red, p.prenorm_w + gks * KSg, KSg * 2, &gbar);
            }
            if (!gate) {
                mbar_expect_tx(&xbar, (uint32_t)(p.M + (p.prenorm_w ? 1 : 0)) * p.K * 2);
                for (int m = 0; m < p.M; ++m) tma_load_1d(xs + m * LDS, p.x + (size_t)m * p.ldx, p.K * 2, &xbar);
                if (p.prenorm_w) tma_load_1d(pw, p.prenorm_w, p.K * 2, &xbar);
            }
        }
    }
    __shared__ float rsr[16];
    auto row_stats = [&](int n, auto rowof) {
        if (warp < MT) {
            float ss = 0.f;
            if (warp < n) {
                const int ns = p.norm_nslices;
                float v = 0.f;
                for (int sl = lane; sl < ns; sl += 32) v = __ldcg(p.norm_part + ((size_t)rowof(warp) * ns + sl) * 26 + 25);
                for (int sl = 0; sl < ns; ++sl) ss += __shfl_sync(0xffffffffu, v, sl);
            }
            if (lane == 0) rsr[warp] = rsqrtf(ss / p.K + p.prenorm_eps);
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    };
    auto stage = [&](int n, auto rowof, bool in_smem, bool stats = true) {
        if (p.prenorm_w && stats) row_stats(n, rowof);
        quant_blocks8(warp, CONSUMERS, lane, MT, n, nkb, xs, LDS, sa, [&](int m, int kb, int c, float* v) {
            const int k = kb * 32 + 8 * c;
            const uint4 q = in_smem ? *(const uint4*)(xs + m * LDS + k) : *(const uint4*)(p.x + (size_t)rowof(m) * p.ldx + k);
            const uint4 gq = p.prenorm_w ? (in_smem ? *(const uint4*)(pw + k) : *(const uint4*)(p.prenorm_w + k)) : make_uint4(0, 0, 0, 0);
            const uint32_t w[4] = {q.x, q.y, q.z, q.w}, gw[4] = {gq.x, gq.y, gq.z, gq.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float2 f = bf16x2_to_f2(w[e]);
                if (p.prenorm_w) {
                    const float2 g2 = bf16x2_to_f2(gw[e]);
                    f.x = __bfloat162float(__float2bfloat16(g2.x * (f.x * rsr[m])));
                    f.y = __bfloat162float(__float2bfloat16(g2.y * (f.y * rsr[m])));
                }
                v[2 * e] = f.x;
                v[2 * e + 1] = f.y;
            }
        });
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    };
    if constexpr (gate) {
        if (dcta && warp < CONSUMERS) {
            moe_dots_item<CONSUMERS>(*gt, p.M, dslice, dz, (const float*)xs, (float*)(xs + 2 * (MIX / gt->dots_z) * HC * SLICE), &gbar);
        }
        if (mcta) moe_mix_item(*gt, p.M, nd, red);
        if (gcta && warp < CONSUMERS) {
            row_stats(p.M, [&](int m) { return m; });
            moe_gate_item<CONSUMERS, MT>(p, *gt, xcta, gq, gks, KSg, ntg, SS16, nst_g, SS * STEP, (uint16_t*)pw, (const bf16*)red, rsr, ring, full, empty, &gbar, red);
        }
        float* s_scores = (float*)pw;
        moe_gate_scores<MT>(p, *gt, ng, LDS, nkb, xs, sa, &xbar, s_scores);
        route_plan(p, sp, (MoePlan*)plan, p.M, p.w_sh != nullptr, s_scores, 1);
    } else if (fr) {
        route_plan(p, sp, (MoePlan*)plan, p.M, p.w_sh != nullptr, p.r_scores, p.r_nsum);
    }
    if (FUSED && warp == CONSUMERS + 1) {
        if (lane == 0) {
            for (int k = 0;; ++k) {
                mbar_wait(&epi_bar, k & 1);
                const int sl = s_pend[k & 1];
                if (sl < 0) break;
                asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(q->done + sl) : "memory");
                mbar_arrive(&epi_ack);
            }
        }
        __syncwarp();
        asm volatile("bar.sync 3, %0;" ::"r"((CONSUMERS + 1) * 32));
        return;
    }
    const int Q2 = FUSED ? q->N / 64 : 4, ns2 = FUSED ? q->K / 64 : 0, h0 = (ns2 + 1) / 2;
    const int shp = FUSED && p.w_sh ? 1 : 0;
    const int qpi = 4, qsh = FUSED ? q->qps : 2;
    const int nstg = FUSED ? 4 / q->qps : 2;
    const int T13 = pl_nslots() * quads, nsh2 = shp ? Q2 / qsh : 0;
    const int rq2 = FUSED ? Q2 / qpi : 1;
    const int T = T13 + (FUSED ? nsh2 + (pl_nslots() - shp) * rq2 : 0), G = gridDim.x;
    const int nr2 = (pl_nslots() - shp) * rq2;
    auto item2 = [&](int it, int& sl, int& q0) -> bool {
        const int r = it - T13;
        if (r >= nr2) { sl = 0; q0 = qsh * (r - nr2); return true; }
        sl = shp + r / rq2;
        q0 = (r % rq2) * qpi;
        return false;
    };
    const int i0 = dyn ? 0 : (int)((long long)blockIdx.x * T / G), i1 = dyn ? T : (int)((long long)(blockIdx.x + 1) * T / G);
    if (!dyn && i0 >= i1) return;
    if (warp == CONSUMERS) {
        if (lane == 0) {
            int nw2 = 0;
            int w2p = -1;
            bool xs_ok = false;
            (void)nw2; (void)w2p; (void)xs_ok;
            if (!pre) nxt = dyn ? (int)atomicAdd(p.sched, 1u) : i0;
            for (int it = i0;; ++it) {
                if (dyn) {
                    it = nxt;
                    if (it < i1) nxt = (int)atomicAdd(p.sched, 1u);
                }
                if (it >= i1) {
                    if (dyn) {
                        const int s = slot_of(pg, w2p);
                        const int u = uses_of(s, pg, w2p);
                        if (u >= 1) mbar_wait(&empty[s], (u - 1) & 1);
                        sitem[s] = -1;
                        mbar_arrive(&full[s]);
                    }
                    break;
                }
                if (FUSED && it >= T13) {
                    int sl, q0;
                    const bool sh = item2(it, sl, q0);
                    if (w2p < 0) w2p = 0;
                    const int nst_it = sh ? 2 : nstg;
                    for (int qi = 0; qi < nst_it; ++qi) {
                        const int s = slot_of(pg, w2p);
                        const int u = uses_of(s, pg, w2p);
                        if (s >= STAGES && !xs_ok) { mbar_wait(&xsfree, 0); xs_ok = true; }
                        if (u >= 1) mbar_wait(&empty[s], (u - 1) & 1);
                        if (s < STAGES) ++pg;
                        ++w2p;
                        sitem[s] = it;
                        uint32_t bytes;
                        const uint8_t* src;
                        if (q->qps == 2) {
                            bytes = sh ? ns2 * STEP8 : 2 * ns2 * STEP;
                            src = sh ? q->w_sh + (size_t)(q0 + qi) * ns2 * STEP8
                                     : q->w + (size_t)pl_expert(sl) * q->w_stride + (size_t)(q0 + 2 * qi) * ns2 * STEP;
                        } else if (sh) {
                            bytes = (uint32_t)(qi ? ns2 - h0 : h0) * STEP8;
                            src = q->w_sh + ((size_t)q0 * ns2 + qi * h0) * STEP8;
                        } else {
                            bytes = ns2 * STEP;
                            src = q->w + (size_t)pl_expert(sl) * q->w_stride + (size_t)(q0 + qi) * ns2 * STEP;
                        }
                        mbar_expect_tx(&full[s], bytes);
                        tma_load_1d(slot_base(s), src, bytes, &full[s]);
                    }
                    const int b = nw2 & 1;
                    if (nw2 >= 2) mbar_wait(&x2empty[b], ((nw2 >> 1) - 1) & 1);
                    for (;;) {
                        unsigned d;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(d) : "l"(q->done + sl) : "memory");
                        if (d >= (unsigned)quads) break;
                        __nanosleep(32);
                    }
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    const int cnt = pl_cnt(sl);
                    mbar_expect_tx(&x2full[b], (uint32_t)cnt * (LDS2 * 2 + q->sq_ld * 4));
                    for (int m = 0; m < cnt; ++m) {
                        const int pr = pl_pair(sl, m);
                        tma_load_1d(xs2 + (size_t)(b * MT + m) * LDS2, p.oq + (size_t)pr * LDS2, LDS2 * 2, &x2full[b]);
                        tma_load_1d(sa2 + (size_t)(b * MT + m) * q->sq_ld, p.osq + (size_t)pr * q->sq_ld, q->sq_ld * 4, &x2full[b]);
                    }
                    ++nw2;
                } else {
                    issue(it, st0, 0x7fffffff);
                }
                st0 = 0;
            }
        }
        return;
    }
    const int M = p.M;
    if (MODE == MODE_SWIGLU) {
        mbar_wait(&xbar, 0);
        if (!p.xq && !gate) stage(M, [&](int m) { return m; }, true);
    }
    const int g = lane >> 2, t = lane & 3;
    const int rg = warp % 4, ph = warp / 4;
    int gs = nst_g;
    int staged = -1;
    int cw2 = 0;
    (void)cw2;
    int n_epi = 0;
    __shared__ float wred2[FUSED ? 2 : 1][4][32][4 * NT];
    int w2r = -1;
    for (int it = i0; dyn || it < i1; ++it) {
        if (dyn) {
            const int s0 = slot_of(gs, w2r);
            mbar_wait(&full[s0], uses_of(s0, gs, w2r) & 1);
            it = sitem[s0];
            if (FUSED) asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
            if (it < 0) break;
            if (FUSED && it >= T13 && w2r < 0) {
                w2r = 0;
                __syncwarp();
                if (lane == 0) mbar_arrive(&xsfree);
            }
        }
        if constexpr (FUSED) {
            if (it >= T13) {
                int sl, q0;
                const bool sh = item2(it, sl, q0);
                const int cnt = pl_cnt(sl);
                const int b = cw2 & 1;
                mbar_wait(&x2full[b], (cw2 >> 1) & 1);
                const uint16_t* xb2 = xs2 + (size_t)b * MT * LDS2;
                const float* sb2 = sa2 + (size_t)b * MT * q->sq_ld;
                const int sq = q->sq_ld;
                const bool one = q->qps == 1;
                const int qi = ph >> 1, hk = ph & 1;
                const bool active = !(one && sh) || ph < 2;
                const int st2 = one ? (sh ? hk : ph) : qi;
                const int nst_it = sh ? 2 : nstg;
                int sl_[4], us_[4], gsx = gs;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sl_[i] = slot_of(gsx, w2r + i);
                    us_[i] = uses_of(sl_[i], gsx, w2r + i);
                    if (i < nst_it && sl_[i] < STAGES) ++gsx;
                }
                int s = sl_[0], us2 = us_[0];
#pragma unroll
                for (int i = 1; i < 4; ++i)
                    if (st2 == i) s = sl_[i], us2 = us_[i];
                if (active) mbar_wait(&full[s], us2 & 1);
                const uint8_t* base = slot_base(s);
                float acc[NT][4];
#pragma unroll
                for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
                int quad2 = q0;
                const int j0 = sh ? hk * h0 : 0, j1 = sh ? (hk ? ns2 : h0) : ns2;
                const int jb = one && sh ? j0 : 0;
                if (sh && active) {
                    quad2 = one ? q0 : q0 + qi;
                    for (int j = j0; j < j1; ++j) {
                        const uint8_t* sp = base + (size_t)(j - jb) * STEP8;
                        const int kl = j * 64;
                        const uint32_t sc2 = *(const uint16_t*)(sp + 4096 + (rg >> 1) * 2);
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const uint4 v = *(const uint4*)(sp + h * 2048 + rg * 512 + lane * 16);
                            const float sw = e8m0_to_f32((sc2 >> (8 * h)) & 0xff);
                            const uint32_t a0[4] = {e4m3x2_to_f16x2(v.x & 0xffff), e4m3x2_to_f16x2(v.z & 0xffff),
                                                    e4m3x2_to_f16x2(v.x >> 16), e4m3x2_to_f16x2(v.z >> 16)};
                            const uint32_t a1[4] = {e4m3x2_to_f16x2(v.y & 0xffff), e4m3x2_to_f16x2(v.w & 0xffff),
                                                    e4m3x2_to_f16x2(v.y >> 16), e4m3x2_to_f16x2(v.w >> 16)};
#pragma unroll
                            for (int n = 0; n < NT; ++n) {
                                const uint4 xb = *(const uint4*)(xb2 + (n * 8 + g) * LDS2 + kl + h * 32 + 8 * t);
                                float c[4] = {0.f, 0.f, 0.f, 0.f};
                                mma_f16(c, a0, xb.x, xb.y);
                                mma_f16(c, a1, xb.z, xb.w);
                                acc[n][0] = fmaf(c[0], sw, acc[n][0]);
                                acc[n][1] = fmaf(c[1], sw, acc[n][1]);
                                acc[n][2] = fmaf(c[2], sw, acc[n][2]);
                                acc[n][3] = fmaf(c[3], sw, acc[n][3]);
                            }
                        }
                    }
                } else if (!sh) {
                    quad2 = q0 + ph;
                    const uint8_t* qb = one ? base : base + (size_t)hk * ns2 * STEP;
                    for (int j = j0; j < j1; ++j) {
                        const uint4 v = *(const uint4*)(qb + (size_t)j * STEP + rg * 512 + lane * 16);
                        const uint32_t sc = *(const uint32_t*)(qb + (size_t)j * STEP + 2048 + rg * 32 + g * 4);
                        const int kl = j * 64;
                        const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const float sg = e8m0_to_f32((sc >> (8 * h)) & 0xff), shs = e8m0_to_f32((sc >> (16 + 8 * h)) & 0xff);
                            float c[NT][4];
                            uint4 xb[NT];
#pragma unroll
                            for (int n = 0; n < NT; ++n) {
                                c[n][0] = c[n][1] = c[n][2] = c[n][3] = 0.f;
                                xb[n] = *(const uint4*)(xb2 + (n * 8 + g) * LDS2 + kl + h * 32 + 8 * t);
                            }
                            uint32_t a[4];
                            cvt4_e2m1(wv[h * 2], a);
#pragma unroll
                            for (int n = 0; n < NT; ++n) mma_f16(c[n], a, xb[n].x, xb[n].y);
                            cvt4_e2m1(wv[h * 2 + 1], a);
#pragma unroll
                            for (int n = 0; n < NT; ++n) mma_f16(c[n], a, xb[n].z, xb[n].w);
#pragma unroll
                            for (int n = 0; n < NT; ++n) {
                                acc[n][0] = fmaf(c[n][0], sg, acc[n][0]);
                                acc[n][1] = fmaf(c[n][1], sg, acc[n][1]);
                                acc[n][2] = fmaf(c[n][2], shs, acc[n][2]);
                                acc[n][3] = fmaf(c[n][3], shs, acc[n][3]);
                            }
                        }
                    }
                }
                __syncwarp();
                if (lane == 0) {
                    if (active) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(&empty[s])), "r"(one ? 4 : 2) : "memory");
                    mbar_arrive(&x2empty[b]);
                }
                gs = gsx;
                w2r += nst_it;
                ++cw2;
                const bool halves = sh;
                if (halves) {
                    if (hk && active) {
#pragma unroll
                        for (int n = 0; n < NT; ++n)
#pragma unroll
                            for (int e = 0; e < 4; ++e) wred2[qi][rg][lane][n * 4 + e] = acc[n][e];
                    }
                    asm volatile("bar.sync 2, %0;" ::"r"(CONSUMERS * 32));
                    if (!hk && active) {
#pragma unroll
                        for (int n = 0; n < NT; ++n)
#pragma unroll
                            for (int e = 0; e < 4; ++e) acc[n][e] += wred2[qi][rg][lane][n * 4 + e];
                    }
                }
                if (active && (!halves || !hk)) {
                    const int r = quad2 * 64 + rg * 16 + g;
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        const int m0 = n * 8 + 2 * t;
                        if (m0 < cnt) {
                            float* o = q->out + (size_t)pl_pair(sl, m0) * q->ldo + r;
                            o[0] = acc[n][0] * q->alpha;
                            o[8] = acc[n][2] * q->alpha;
                        }
                        if (m0 + 1 < cnt) {
                            float* o = q->out + (size_t)pl_pair(sl, m0 + 1) * q->ldo + r;
                            o[0] = acc[n][1] * q->alpha;
                            o[8] = acc[n][3] * q->alpha;
                        }
                    }
                }
                if (halves) asm volatile("bar.sync 2, %0;" ::"r"(CONSUMERS * 32));
                continue;
            }
        }
        const int sl = it / quads, quad = it % quads;
        const int cnt = pl_cnt(sl);
        if (MODE != MODE_SWIGLU && sl != staged) {
            if (p.xq) {
                const int rq = (p.K + 32) / 8;
                for (int i = tid; i < cnt * rq; i += CONSUMERS * 32) {
                    const int m = i / rq, c = i - m * rq;
                    ((uint4*)(xs + m * LDS))[c] = __ldcg((const uint4*)(p.xq + (size_t)pl_pair(sl, m) * (p.K + 32)) + c);
                }
                for (int i = tid; i < cnt * nkb; i += CONSUMERS * 32) {
                    const int m = i / nkb, kb = i - m * nkb;
                    sa[kb * MT + m] = __ldcg(p.xsq + (size_t)pl_pair(sl, m) * nkb + kb);
                }
                asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
            } else {
                stage(cnt, [&](int m) { return pl_pair(sl, m); }, false);
            }
            staged = sl;
        }
        {
        float acc[NT][4];
        for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
        const bool shx = pl_expert(sl) < 0;
        for (int st = 0; st < (shx ? nst8 : nst); ++st, ++gs) {
            const int s = gs % STAGES;
            mbar_wait(&full[s], (gs / STAGES) & 1);
            const uint8_t* base = ring + (size_t)s * SS * STEP;
            if (shx) {
                const int steps = min(SS8, nsteps - st * SS8);
                for (int j = 0; j < steps; ++j) {
                    if ((FUSED ? ((st * SS8 + j) % 6) % PH : j % PH) != ph) continue;
                    const uint8_t* sp = base + (size_t)j * STEP8;
                    const int kl = (st * SS8 + j) * TK;
                    const uint32_t sc2 = *(const uint16_t*)(sp + 4096 + (rg >> 1) * 2);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint4 v = *(const uint4*)(sp + h * 2048 + rg * 512 + lane * 16);
                        const float sw = e8m0_to_f32((sc2 >> (8 * h)) & 0xff);
                        const uint32_t a0[4] = {e4m3x2_to_f16x2(v.x & 0xffff), e4m3x2_to_f16x2(v.z & 0xffff),
                                                e4m3x2_to_f16x2(v.x >> 16), e4m3x2_to_f16x2(v.z >> 16)};
                        const uint32_t a1[4] = {e4m3x2_to_f16x2(v.y & 0xffff), e4m3x2_to_f16x2(v.w & 0xffff),
                                                e4m3x2_to_f16x2(v.y >> 16), e4m3x2_to_f16x2(v.w >> 16)};
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            const uint4 xb = *(const uint4*)(xs + (n * 8 + g) * LDS + kl + h * 32 + 8 * t);
                            float c[4] = {0.f, 0.f, 0.f, 0.f};
                            mma_f16(c, a0, xb.x, xb.y);
                            mma_f16(c, a1, xb.z, xb.w);
                            acc[n][0] = fmaf(c[0], sw, acc[n][0]);
                            acc[n][1] = fmaf(c[1], sw, acc[n][1]);
                            acc[n][2] = fmaf(c[2], sw, acc[n][2]);
                            acc[n][3] = fmaf(c[3], sw, acc[n][3]);
                        }
                    }
                }
                __syncwarp();
                if (lane == 0) mbar_arrive(&empty[s]);
                continue;
            }
            const int steps = min(SS, nsteps - st * SS);
            auto mma_step = [&](int j, float (&c)[2][NT][4], uint32_t& sc) {
                const uint4 v = *(const uint4*)(base + (size_t)j * STEP + rg * 512 + lane * 16);
                sc = *(const uint32_t*)(base + (size_t)j * STEP + 2048 + rg * 32 + g * 4);
                const int kl = (st * SS + j) * TK;
                const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int n = 0; n < NT; ++n) c[h][n][0] = c[h][n][1] = c[h][n][2] = c[h][n][3] = 0.f;
                    uint4 xb[NT];
#pragma unroll
                    for (int n = 0; n < NT; ++n) xb[n] = *(const uint4*)(xs + (n * 8 + g) * LDS + kl + h * 32 + 8 * t);
                    uint32_t a[4];
                    cvt4_e2m1(wv[h * 2], a);
#pragma unroll
                    for (int n = 0; n < NT; ++n) mma_f16(c[h][n], a, xb[n].x, xb[n].y);
                    cvt4_e2m1(wv[h * 2 + 1], a);
#pragma unroll
                    for (int n = 0; n < NT; ++n) mma_f16(c[h][n], a, xb[n].z, xb[n].w);
                }
            };
            auto acc_step = [&](const float (&c)[2][NT][4], uint32_t sc) {
                const float sg[2] = {e8m0_to_f32(sc & 0xff), e8m0_to_f32((sc >> 8) & 0xff)};
                const float sh[2] = {e8m0_to_f32((sc >> 16) & 0xff), e8m0_to_f32(sc >> 24)};
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        acc[n][0] = fmaf(c[h][n][0], sg[h], acc[n][0]);
                        acc[n][1] = fmaf(c[h][n][1], sg[h], acc[n][1]);
                        acc[n][2] = fmaf(c[h][n][2], sh[h], acc[n][2]);
                        acc[n][3] = fmaf(c[h][n][3], sh[h], acc[n][3]);
                    }
            };
            int j = FUSED ? ((ph - (st * SS) % PH) + PH) % PH : ph;
            for (; j + PH < steps; j += 2 * PH) {
                float c0[2][NT][4], c1[2][NT][4];
                uint32_t s0, s1;
                mma_step(j, c0, s0);
                mma_step(j + PH, c1, s1);
                acc_step(c0, s0);
                acc_step(c1, s1);
            }
            if (j < steps) {
                float c0[2][NT][4];
                uint32_t s0;
                mma_step(j, c0, s0);
                acc_step(c0, s0);
            }

            __syncwarp();
            if (lane == 0) mbar_arrive(&empty[s]);
        }
        float* rp = red + (size_t)ph * MT * 64;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int m0 = n * 8 + 2 * t, r = rg * 16 + g;
            rp[m0 * 64 + r] = acc[n][0];
            rp[(m0 + 1) * 64 + r] = acc[n][1];
            rp[m0 * 64 + r + 8] = acc[n][2];
            rp[(m0 + 1) * 64 + r + 8] = acc[n][3];
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        for (int i = tid; i < MT * 64; i += CONSUMERS * 32) {
            float v = red[i];
#pragma unroll
            for (int q = 1; q < PH; ++q) v += red[(size_t)q * MT * 64 + i];
            red[i] = v;
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        if (MODE == MODE_SWIGLU) {
            const float L = p.swiglu_limit;
            for (int k = warp; k < cnt; k += CONSUMERS) {
                const int j = lane;
                const int col = pl_tok(sl, k), pr = pl_pair(sl, k);
                float gte = __bfloat162float(__float2bfloat16(red[col * 64 + j]));
                float up = __bfloat162float(__float2bfloat16(red[col * 64 + 32 + j]));
                if (L > 0.f) { up = fminf(fmaxf(up, -L), L); gte = fminf(gte, L); }
                const float a = gte / (1.0f + expf(-gte)) * up * pl_w(sl, k);
                const bf16 ab = __float2bfloat16(a);
                ((bf16*)p.out)[(size_t)pr * p.ldo + quad * 32 + j] = ab;
                if (p.oq) {
                    const float af = __bfloat162float(ab);
                    const float sc = round_scale_pow2(fmaxf(warp_max(fabsf(af)), 1e-4f), 1.0f / 448.0f);
                    const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
                    const float q = af * inv, q1 = __shfl_down_sync(0xffffffff, q, 1);
                    const int act_n = (p.N / 2 + 63) / 64 * 64;
                    if ((j & 1) == 0)
                        *(uint32_t*)(p.oq + (size_t)pr * (act_n + 32) + quad * 32 + kpos32(j)) = f16x2_dequant(e4m3x2_to_f16x2(f32x2_to_e4m3x2(q, q1)), sc);
                    if (j == 0) p.osq[(size_t)pr * (FUSED ? sq_ld1 : act_n / 32) + quad] = sc;
                }
            }
        } else {
            for (int i = tid; i < cnt * 64; i += CONSUMERS * 32) {
                const int k = i / 64, r = i % 64;
                ((float*)p.out)[(size_t)pl_pair(sl, k) * p.ldo + quad * 64 + r] = red[k * 64 + r] * p.alpha;
            }
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        if constexpr (FUSED) {
            if (lane == 0) {
                if (n_epi > 0) mbar_wait(&epi_ack, (n_epi - 1) & 1);
                if (tid == 0) s_pend[n_epi & 1] = sl;
                mbar_arrive(&epi_bar);
            }
            __syncwarp();
            ++n_epi;
        }
        }
    }
    if constexpr (FUSED) {
        if (lane == 0) {
            if (n_epi > 0) mbar_wait(&epi_ack, (n_epi - 1) & 1);
            if (tid == 0) s_pend[n_epi & 1] = -1;
            mbar_arrive(&epi_bar);
        }
        __syncwarp();
        asm volatile("bar.sync 3, %0;" ::"r"((CONSUMERS + 1) * 32));
    }
    if (dyn) {
        if (tid == 0) {
            __threadfence();
            if (atomicAdd(p.sched + 1, 1u) == gridDim.x - 1) {
                p.sched[0] = 0;
                p.sched[1] = 0;
                if (gate) { gt->ctr[0] = 0; gt->ctr[1] = 0; }
                if (dots) *gt->dots_ctr = 0;
                if (FUSED)
                    for (int i = 0; i < pl_nslots(); ++i) q->done[i] = 0;
                __threadfence();
            }
        }
    }
}

#ifndef GEMV_NO_ENTRIES
#define MOE_ENTRY(name, NT, MODE, PH) \
    extern "C" __global__ void __launch_bounds__((4 * PH + 1) * 32) name(const __grid_constant__ GemvParams p) { moe_body<NT, MODE, PH>(p, p.ksplit_slots); }
MOE_ENTRY(moe_fp4_swiglu_n1_p4, 1, MODE_SWIGLU, 4)
MOE_ENTRY(moe_fp4_swiglu_n2_p4, 2, MODE_SWIGLU, 4)
MOE_ENTRY(moe_fp4_store_n1_p2, 1, MODE_STORE, 2)
MOE_ENTRY(moe_fp4_store_n2_p2, 2, MODE_STORE, 2)
MOE_ENTRY(moe_fp4_store_n1_p4, 1, MODE_STORE, 4)
MOE_ENTRY(moe_fp4_store_n2_p4, 2, MODE_STORE, 4)
extern "C" __global__ void __launch_bounds__(18 * 32) moe_fp4_fused_n1(const __grid_constant__ GemvParams p, const __grid_constant__ W2Params q) {
    moe_body<1, MODE_SWIGLU, 4, true>(p, 0, &q);
}
extern "C" __global__ void __launch_bounds__(18 * 32) moe_fp4_fused_gate_n1(const __grid_constant__ GemvParams p, const __grid_constant__ W2Params q, const __grid_constant__ GateParams g) {
    moe_body<1, MODE_SWIGLU, 4, true, true>(p, 0, &q, &g);
}
#endif

constexpr int W2_STAGES = 4;
template <int NT>
DEV void moe_w2_body(const GemvParams& p) {
    constexpr int MT = 8 * NT, CONS = 8, STEP = 2048 + 128;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int quads = p.N / 64, nsteps = p.K / 64, nkb = p.K / 32, LDS = p.K + 32;
    const int qbytes = nsteps * STEP;
    const int h0 = (nsteps + 1) / 2;
    const int QB = max(qbytes, h0 * STEP8);
    uint8_t* ring = smem;
    uint16_t* xs = (uint16_t*)(ring + (size_t)W2_STAGES * QB);
    float* sa = (float*)(xs + MT * LDS);
    __shared__ __align__(8) uint64_t full[W2_STAGES], empty[W2_STAGES];
    if (tid == 0) {
        for (int s = 0; s < W2_STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 4); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    pdl_wait();
    pdl_launch();
    const MoePlan* plan = p.plan;
    const int pairs = (quads + 1) / 2;
    const int shp = plan->n_slots > 0 && plan->expert[0] < 0;
    const int nsh = shp ? quads : 0;
    const int T = (plan->n_slots - shp) * pairs + nsh, G = gridDim.x;
    auto item = [&](int it, int& sl, int& q0) -> bool {
        if (it < nsh) { sl = 0; q0 = it; return true; }
        it -= nsh;
        sl = shp + it / pairs;
        q0 = (it % pairs) * 2;
        return false;
    };
    const int i0 = (int)((long long)blockIdx.x * T / G), i1 = (int)((long long)(blockIdx.x + 1) * T / G);
    if (i0 >= i1) return;
    if (warp == CONS) {
        if (lane == 0) {
            int g = 0;
            for (int it = i0; it < i1; ++it) {
                int sl, q0;
                const bool sh = item(it, sl, q0);
                for (int qi = 0; qi < 2; ++qi, ++g) {
                    const int s = g % W2_STAGES;
                    if (g >= W2_STAGES) mbar_wait(&empty[s], ((g / W2_STAGES) - 1) & 1);
                    if (sh) {
                        const int steps = qi == 0 ? h0 : nsteps - h0;
                        if (steps == 0) { mbar_arrive(&full[s]); continue; }
                        mbar_expect_tx(&full[s], (uint32_t)(steps * STEP8));
                        tma_load_1d(ring + (size_t)s * QB, p.w_sh + ((size_t)q0 * nsteps + qi * h0) * STEP8, (uint32_t)(steps * STEP8), &full[s]);
                        continue;
                    }
                    if (q0 + qi >= quads) { mbar_arrive(&full[s]); continue; }
                    mbar_expect_tx(&full[s], (uint32_t)qbytes);
                    tma_load_1d(ring + (size_t)s * QB, p.w + (size_t)plan->expert[sl] * p.w_stride + (size_t)(q0 + qi) * qbytes, (uint32_t)qbytes, &full[s]);
                }
            }
        }
        return;
    }
    const int qi = warp / 4, rg = warp % 4, g = lane >> 2, t = lane & 3;
    int staged = -1, gq = 0;
    __shared__ float wred[4][32][4 * NT];
    for (int it = i0; it < i1; ++it, gq += 2) {
        int sl, q0;
        const bool sh = item(it, sl, q0);
        const int quad = sh ? q0 : q0 + qi;
        const int cnt = plan->cnt[sl];
        if (sl != staged) {
            asm volatile("bar.sync 1, %0;" ::"r"(CONS * 32));
            const int rq = LDS / 8;
            for (int i = tid; i < cnt * rq; i += CONS * 32) {
                const int m = i / rq, c = i - m * rq;
                ((uint4*)(xs + m * LDS))[c] = __ldcg((const uint4*)(p.xq + (size_t)plan->pair[sl][m] * LDS) + c);
            }
            for (int i = tid; i < cnt * nkb; i += CONS * 32) {
                const int m = i / nkb, kb = i - m * nkb;
                sa[kb * MT + m] = __ldcg(p.xsq + (size_t)plan->pair[sl][m] * nkb + kb);
            }
            asm volatile("bar.sync 1, %0;" ::"r"(CONS * 32));
            staged = sl;
        }
        const int s = (gq + qi) % W2_STAGES;
        mbar_wait(&full[s], ((gq + qi) / W2_STAGES) & 1);
        if (sh) {
            const uint8_t* base = ring + (size_t)s * QB;
            const int j0 = qi * h0, j1 = qi == 0 ? h0 : nsteps;
            float acc[NT][4];
#pragma unroll
            for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
            for (int j = j0; j < j1; ++j) {
                const uint8_t* sp = base + (size_t)(j - j0) * STEP8;
                const int kl = j * 64;
                const uint32_t sc2 = *(const uint16_t*)(sp + 4096 + (rg >> 1) * 2);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 v = *(const uint4*)(sp + h * 2048 + rg * 512 + lane * 16);
                    const float sw = e8m0_to_f32((sc2 >> (8 * h)) & 0xff);
                    const uint32_t a0[4] = {e4m3x2_to_f16x2(v.x & 0xffff), e4m3x2_to_f16x2(v.z & 0xffff),
                                            e4m3x2_to_f16x2(v.x >> 16), e4m3x2_to_f16x2(v.z >> 16)};
                    const uint32_t a1[4] = {e4m3x2_to_f16x2(v.y & 0xffff), e4m3x2_to_f16x2(v.w & 0xffff),
                                            e4m3x2_to_f16x2(v.y >> 16), e4m3x2_to_f16x2(v.w >> 16)};
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        const uint4 xb = *(const uint4*)(xs + (n * 8 + g) * LDS + kl + h * 32 + 8 * t);
                        float c[4] = {0.f, 0.f, 0.f, 0.f};
                        mma_f16(c, a0, xb.x, xb.y);
                        mma_f16(c, a1, xb.z, xb.w);
                        acc[n][0] = fmaf(c[0], sw, acc[n][0]);
                        acc[n][1] = fmaf(c[1], sw, acc[n][1]);
                        acc[n][2] = fmaf(c[2], sw, acc[n][2]);
                        acc[n][3] = fmaf(c[3], sw, acc[n][3]);
                    }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty[s]);
            if (qi == 1) {
#pragma unroll
                for (int n = 0; n < NT; ++n)
#pragma unroll
                    for (int q = 0; q < 4; ++q) wred[rg][lane][n * 4 + q] = acc[n][q];
            }
            asm volatile("bar.sync 2, %0;" ::"r"(CONS * 32));
            if (qi == 0) {
                const int r = quad * 64 + rg * 16 + g;
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    float v[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) v[q] = acc[n][q] + wred[rg][lane][n * 4 + q];
                    const int m0 = n * 8 + 2 * t;
                    if (m0 < cnt) {
                        float* o = (float*)p.out + (size_t)plan->pair[sl][m0] * p.ldo + r;
                        o[0] = v[0] * p.alpha;
                        o[8] = v[2] * p.alpha;
                    }
                    if (m0 + 1 < cnt) {
                        float* o = (float*)p.out + (size_t)plan->pair[sl][m0 + 1] * p.ldo + r;
                        o[0] = v[1] * p.alpha;
                        o[8] = v[3] * p.alpha;
                    }
                }
            }
            asm volatile("bar.sync 2, %0;" ::"r"(CONS * 32));
            continue;
        }
        if (quad < quads) {
            const uint8_t* base = ring + (size_t)s * QB;
            float acc[NT][4];
#pragma unroll
            for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
            for (int j = 0; j < nsteps; ++j) {
                const uint4 v = *(const uint4*)(base + (size_t)j * STEP + rg * 512 + lane * 16);
                const uint32_t sc = *(const uint32_t*)(base + (size_t)j * STEP + 2048 + rg * 32 + g * 4);
                const int kl = j * 64;
                const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float sg = e8m0_to_f32((sc >> (8 * h)) & 0xff), sh = e8m0_to_f32((sc >> (16 + 8 * h)) & 0xff);
                    float c[NT][4];
                    uint4 xb[NT];
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        c[n][0] = c[n][1] = c[n][2] = c[n][3] = 0.f;
                        xb[n] = *(const uint4*)(xs + (n * 8 + g) * LDS + kl + h * 32 + 8 * t);
                    }
                    uint32_t a[4];
                    cvt4_e2m1(wv[h * 2], a);
#pragma unroll
                    for (int n = 0; n < NT; ++n) mma_f16(c[n], a, xb[n].x, xb[n].y);
                    cvt4_e2m1(wv[h * 2 + 1], a);
#pragma unroll
                    for (int n = 0; n < NT; ++n) mma_f16(c[n], a, xb[n].z, xb[n].w);
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        acc[n][0] = fmaf(c[n][0], sg, acc[n][0]);
                        acc[n][1] = fmaf(c[n][1], sg, acc[n][1]);
                        acc[n][2] = fmaf(c[n][2], sh, acc[n][2]);
                        acc[n][3] = fmaf(c[n][3], sh, acc[n][3]);
                    }
                }
            }
            const int r = quad * 64 + rg * 16 + g;
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int m0 = n * 8 + 2 * t;
                if (m0 < cnt) {
                    float* o = (float*)p.out + (size_t)plan->pair[sl][m0] * p.ldo + r;
                    o[0] = acc[n][0] * p.alpha;
                    o[8] = acc[n][2] * p.alpha;
                }
                if (m0 + 1 < cnt) {
                    float* o = (float*)p.out + (size_t)plan->pair[sl][m0 + 1] * p.ldo + r;
                    o[0] = acc[n][1] * p.alpha;
                    o[8] = acc[n][3] * p.alpha;
                }
            }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
    }
}
#ifndef GEMV_NO_ENTRIES
extern "C" __global__ void __launch_bounds__(9 * 32) moe_w2_n1(const __grid_constant__ GemvParams p) { moe_w2_body<1>(p); }
extern "C" __global__ void __launch_bounds__(9 * 32) moe_w2_n2(const __grid_constant__ GemvParams p) { moe_w2_body<2>(p); }
#endif
#endif
