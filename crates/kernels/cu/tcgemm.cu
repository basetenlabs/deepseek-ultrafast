#include "tc.cuh"

struct TgParams {
    const void* ta;
    const void* tb;
    const uint8_t* sfa;
    const uint8_t* sfb;
    void* out;
    int ldo, out_f32;
    int M, N, K, nt;
    float alpha;
    void* out2;
    int ldo2, n_split;
    int swiglu;
    float swiglu_limit;
};

constexpr int TG_STAGES = 4;

extern "C" __global__ void __launch_bounds__(128, 1) tc_gemm_fp8(const __grid_constant__ TgParams p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* ring = align1k(smem_raw);
    const int NT = p.nt, nsb = NT >= 128 ? NT / 128 : 1;
    const int OFF_B = 16384, OFF_SFA = OFF_B + NT * 128, OFF_SFB = OFF_SFA + 512;
    const int SB = (OFF_SFB + nsb * 512 + 1023) & ~1023;
    const int KT = p.K / 128;
    __shared__ __align__(8) uint64_t full[TG_STAGES], empty[TG_STAGES], done;
    __shared__ uint32_t tbase;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nt0 = blockIdx.x * NT, mt = blockIdx.y, m0 = mt * 128;
    const int sb0 = NT >= 128 ? nt0 / 128 : nt0 / 64;
    if (warp == 0) tmalloc(&tbase, 512);
    if (tid == 0) {
        for (int s = 0; s < TG_STAGES; ++s) { mb_init_t(&full[s], 1); mb_init_t(&empty[s], 1); }
        mb_init_t(&done, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    const uint32_t tm = tbase;
    if (warp == 0 && lane == 0) {
        const int pre = min(KT, TG_STAGES);
        for (int kt = 0; kt < KT; ++kt) {
            const int s = kt % TG_STAGES;
            if (kt >= TG_STAGES) mb_wait_t(&empty[s], ((kt / TG_STAGES) - 1) & 1);
            uint8_t* st = ring + (size_t)s * SB;
            mb_expect_t(&full[s], 16384 + NT * 128 + 512 + nsb * 512);
            tma2d(st + OFF_B, p.tb, kt * 128, nt0, &full[s]);
            for (int b = 0; b < nsb; ++b) bulk1d(st + OFF_SFB + b * 512, p.sfb + (((size_t)(sb0 + b) * KT) + kt) * 512, 512, &full[s]);
            if (kt >= pre) {
                tma2d(st, p.ta, kt * 128, m0, &full[s]);
                bulk1d(st + OFF_SFA, p.sfa + ((size_t)mt * KT + kt) * 512, 512, &full[s]);
            }
            if (kt == pre - 1) {
                pdl_wait();
                pdl_launch();
                for (int k2 = 0; k2 < pre; ++k2) {
                    uint8_t* s2 = ring + (size_t)k2 * SB;
                    tma2d(s2, p.ta, k2 * 128, m0, &full[k2]);
                    bulk1d(s2 + OFF_SFA, p.sfa + ((size_t)mt * KT + k2) * 512, 512, &full[k2]);
                }
            }
        }
    } else if (warp == 1 && lane == 0) {
        for (int kt = 0; kt < KT; ++kt) {
            const int s = kt % TG_STAGES;
            mb_wait_t(&full[s], (kt / TG_STAGES) & 1);
            tcafter();
            uint8_t* st = ring + (size_t)s * SB;
            const uint32_t t_sfa = tm + 256 + 4 * s, t_sfb = tm + 288 + 8 * s;
            tccpsf(t_sfa, sdesc_t(st + OFF_SFA, 512, 128, 0));
            for (int b = 0; b < nsb; ++b) tccpsf(t_sfb + 4 * b, sdesc_t(st + OFF_SFB + b * 512, 512, 128, 0));
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) {
                const uint64_t ad = sdesc_t(st + kc * 32, 16, 1024, 2);
                const uint64_t bd = sdesc_t(st + OFF_B + kc * 32, 16, 1024, 2);
                tcmma(tm, ad, bd, idesc_e4m3(NT, kc), t_sfa, t_sfb, (kt | kc) != 0);
            }
            tccommit(&empty[s]);
        }
        tccommit(&done);
    }
    mb_wait_t(&done, 0);
    tcafter();
    const int row = m0 + 32 * warp + lane;
    if (p.swiglu) {
        const float L = p.swiglu_limit;
        for (int c = 0; c < NT; c += 64) {
            float g[32], u[32];
            tcld32(tm + ((uint32_t)(32 * warp) << 16) + c, g);
            tcld32(tm + ((uint32_t)(32 * warp) << 16) + c + 32, u);
            if (row < p.M) {
                bf16* o = (bf16*)p.out + (size_t)row * p.ldo + (nt0 + c) / 2;
#pragma unroll
                for (int j = 0; j < 32; j += 8) {
                    uint4 q;
                    bf16* h = (bf16*)&q;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float gte = __bfloat162float(__float2bfloat16(g[j + e])), up = __bfloat162float(__float2bfloat16(u[j + e]));
                        if (L > 0.f) { up = fminf(fmaxf(up, -L), L); gte = fminf(gte, L); }
                        h[e] = __float2bfloat16(gte / (1.0f + expf(-gte)) * up);
                    }
                    *(uint4*)(o + j) = q;
                }
            }
        }
    }
    for (int c = 0; c < NT && !p.swiglu; c += 32) {
        float v[32];
        tcld32(tm + ((uint32_t)(32 * warp) << 16) + c, v);
        if (row < p.M) {
            const int col = nt0 + c;
            void* ob = col < p.n_split ? p.out : p.out2;
            const int ldo = col < p.n_split ? p.ldo : p.ldo2, oc = col < p.n_split ? col : col - p.n_split;
            if (p.out_f32) {
                float* o = (float*)ob + (size_t)row * ldo + oc;
#pragma unroll
                for (int j = 0; j < 32; j += 4) *(float4*)(o + j) = make_float4(v[j] * p.alpha, v[j + 1] * p.alpha, v[j + 2] * p.alpha, v[j + 3] * p.alpha);
            } else {
                bf16* o = (bf16*)ob + (size_t)row * ldo + oc;
#pragma unroll
                for (int j = 0; j < 32; j += 8) {
                    uint4 q;
                    __nv_bfloat162* h = (__nv_bfloat162*)&q;
#pragma unroll
                    for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(v[j + 2 * e] * p.alpha, v[j + 2 * e + 1] * p.alpha);
                    *(uint4*)(o + j) = q;
                }
            }
        }
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (warp == 0) tmdealloc(tm, 512);
}

constexpr int TP_STAGES = 6;
extern "C" __global__ void __launch_bounds__(256, 1) tc_gemm_fp8p(const __grid_constant__ TgParams p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* ring = align1k(smem_raw);
    constexpr int NT = 128, OFF_B = 16384, OFF_SFA = 32768, OFF_SFB = 33280, SB = 34816;
    const int KT = p.K / 128;
    const int MTL = (p.M + 127) / 128, T = MTL * (p.N / NT);
    __shared__ __align__(8) uint64_t full[TP_STAGES], empty[TP_STAGES], acc_full[2], acc_empty[2];
    __shared__ uint32_t tbase;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (warp == 0) tmalloc(&tbase, 512);
    if (tid == 0) {
        for (int s = 0; s < TP_STAGES; ++s) { mb_init_t(&full[s], 1); mb_init_t(&empty[s], 1); }
        for (int s = 0; s < 2; ++s) { mb_init_t(&acc_full[s], 1); mb_init_t(&acc_empty[s], 4); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    const uint32_t tm = tbase;
    if (warp == 0 && lane == 0) {
        int s = 0, ph = 0, g = 0;
        bool waited = false;
        int pend[TP_STAGES], np = 0;
        for (int t = blockIdx.x; t < T; t += gridDim.x) {
            const int mt = t % MTL, n0 = (t / MTL) * NT;
            for (int kt = 0; kt < KT; ++kt, ++g) {
                if (g >= TP_STAGES) mb_wait_t(&empty[s], ph ^ 1);
                uint8_t* st = ring + (size_t)s * SB;
                mb_expect_t(&full[s], 16384 + NT * 128 + 512 + 512);
                tma2d(st + OFF_B, p.tb, kt * 128, n0, &full[s]);
                bulk1d(st + OFF_SFB, p.sfb + (((size_t)(n0 / 128) * KT) + kt) * 512, 512, &full[s]);
                if (!waited) {
                    pend[np++] = (s << 16) | (mt << 8) | kt;
                    if (np == TP_STAGES || kt == KT - 1) {
                        pdl_wait();
                        pdl_launch();
                        waited = true;
                        for (int i = 0; i < np; ++i) {
                            const int ps = pend[i] >> 16, pm = (pend[i] >> 8) & 255, pk = pend[i] & 255;
                            uint8_t* s2 = ring + (size_t)ps * SB;
                            tma2d(s2, p.ta, pk * 128, pm * 128, &full[ps]);
                            bulk1d(s2 + OFF_SFA, p.sfa + ((size_t)pm * KT + pk) * 512, 512, &full[ps]);
                        }
                    }
                } else {
                    tma2d(st, p.ta, kt * 128, mt * 128, &full[s]);
                    bulk1d(st + OFF_SFA, p.sfa + ((size_t)mt * KT + kt) * 512, 512, &full[s]);
                }
                if (++s == TP_STAGES) { s = 0; ph ^= 1; }
            }
        }
        if (!waited) { pdl_wait(); pdl_launch(); }
    } else if (warp == 1) {
        const uint32_t id0 = uni(idesc_e4m3(NT, 0)), tmu = uni(tm);
        const uint64_t r0 = uni64(sdesc_t(ring, 16, 1024, 2));
        const uint64_t f0 = uni64(sdesc_t(ring + OFF_SFA, 512, 128, 0));
        int s = 0, ph = 0, ai = 0;
        for (int t = blockIdx.x; t < T; t += gridDim.x, ++ai) {
            const int ab = ai & 1;
            if (ai >= 2) mb_wait_t(&acc_empty[ab], ((ai >> 1) - 1) & 1);
            tcafter();
            const uint32_t d = tmu + ab * NT;
            for (int kt = 0; kt < KT; ++kt) {
                mb_wait_t(&full[s], ph);
                tcafter();
                if (elect_one()) {
                    const uint64_t so = (uint64_t)((s * SB) >> 4);
                    const uint32_t t_sfa = tmu + 256 + 4 * s, t_sfb = tmu + 288 + 4 * s;
                    tccpsf(t_sfa, f0 + so);
                    tccpsf(t_sfb, f0 + so + ((OFF_SFB - OFF_SFA) >> 4));
#pragma unroll
                    for (int kc = 0; kc < 4; ++kc)
                        tcmma(d, r0 + so + (uint64_t)(kc * 2), r0 + so + (uint64_t)((OFF_B + kc * 32) >> 4), id0 | ((uint32_t)kc << 29) | ((uint32_t)kc << 4), t_sfa, t_sfb, (kt | kc) != 0);
                    tccommit(&empty[s]);
                }
                __syncwarp();
                if (++s == TP_STAGES) { s = 0; ph ^= 1; }
            }
            if (elect_one()) tccommit(&acc_full[ab]);
            __syncwarp();
        }
    } else if (warp >= 4) {
        const int q = warp & 3;
        int ai = 0;
        for (int t = blockIdx.x; t < T; t += gridDim.x, ++ai) {
            const int ab = ai & 1, mt = t % MTL, nt0 = (t / MTL) * NT;
            mb_wait_sleep(&acc_full[ab], (ai >> 1) & 1, 100);
            tcafter();
            const uint32_t tb0 = tm + ab * NT + ((uint32_t)(32 * q) << 16);
            const int row = mt * 128 + 32 * q + lane;
            if (p.swiglu) {
                const float L = p.swiglu_limit;
                for (int c = 0; c < NT; c += 64) {
                    float g_[32], u[32];
                    tcld32(tb0 + c, g_);
                    tcld32(tb0 + c + 32, u);
                    if (row < p.M) {
                        bf16* o = (bf16*)p.out + (size_t)row * p.ldo + (nt0 + c) / 2;
#pragma unroll
                        for (int j = 0; j < 32; j += 8) {
                            uint4 qq;
                            bf16* h = (bf16*)&qq;
#pragma unroll
                            for (int e = 0; e < 8; ++e) {
                                float gte = __bfloat162float(__float2bfloat16(g_[j + e])), up = __bfloat162float(__float2bfloat16(u[j + e]));
                                if (L > 0.f) { up = fminf(fmaxf(up, -L), L); gte = fminf(gte, L); }
                                h[e] = __float2bfloat16(gte / (1.0f + expf(-gte)) * up);
                            }
                            *(uint4*)(o + j) = qq;
                        }
                    }
                }
            } else {
                for (int c = 0; c < NT; c += 32) {
                    float v[32];
                    tcld32(tb0 + c, v);
                    if (row < p.M) {
                        const int col = nt0 + c;
                        void* ob = col < p.n_split ? p.out : p.out2;
                        const int ldo = col < p.n_split ? p.ldo : p.ldo2, oc = col < p.n_split ? col : col - p.n_split;
                        if (p.out_f32) {
                            float* o = (float*)ob + (size_t)row * ldo + oc;
#pragma unroll
                            for (int j = 0; j < 32; j += 4) *(float4*)(o + j) = make_float4(v[j] * p.alpha, v[j + 1] * p.alpha, v[j + 2] * p.alpha, v[j + 3] * p.alpha);
                        } else {
                            bf16* o = (bf16*)ob + (size_t)row * ldo + oc;
#pragma unroll
                            for (int j = 0; j < 32; j += 8) {
                                uint4 qq;
                                __nv_bfloat162* h = (__nv_bfloat162*)&qq;
#pragma unroll
                                for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(v[j + 2 * e] * p.alpha, v[j + 2 * e + 1] * p.alpha);
                                *(uint4*)(o + j) = qq;
                            }
                        }
                    }
                }
            }
            tcbefore();
            __syncwarp();
            if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&acc_empty[ab])) : "memory");
        }
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (warp == 0) tmdealloc(tm, 512);
}

struct TqParams {
    const bf16* x;
    int ldx, M, K;
    const bf16* pw;
    const float* norm_part;
    int nslices;
    float eps;
    uint8_t* aq;
    uint8_t* sfa;
};
extern "C" __global__ void __launch_bounds__(256) tc_quant_rows(const __grid_constant__ TqParams p) {
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x, nkb = p.K / 32;
    __shared__ float s_rs;
    if (threadIdx.x == 0) s_rs = p.pw ? rsqrtf(norm_part_sum(p.norm_part, row, p.nslices) / p.K + p.eps) : 1.f;
    __syncthreads();
    const float rs = s_rs;
    for (int kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
        const bf16* xr = p.x + (size_t)row * p.ldx + kb * 32;
        float v[32];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 q = *(const uint4*)(xr + 8 * j);
            const bf16* h = (const bf16*)&q;
            if (p.pw) {
                const uint4 wq = *(const uint4*)(p.pw + kb * 32 + 8 * j);
                const bf16* hw = (const bf16*)&wq;
#pragma unroll
                for (int e = 0; e < 8; ++e) v[8 * j + e] = __bfloat162float(__float2bfloat16(__bfloat162float(hw[e]) * (__bfloat162float(h[e]) * rs)));
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) v[8 * j + e] = __bfloat162float(h[e]);
            }
        }
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(v[j]));
        const float sc = round_scale_pow2(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
        const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
        uint32_t w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            w[j] = (uint32_t)f32x2_to_e4m3x2(v[4 * j] * inv, v[4 * j + 1] * inv) | ((uint32_t)f32x2_to_e4m3x2(v[4 * j + 2] * inv, v[4 * j + 3] * inv) << 16);
        uint4* o = (uint4*)(p.aq + (size_t)row * p.K + kb * 32);
        o[0] = make_uint4(w[0], w[1], w[2], w[3]);
        o[1] = make_uint4(w[4], w[5], w[6], w[7]);
        const int r = row % 128;
        p.sfa[(((size_t)(row / 128) * (p.K / 128)) + kb / 4) * 512 + (r % 32) * 16 + (r / 32) * 4 + kb % 4] = (uint8_t)((__float_as_uint(sc) >> 23) & 0xff);
    }
}

extern "C" __global__ void __launch_bounds__(256) tc_row_sumsq(const bf16* x, int ldx, int M, int K, float* part) {
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= M) return;
    float ss = 0.f;
    for (int k = lane * 8; k < K; k += 256) {
        const uint4 q = *(const uint4*)(x + (size_t)row * ldx + k);
        const bf16* h = (const bf16*)&q;
#pragma unroll
        for (int e = 0; e < 8; ++e) { const float v = __bfloat162float(h[e]); ss += v * v; }
    }
    ss = warp_sum(ss);
    if (lane == 0) part[(size_t)row * 26 + 25] = ss;
}

struct TbParams {
    const void* ta;
    const void* tb;
    void* out;
    int ldo, out_f32;
    int M, N, K, nt;
    float alpha;
    int epi;
    int a_group_rows, a_group_k;
};

constexpr int TB_STAGES = 4;

extern "C" __global__ void __launch_bounds__(128, 1) tc_gemm_bf16(const __grid_constant__ TbParams p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* ring = align1k(smem_raw);
    const int NT = p.nt;
    const int OFF_B = 16384, SB = (OFF_B + NT * 128 + 1023) & ~1023;
    const int KT = p.K / 64;
    __shared__ __align__(8) uint64_t full[TB_STAGES], empty[TB_STAGES], done;
    __shared__ uint32_t tbase;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nt0 = blockIdx.x * NT, mt = blockIdx.y, m0 = mt * 128;
    const int acol = p.a_group_rows ? (nt0 / p.a_group_rows) * p.a_group_k : 0;
    if (warp == 0) tmalloc(&tbase, 256);
    if (tid == 0) {
        for (int s = 0; s < TB_STAGES; ++s) { mb_init_t(&full[s], 1); mb_init_t(&empty[s], 1); }
        mb_init_t(&done, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    const uint32_t tm = tbase;
    if (warp == 0 && lane == 0) {
        const int pre = min(KT, TB_STAGES);
        for (int kt = 0; kt < KT; ++kt) {
            const int s = kt % TB_STAGES;
            if (kt >= TB_STAGES) mb_wait_t(&empty[s], ((kt / TB_STAGES) - 1) & 1);
            uint8_t* st = ring + (size_t)s * SB;
            mb_expect_t(&full[s], 16384 + NT * 128);
            tma2d(st + OFF_B, p.tb, kt * 64, nt0, &full[s]);
            if (kt >= pre) tma2d(st, p.ta, acol + kt * 64, m0, &full[s]);
            if (kt == pre - 1) {
                pdl_wait();
                pdl_launch();
                for (int k2 = 0; k2 < pre; ++k2) tma2d(ring + (size_t)k2 * SB, p.ta, acol + k2 * 64, m0, &full[k2]);
            }
        }
    } else if (warp == 1 && lane == 0) {
        const uint32_t id = idesc_bf16(NT);
        for (int kt = 0; kt < KT; ++kt) {
            const int s = kt % TB_STAGES;
            mb_wait_t(&full[s], (kt / TB_STAGES) & 1);
            tcafter();
            uint8_t* st = ring + (size_t)s * SB;
#pragma unroll
            for (int kc = 0; kc < 4; ++kc)
                tcmma_f16(tm, sdesc_t(st + kc * 32, 16, 1024, 2), sdesc_t(st + OFF_B + kc * 32, 16, 1024, 2), id, (kt | kc) != 0);
            tccommit(&empty[s]);
        }
        tccommit(&done);
    }
    mb_wait_t(&done, 0);
    tcafter();
    const int row = m0 + 32 * warp + lane;
    for (int c = 0; c < NT; c += 32) {
        float v[32];
        tcld32(tm + ((uint32_t)(32 * warp) << 16) + c, v);
        if (row < p.M) {
            if (p.epi == 1) {
                float* o = (float*)p.out + (size_t)row * p.ldo + nt0 + c;
#pragma unroll
                for (int j = 0; j < 32; ++j) o[j] = sqrt_softplus(v[j]);
            } else if (p.out_f32) {
                float* o = (float*)p.out + (size_t)row * p.ldo + nt0 + c;
#pragma unroll
                for (int j = 0; j < 32; j += 4) *(float4*)(o + j) = make_float4(v[j] * p.alpha, v[j + 1] * p.alpha, v[j + 2] * p.alpha, v[j + 3] * p.alpha);
            } else {
                bf16* o = (bf16*)p.out + (size_t)row * p.ldo + nt0 + c;
#pragma unroll
                for (int j = 0; j < 32; j += 8) {
                    uint4 q;
                    __nv_bfloat162* h = (__nv_bfloat162*)&q;
#pragma unroll
                    for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(v[j + 2 * e] * p.alpha, v[j + 2 * e + 1] * p.alpha);
                    *(uint4*)(o + j) = q;
                }
            }
        }
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (warp == 0) tmdealloc(tm, 256);
}

extern "C" __global__ void __launch_bounds__(256) tc_norm_rows(const bf16* x, int ldx, int M, int K, const bf16* pw, const float* part, int nslices, float eps, bf16* y) {
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x;
    __shared__ float s_rs;
    if (threadIdx.x == 0) s_rs = rsqrtf(norm_part_sum(part, row, nslices) / K + eps);
    __syncthreads();
    const float rs = s_rs;
    for (int c = threadIdx.x * 8; c < K; c += blockDim.x * 8) {
        uint4 q = *(const uint4*)(x + (size_t)row * ldx + c);
        const uint4 wq = *(const uint4*)(pw + c);
        bf16* h = (bf16*)&q;
        const bf16* hw = (const bf16*)&wq;
#pragma unroll
        for (int e = 0; e < 8; ++e) h[e] = __float2bfloat16(__bfloat162float(hw[e]) * (__bfloat162float(h[e]) * rs));
        *(uint4*)(y + (size_t)row * K + c) = q;
    }
}

struct IxParams {
    const void* tk;
    const void* tq;
    const bf16* w;
    int ldw;
    float* scores;
    int ld, ratio, tmax, pos, M;
    int pass;
    const uint8_t* cand;
    int ld_cand, cand_block;
};

struct Ix4Params {
    const void* tk;
    const uint8_t* ksf;
    const void* tq;
    const uint8_t* qsf;
    const bf16* w;
    int ldw;
    float* scores;
    int ld, ratio, tmax, pos, M;
    int flags;
    const uint8_t* cand;
    int ld_cand, cand_block;
};
constexpr int IX4_D = 3, IX4_G = 64;
DEV uint32_t idesc_mxf4(int n, int sf) {
    return (1u << 7) | (1u << 10) | ((uint32_t)(n >> 3) << 17) | (1u << 23) | ((128u >> 4) << 24) | ((uint32_t)sf << 29) | ((uint32_t)sf << 4);
}
DEV void tcmma_mxf4(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; }" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb)
        : "memory");
}

constexpr int IX4B_Q = 16, IX4B_K = 4, IX4B_ST = 8192 + 512;
constexpr int IX4B_E = 16, IX4B_THREADS = (IX4B_E + 2) * 32;
DEV void tcld32_issue(uint32_t taddr, uint32_t* r) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
          "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]),
          "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
          "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr)
        : "memory");
}
DEV float head_sum(const uint32_t* v, const float* w) {
    unsigned long long a0 = 0ull, a1 = 0ull;
#pragma unroll
    for (int h4 = 0; h4 < 8; ++h4) {
        const ulonglong2 w4 = *(const ulonglong2*)(w + 4 * h4);
        const float r0 = fmaxf(__uint_as_float(v[4 * h4]), 0.f), r1 = fmaxf(__uint_as_float(v[4 * h4 + 1]), 0.f);
        const float r2 = fmaxf(__uint_as_float(v[4 * h4 + 2]), 0.f), r3 = fmaxf(__uint_as_float(v[4 * h4 + 3]), 0.f);
        const unsigned long long x0 = ((unsigned long long)__float_as_uint(r1) << 32) | __float_as_uint(r0);
        const unsigned long long x1 = ((unsigned long long)__float_as_uint(r3) << 32) | __float_as_uint(r2);
        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(a0) : "l"(x0), "l"(w4.x));
        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(a1) : "l"(x1), "l"(w4.y));
    }
    const float b0 = __uint_as_float((uint32_t)a0), b1 = __uint_as_float((uint32_t)(a0 >> 32));
    const float b2 = __uint_as_float((uint32_t)a1), b3 = __uint_as_float((uint32_t)(a1 >> 32));
    return (b0 + b1) + (b2 + b3);
}

extern "C" __global__ void __launch_bounds__(IX4B_THREADS, 1) idx_score_tc4b(const __grid_constant__ Ix4Params p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* sm = align1k(smem_raw);
    uint8_t* Qs = sm;
    uint8_t* Ks = sm + IX4B_Q * IX4B_ST;
    float* ws = (float*)(Ks + IX4B_K * IX4B_ST);
    int* lim = (int*)(ws + IX4B_Q * 4 * 32);
    __shared__ __align__(8) uint64_t q_full, q_empty, k_full[IX4B_K], k_empty[IX4B_K], d_full[IX4_D], d_empty[IX4_D];
    __shared__ uint32_t tbase;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ng = (p.M + 3) / 4, nqb = (ng + IX4B_Q - 1) / IX4B_Q, nkt = (p.tmax + 127) / 128;
    const int T = nqb * nkt;
    const int i0 = (int)((long long)blockIdx.x * T / gridDim.x), i1 = (int)((long long)(blockIdx.x + 1) * T / gridDim.x);
    pdl_wait();
    pdl_launch();
    if (i0 >= i1) return;
    if (warp == 0) tmalloc(&tbase, 512);
    if (tid == 0) {
        mb_init_t(&q_full, 1);
        mb_init_t(&q_empty, 1);
        for (int s = 0; s < IX4B_K; ++s) { mb_init_t(&k_full[s], 1); mb_init_t(&k_empty[s], 1); }
        for (int s = 0; s < IX4_D; ++s) { mb_init_t(&d_full[s], 1); mb_init_t(&d_empty[s], IX4B_E / 2); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    const uint32_t tm = tbase;
    const uint8_t* cand = p.cand;
    constexpr uint32_t T_SFB = IX4_D * 128, T_SFA = T_SFB + 4 * IX4B_Q;
    if (warp == IX4B_E) {
        if (lane == 0) {
            int s = 0, ph = 0, k = 0, nq = 0, prev = -1;
            for (int it = i0; it < i1; ++it) {
                const int qb = it / nkt, kt = it % nkt;
                if (qb != prev) {
                    if (nq > 0) mb_wait_t(&q_empty, (nq - 1) & 1);
                    const int gq0 = qb * IX4B_Q, gq1 = min(ng, gq0 + IX4B_Q);
                    mb_expect_t(&q_full, (gq1 - gq0) * IX4B_ST);
                    for (int g = gq0; g < gq1; ++g) {
                        tma3d(Qs + (g - gq0) * IX4B_ST, p.tq, 0, g * 128, 0, &q_full);
                        bulk1d(Qs + (g - gq0) * IX4B_ST + 8192, p.qsf + (size_t)g * 512, 512, &q_full);
                    }
                    prev = qb;
                    ++nq;
                }
                if (k >= IX4B_K) mb_wait_t(&k_empty[s], ph ^ 1);
                uint8_t* st = Ks + s * IX4B_ST;
                mb_expect_t(&k_full[s], IX4B_ST);
                tma3d(st, p.tk, 0, kt * 128, 0, &k_full[s]);
                bulk1d(st + 8192, p.ksf + (size_t)kt * 512, 512, &k_full[s]);
                ++k;
                if (++s == IX4B_K) { s = 0; ph ^= 1; }
            }
        }
    } else if (warp == IX4B_E + 1) {
        const uint32_t tmu = uni(tm);
        const uint64_t q0d = uni64(sdesc_t(Qs, 2048, 128, 0)), k0d = uni64(sdesc_t(Ks, 2048, 128, 0));
        const uint64_t qf = uni64(sdesc_t(Qs + 8192, 512, 128, 0)), kf = uni64(sdesc_t(Ks + 8192, 512, 128, 0));
        int s = 0, ph = 0, db = 0, dph = 0, nd = 0, nq = 0, prev = -1;
        for (int it = i0; it < i1; ++it) {
            const int qb = it / nkt;
            const int gq0 = qb * IX4B_Q, ngq = min(ng, gq0 + IX4B_Q) - gq0;
            if (qb != prev) {
                if (nq > 0 && elect_one()) tccommit(&q_empty);
                __syncwarp();
                mb_wait_t(&q_full, nq & 1);
                tcafter();
                if (elect_one())
                    for (int g = 0; g < ngq; ++g) tccpsf(tmu + T_SFB + 4 * g, qf + (uint64_t)((g * IX4B_ST) >> 4));
                __syncwarp();
                prev = qb;
                ++nq;
            }
            mb_wait_t(&k_full[s], ph);
            tcafter();
            if (elect_one()) tccpsf(tmu + T_SFA + 4 * s, kf + (uint64_t)((s * IX4B_ST) >> 4));
            __syncwarp();
            for (int g = 0; g < ngq; ++g, ++nd) {
                if (nd >= IX4_D) mb_wait_t(&d_empty[db], dph ^ 1);
                tcafter();
                if (elect_one()) {
                    const uint32_t d = tmu + db * 128;
#pragma unroll
                    for (int j = 0; j < 2; ++j)
                        if (!(p.flags & 2))
                            tcmma_mxf4(d, k0d + (uint64_t)((s * IX4B_ST + j * 4096) >> 4), q0d + (uint64_t)((g * IX4B_ST + j * 4096) >> 4), idesc_mxf4(128, 2 * j), tmu + T_SFA + 4 * s, tmu + T_SFB + 4 * g, j);
                    tccommit(&d_full[db]);
                }
                __syncwarp();
                if (++db == IX4_D) { db = 0; dph ^= 1; }
            }
            if (elect_one()) tccommit(&k_empty[s]);
            __syncwarp();
            if (++s == IX4B_K) { s = 0; ph ^= 1; }
        }
    } else if (warp < IX4B_E) {
        const int q4 = warp & 3, hq = (warp >> 2) & 1, set = warp >> 3;
        int db = 0, dph = 0, prev = -1, gi = 0;
        for (int it = i0; it < i1; ++it) {
            const int qb = it / nkt, kt = it % nkt;
            const int gq0 = qb * IX4B_Q, ngq = min(ng, gq0 + IX4B_Q) - gq0;
            if (qb != prev) {
                asm volatile("bar.sync 1, %0;" ::"r"(IX4B_E * 32) : "memory");
                for (int i = tid; i < ngq * 4 * 32; i += IX4B_E * 32) {
                    const int m = gq0 * 4 + i / 32;
                    ws[i] = m < p.M ? __bfloat162float(p.w[(size_t)m * p.ldw + i % 32]) : 0.f;
                }
                for (int i = tid; i < ngq * 4; i += IX4B_E * 32) lim[i] = (p.pos + gq0 * 4 + i + 1) / p.ratio;
                asm volatile("bar.sync 1, %0;" ::"r"(IX4B_E * 32) : "memory");
                prev = qb;
            }
            const int t = kt * 128 + 32 * q4 + lane;
            for (int g = 0; g < ngq; ++g, ++gi) {
                const int b = db;
                const uint32_t ph = dph;
                if (++db == IX4_D) { db = 0; dph ^= 1; }
                if ((gi & 1) != set) continue;
                mb_wait_t(&d_full[b], ph);
                tcafter();
                uint32_t v0[32], v1[32];
                const uint32_t ta = tm + b * 128 + 2 * hq * 32 + ((uint32_t)(32 * q4) << 16);
                tcld32_issue(ta, v0);
                tcld32_issue(ta + 32, v1);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                tcbefore();
                __syncwarp();
                if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&d_empty[b])) : "memory");
                if (p.flags & 1) continue;
                const int ql = g * 4 + 2 * hq;
#pragma unroll
                for (int qq = 0; qq < 2; ++qq) {
                    const int m = gq0 * 4 + ql + qq;
                    if (m >= p.M) break;
                    bool ok = t < p.tmax && t < lim[ql + qq];
                    if (ok && cand) ok = cand[(size_t)m * p.ld_cand + t / p.cand_block] != 0;
                    const float sc = __any_sync(0xffffffffu, ok) ? head_sum(qq ? v1 : v0, ws + (ql + qq) * 32) : 0.f;
                    if (t < p.tmax && !(p.flags & 4)) p.scores[(size_t)m * p.ld + t] = ok ? sc : -INFINITY;
                }
            }
        }
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (warp == 0) tmdealloc(tm, 512);
}

struct MetaT {
    int pos;
    int m;
    int tokens[16];
};
#ifndef IDX_SPLIT
#define IDX_SPLIT 0
#endif
extern "C" __global__ void __launch_bounds__(320, 1) idx_score_tc_dec(const MetaT* meta, const __grid_constant__ IxParams p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* sm = align1k(smem_raw);
    uint8_t* Bs = sm;
    uint8_t* As = sm + 65536;
    float* ws = (float*)(sm + 65536 + 3 * 32768);
    __shared__ __align__(8) uint64_t b_full, a_full[3], a_empty[3], d_full[2], d_empty[2];
    __shared__ uint32_t tbase;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    pdl_wait();
    pdl_launch();
    const int pos = meta->pos, M = meta->m;
    const int reach = (pos + M) / p.ratio;
#if IDX_SPLIT
    const int n_old = pos / p.ratio;
    const int ntiles = p.pass == 1 ? (n_old + 127) / 128 : (reach + 127) / 128;
    const int tb = p.pass == 2 ? n_old / 128 : 0;
#else
    const int ntiles = (reach + 127) / 128;
    const int tb = 0;
#endif
    const int t1 = tb + (int)blockIdx.x;
    if (t1 >= ntiles) return;
    if (warp == 0) tmalloc(&tbase, 512);
    for (int i = tid; i < 8 * 32; i += blockDim.x) {
        const int m = i / 32;
        ws[i] = m < M ? __bfloat162float(p.w[(size_t)m * p.ldw + i % 32]) : 0.f;
    }
    if (tid == 0) {
        mb_init_t(&b_full, 1);
        for (int s = 0; s < 3; ++s) { mb_init_t(&a_full[s], 1); mb_init_t(&a_empty[s], 1); }
        for (int s = 0; s < 2; ++s) { mb_init_t(&d_full[s], 1); mb_init_t(&d_empty[s], 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    const uint32_t tm = tbase;
    if (warp == 8) {
        if (lane == 0) {
            mb_expect_t(&b_full, 65536);
            tma2d(Bs, p.tq, 0, 0, &b_full);
            tma2d(Bs + 32768, p.tq, 64, 0, &b_full);
            int i = 0;
            for (int t = t1; t < ntiles; t += gridDim.x, ++i) {
                const int s = i % 3;
                if (i >= 3) mb_wait_t(&a_empty[s], ((i / 3) - 1) & 1);
                mb_expect_t(&a_full[s], 32768);
                tma2d(As + s * 32768, p.tk, 0, t * 128, &a_full[s]);
                tma2d(As + s * 32768 + 16384, p.tk, 64, t * 128, &a_full[s]);
            }
        }
    } else if (warp == 9) {
        if (lane == 0) {
            const uint32_t id = idesc_bf16(32 * M);
            mb_wait_t(&b_full, 0);
            int i = 0;
            for (int t = t1; t < ntiles; t += gridDim.x, ++i) {
                const int s = i % 3, b = i & 1;
                if (i >= 2) mb_wait_t(&d_empty[b], ((i >> 1) - 1) & 1);
                mb_wait_t(&a_full[s], (i / 3) & 1);
                tcafter();
#pragma unroll
                for (int kc = 0; kc < 8; ++kc) {
                    const int kb = kc >> 2, ko = (kc & 3) * 32;
                    tcmma_f16(tm + b * 256, sdesc_t(As + s * 32768 + kb * 16384 + ko, 16, 1024, 2), sdesc_t(Bs + kb * 32768 + ko, 16, 1024, 2), id, kc != 0);
                }
                tccommit(&a_empty[s]);
                tccommit(&d_full[b]);
            }
        }
    } else {
        const int qd = warp & 3, g0 = (warp >> 2) * 4;
        const int nq = min(4, M - g0);
        const uint8_t* cb = p.cand ? p.cand + (size_t)g0 * p.ld_cand : nullptr;
        uint8_t cn[4] = {1, 1, 1, 1};
        auto cload = [&](int t) {
#pragma unroll
            for (int j = 0; j < 4; ++j) cn[j] = (cb && j < nq && t < reach) ? cb[(size_t)j * p.ld_cand + t / p.cand_block] : (uint8_t)1;
        };
        cload(t1 * 128 + 32 * qd + lane);
        int i = 0;
        for (int t0 = t1; t0 < ntiles; t0 += gridDim.x, ++i) {
            const int b = i & 1, t = t0 * 128 + 32 * qd + lane;
            uint8_t cc[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) cc[j] = cn[j];
            if (t0 + (int)gridDim.x < ntiles) cload(t + (int)gridDim.x * 128);
            bool ok[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) ok[j] = j < nq && t < reach && t < (pos + g0 + j + 1) / p.ratio;
            mb_wait_t(&d_full[b], (i >> 1) & 1);
            tcafter();
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                if (j >= nq) break;
                uint32_t v0[32], v1[32];
                const uint32_t ta = tm + b * 256 + (g0 + j) * 32 + ((uint32_t)(32 * qd) << 16);
                tcld32_issue(ta, v0);
                if (j + 1 < nq) tcld32_issue(ta + 32, v1);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (j + 2 >= nq) {
                    tcbefore();
                    __syncwarp();
                    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&d_empty[b])) : "memory");
                }
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    if (j + jj >= nq) break;
                    const uint32_t* v = jj ? v1 : v0;
                    const int q = g0 + j + jj;
                    const float4* wr = (const float4*)(ws + q * 32);
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                    for (int h4 = 0; h4 < 8; ++h4) {
                        const float4 w4 = wr[h4];
                        a0 += fmaxf(__uint_as_float(v[4 * h4]), 0.f) * w4.x;
                        a1 += fmaxf(__uint_as_float(v[4 * h4 + 1]), 0.f) * w4.y;
                        a2 += fmaxf(__uint_as_float(v[4 * h4 + 2]), 0.f) * w4.z;
                        a3 += fmaxf(__uint_as_float(v[4 * h4 + 3]), 0.f) * w4.w;
                    }
#if IDX_SPLIT
                    const bool wt = t < reach && (p.pass == 1 ? t < n_old : t >= n_old);
#else
                    const bool wt = t < reach;
#endif
                    if (wt) p.scores[(size_t)q * p.ld + t] = ok[j + jj] && cc[j + jj] ? (a0 + a1) + (a2 + a3) : -INFINITY;
                }
            }
            if (nq <= 0) {
                tcbefore();
                __syncwarp();
                if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&d_empty[b])) : "memory");
            }
        }
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (warp == 0) tmdealloc(tm, 512);
#if IDX_SPLIT
#else
#endif
}
