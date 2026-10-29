#include "tc.cuh"
#include "rope.cuh"
#include "quantblocks.cuh"

struct Tg2Params {
    const void* tmap;
    const uint8_t* sfa;
    const bf16* x;
    const bf16* gamma;
    const uint8_t* bimg;
    const float* rs_part;
    void* out;
    void* out2;
    const float* rope_cos;
    const float* rope_sin;
    const int* meta;
    uint8_t* qimg;
    const bf16* qgamma;
    float* qssq;
    RopeParams kvr;
    float alpha, eps;
    int M, N, K, kpt;
    int ldx, ldo, ldo2, n_split, out_f32;
    int bsrc;
    int rs_n, rs_stride, rs_rowstride, rs_off;
    int rope_head, rope_off, iq_head, iq_off;
    int partial;
    int group_rows;
    int qrows;
    int kvr_on;
    uint16_t* xq;
    float* xsq;
    int ssp;
    int rs_pre;
    int nacc;
    int bload;
    int flags;
};
static_assert(sizeof(Tg2Params) == 352, "Tg2Params layout (ops.rs mirrors it)");

constexpr int TG2_KT = 10;

enum { V_GEN = 0, V_QKV = 1, V_QB = 2, V_PART = 3, V_OA = 4, V_GATE = 5 };
template <int KIND, int VAR>
struct TgF {
    static constexpr bool gen = VAR == V_GEN;
    static constexpr bool xq = KIND == 0 && (gen || VAR == V_QKV);
    static constexpr bool img = KIND == 0 && (gen || VAR == V_QB || VAR == V_PART);
    static constexpr bool gamma = gen || VAR == V_QKV || VAR == V_GATE;
    static constexpr bool rsdef = gen || VAR == V_QKV || VAR == V_QB;
    static constexpr bool rspre = KIND == 1 && (gen || VAR == V_GATE);
    static constexpr bool rope = gen || VAR == V_QB;
    static constexpr bool kvr = KIND == 0 && (gen || VAR == V_QB);
    static constexpr bool split = gen || VAR == V_QKV || VAR == V_QB;
    static constexpr bool iq = gen || VAR == V_QB;
    static constexpr bool qimg = gen || VAR == V_QKV || VAR == V_OA;
    static constexpr bool qgamma = gen || VAR == V_QKV;
    static constexpr bool ssp = gen || VAR == V_GATE;
    static constexpr bool f32o = gen || VAR == V_GATE;
    static constexpr bool xqout = KIND == 1 && (gen || VAR == V_GATE);
    static constexpr bool part = gen || VAR == V_PART;
    static constexpr bool group = KIND == 1 && (gen || VAR == V_OA);
};

DEV uint32_t tg2_mapa(const void* p, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(s32a(p)), "r"(rank));
    return r;
}
DEV void tg2_st_async4(uint32_t raddr, float4 v, uint32_t rbar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(raddr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar)
                 : "memory");
}
DEV float tg2_bfr(float v) { return __bfloat162float(__float2bfloat16(v)); }
DEV float tg2_sel4(const float (&q)[4], int b) { return b == 0 ? q[0] : b == 1 ? q[1] : b == 2 ? q[2] : q[3]; }

template <class F>
DEV void tg2_unit(const Tg2Params& p, int m, int r, float v, float qg, const float* rcos, const float* rsin) {
    constexpr uint32_t FULL = 0xffffffffu;
    if (F::split && p.n_split && r >= p.n_split) {
        const int rr = r - p.n_split;
        if (F::iq && p.iq_head) {
            const int d = F::gen ? rr % p.iq_head : rr & (p.iq_head - 1);
            float x = tg2_bfr(v);
            const float pt = __shfl_xor_sync(FULL, x, 1);
            if (d >= p.iq_off) {
                const int f = (d - p.iq_off) >> 1;
                const float c = rcos[m * 32 + f], sn = rsin[m * 32 + f];
                x = (r & 1) ? __fmaf_rn(pt, sn, __fmul_rn(x, c)) : __fmaf_rn(x, c, -__fmul_rn(pt, sn));
                x = tg2_bfr(x);
            }
            float amax = fabsf(x);
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL, amax, o));
            amax = fmaxf(amax, 6.f * 1.1754943508222875e-38f);
            const float sc = round_scale_pow2(amax, 1.0f / 6.0f);
            x = fp4_round(fminf(fmaxf(__fdiv_rn(x, sc), -6.f), 6.f)) * sc;
            ((bf16*)p.out2)[(size_t)m * p.ldo2 + rr] = __float2bfloat16(x);
        } else if (F::f32o && p.out_f32) {
            ((float*)p.out2)[(size_t)m * p.ldo2 + rr] = v;
        } else {
            ((bf16*)p.out2)[(size_t)m * p.ldo2 + rr] = __float2bfloat16(v);
        }
        return;
    }
    const int rh = F::rope ? (F::gen ? r % max(p.rope_head, 1) : r & (p.rope_head - 1)) : 0;
    if (F::rope && p.rope_cos && rh >= p.rope_off) {
        const float a = tg2_bfr(v);
        const float o = __shfl_xor_sync(FULL, a, 1);
        const int f = (rh - p.rope_off) >> 1;
        const float c = rcos[m * 32 + f], sn = rsin[m * 32 + f];
        const float y = (r & 1) ? __fmaf_rn(o, sn, __fmul_rn(a, c)) : __fmaf_rn(a, c, -__fmul_rn(o, sn));
        ((bf16*)p.out)[(size_t)m * p.ldo + r] = __float2bfloat16(y);
        return;
    }
    if (F::gen ? p.out != nullptr : true) {
        if (F::ssp && p.ssp) ((float*)p.out)[(size_t)m * p.ldo + r] = sqrt_softplus(v);
        else if (F::f32o && p.out_f32) ((float*)p.out)[(size_t)m * p.ldo + r] = v;
        else ((bf16*)p.out)[(size_t)m * p.ldo + r] = __float2bfloat16(v);
    }
    if (F::qimg && (F::gen ? p.qimg != nullptr : true) && r < p.qrows) {
        const float x = tg2_bfr(v);
        const float u = (F::qgamma && p.qgamma) ? qg * x : x;
        float amax = fabsf(u);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL, amax, o));
        const float sc = round_scale_pow2(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
        const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
        const float q = u * inv;
        const float qn = __shfl_down_sync(FULL, q, 1);
        const uint32_t pk = f32x2_to_e4m3x2(q, qn);
        const uint32_t hi = __shfl_down_sync(FULL, pk, 2);
        const int t = r >> 7, kk = r & 127;
        if ((r & 3) == 0) *(uint32_t*)(p.qimg + (size_t)t * 1024 + (kk >> 4) * 128 + m * 16 + (kk & 15)) = (pk & 0xffffu) | (hi << 16);
        if ((r & 31) == 0) p.qimg[(size_t)(p.qrows / 128) * 1024 + (size_t)t * 512 + m * 16 + (kk >> 5)] = (uint8_t)(__float_as_uint(sc) >> 23);
        if (F::qgamma && p.qssq) {
            float s = x * x;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
            if ((r & 31) == 0) p.qssq[m * (p.qrows / 32) + (r >> 5)] = s;
        }
    }
}

template <int KIND, int VAR>
DEV void tg2_body(const Tg2Params& p) {
    using F = TgF<KIND, VAR>;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (F::kvr && p.kvr_on && (int)blockIdx.x >= (int)gridDim.x - p.kvr_on) {
        const int grp = tid >> 6;
        const int nb = p.kvr_on * (int)gridDim.y;
        const int b = ((int)blockIdx.x - ((int)gridDim.x - p.kvr_on)) * (int)gridDim.y + (int)blockIdx.y;
        for (int row = b * 2 + grp; row < 16 * p.kvr.rows_per_token; row += 2 * nb) rope_body(p.meta, p.kvr, row, tid & 63, grp);
        return;
    }
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* sm = align1k(smem_raw);
    uint32_t crank, cs;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cs));
    const int rt = blockIdx.x;
    const int M = p.M;
    constexpr int TKB = KIND == 0 ? 128 : 64;
    const int KS = p.K / (int)cs, KT = KS / TKB, k0 = (int)crank * KS;
    const bool xq = F::xq && (F::gen ? p.bsrc == 0 : true);
    const bool img = F::img && (F::gen ? p.bsrc == 1 : true);
    const bool has_gamma = F::gamma && p.gamma;
    const bool late = p.flags & 1;
    uint8_t* sA = sm;
    uint8_t* sB = sA + KT * 16384;
    uint8_t* sSFA = sB + (KIND == 0 ? KT * 1024 : KS * 16);
    uint8_t* sSFB = sSFA + (KIND == 0 ? KT * 512 : 0);
    bf16* gs = (bf16*)(sSFB + (KIND == 0 ? KT * 512 : 0));
    const bool xraw_on = F::xqout && p.rs_pre == 2 && p.xq;
    bf16* xraw = gs + (has_gamma ? KS : 0);
    float* recv = (float*)(xraw + (xraw_on ? 8 * KS : 0));
    __shared__ __align__(8) uint64_t wbar[TG2_KT], xbar, gbar, mbar, rbar;
    __shared__ uint32_t tbase;
    __shared__ float rs[8];
    __shared__ float rcos[8 * 32], rsin[8 * 32];
    __shared__ __align__(16) float red[8 * 128];
    const bool partial = F::part && (F::gen ? p.partial != 0 : true);
    const bool push = cs > 1 && !partial;
    const int nunits = 4 * M;
    const int nown = (int)crank < nunits ? (nunits - (int)crank + (int)cs - 1) / (int)cs : 0;
    constexpr uint32_t TMCOLS = KIND == 0 ? 128 : 32;
    if (warp == 0) tmalloc(&tbase, TMCOLS);
    if (tid == 0) {
        for (int t = 0; t < TG2_KT; ++t) mb_init_t(&wbar[t], 1);
        mb_init_t(&xbar, 1);
        mb_init_t(&gbar, 1);
        mb_init_t(&mbar, 1);
        mb_init_t(&rbar, 1);
        if (push) mb_expect_t(&rbar, (uint32_t)(nown * (int)cs * 128));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (push) asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    const uint32_t tm = tbase;
    if (tid == 0) {
        if (KIND == 0) {
            for (int t = 0; t < KT; ++t) {
                mb_expect_t(&wbar[t], 16384 + (t == 0 ? KT * 512 : 0));
                tma2d_ef(sA + t * 16384, p.tmap, k0 + t * 128, rt * 128, &wbar[t]);
                if (t == 0) bulk1d(sSFA, p.sfa + ((size_t)rt * p.kpt + k0 / 128) * 512, KT * 512, &wbar[0]);
            }
        } else {
            for (int t = 0; t < KT; ++t) {
                mb_expect_t(&wbar[t], 16384);
                tma2d_ef(sA + t * 16384, p.tmap, k0 + t * 64, rt * 128, &wbar[t]);
            }
        }
        if (has_gamma) {
            mb_expect_t(&gbar, KS * 2);
            bulk1d(gs, p.gamma + k0, KS * 2, &gbar);
        }
    }
    float qg[4] = {0.f, 0.f, 0.f, 0.f};
    if (F::qgamma && p.qimg && p.qgamma) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int r = rt * 128 + 32 * b + lane;
            if (r < p.qrows) qg[b] = __bfloat162float(p.qgamma[r]);
        }
    }
    if (xq)
        for (int i = tid; i < KT * 128; i += 128) if ((i & 3) != 0 || (i & 127) >= 4 * M) ((uint32_t*)sSFB)[i] = 0x7f7f7f7fu;
    if (KIND == 0 && warp == 1) {
        mb_wait_t(&wbar[0], 0);
        tcafter();
        if (elect_one()) {
            const uint64_t sfa0 = sdesc_t(sSFA, 512, 128, 0);
            for (int t = 0; t < KT; ++t) tccpsf(tm + 32 + 4 * t, sfa0 + (uint64_t)((t * 512) >> 4));
        }
        __syncwarp();
    }
    pdl_wait();
    pdl_launch();
    const bool bload = img && p.bload;
    if (tid == 0 && img && !bload) {
        mb_expect_t(&xbar, KT * 1024 + KT * 512);
        bulk1d(sB, p.bimg + (size_t)(k0 / 128) * 1024, KT * 1024, &xbar);
        bulk1d(sSFB, p.bimg + (size_t)(p.K / 128) * 1024 + (size_t)(k0 / 128) * 512, KT * 512, &xbar);
    }
    const int nkb = KS / 32, xtotal = xq ? M * nkb : 0;
    const float rnkb = 1.0f / (float)nkb;
    uint4 xr[F::xq ? 8 : 1];
    if (F::xq) {
#pragma unroll
        for (int i = 0; i < (F::xq ? 8 : 1); ++i) {
            const int b = warp * 8 + (lane >> 2) + 32 * i;
            xr[i] = make_uint4(0, 0, 0, 0);
            if (b < xtotal) {
                const int m = F::gen ? b / nkb : __float2int_rd(((float)b + 0.5f) * rnkb);
                xr[i] = *(const uint4*)(p.x + (size_t)m * p.ldx + k0 + (b - m * nkb) * 32 + (lane & 3) * 8);
            }
        }
    }
    uint4 bv[F::img ? 4 : 1];
    const int bimg_n = bload ? KT * 96 : 0;
    if (F::img) {
#pragma unroll
        for (int j = 0; j < (F::img ? 4 : 1); ++j) {
            const int i = tid + 128 * j;
            bv[j] = make_uint4(0, 0, 0, 0);
            if (i < bimg_n) {
                const uint4* src = i < KT * 64 ? (const uint4*)(p.bimg + (size_t)(k0 / 128) * 1024) + i
                                               : (const uint4*)(p.bimg + (size_t)(p.K / 128) * 1024 + (size_t)(k0 / 128) * 512) + (i - KT * 64);
                bv[j] = __ldcg(src);
            }
        }
    }
    if (F::rope && (p.rope_cos != nullptr) && warp == 2) {
        const int pos0 = p.meta[0];
        for (int m = 0; m < M; ++m) {
            rcos[m * 32 + lane] = p.rope_cos[(size_t)(pos0 + m) * 32 + lane];
            rsin[m * 32 + lane] = p.rope_sin[(size_t)(pos0 + m) * 32 + lane];
        }
    }
    const size_t xoff = F::group && p.group_rows ? (size_t)((rt * 128) / p.group_rows) * p.K : 0;
    uint4 xv[KIND == 1 ? 5 : 1];
    const int nch = KIND == 1 ? (KS / 8) * 8 : 0;
    if (KIND == 1) {
#pragma unroll
        for (int j = 0; j < (KIND == 1 ? 5 : 1); ++j) {
            const int i = tid + j * 128, m = i & 7;
            xv[j] = make_uint4(0, 0, 0, 0);
            if (i < nch && m < M) xv[j] = *(const uint4*)(p.x + (size_t)m * p.ldx + xoff + k0 + (i >> 3) * 8);
        }
    }
    const bool rspre = F::rspre && p.rs_part && p.rs_pre == 1;
    const bool rspost = F::rspre && p.rs_part && p.rs_pre == 2;
    const bool rsdef = F::rsdef && p.rs_part && !(F::rspre && p.rs_pre);
    const bool rsown = rspre;
    float rv[F::rspre ? 32 : 1];
    if ((rsown || rspost) && warp == 3 && lane < 8 && lane < M) {
        const float* b = p.rs_part + (size_t)lane * p.rs_rowstride + p.rs_off;
#pragma unroll
        for (int i = 0; i < (F::rspre ? 32 : 1); ++i) rv[i] = i < p.rs_n ? __ldcg(b + (size_t)i * p.rs_stride) : 0.f;
    }
    auto row_stat = [&]() {
        if (warp == 3 && lane < 8) {
            float ss = 0.f;
            if (lane < M) {
#pragma unroll
                for (int i = 0; i < (F::rspre ? 32 : 1); ++i)
                    if (i < p.rs_n) ss += rv[i];
            }
            rs[lane] = rsqrtf(ss / p.K + p.eps);
        }
    };
    if (rsown) row_stat();
    float rsa[8], rsb[8];
    if (rsdef && warp == 3) {
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const float* b = p.rs_part + (size_t)m * p.rs_rowstride + p.rs_off;
            rsa[m] = m < M && lane < p.rs_n ? __ldcg(b + (size_t)lane * p.rs_stride) : 0.f;
            rsb[m] = m < M && lane + 32 < p.rs_n ? __ldcg(b + (size_t)(lane + 32) * p.rs_stride) : 0.f;
        }
    }
    if (KIND == 1) {
        if (has_gamma) {
            if (rspre) __syncthreads();
            mb_wait_t(&gbar, 0);
        }
#pragma unroll
        for (int j = 0; j < (KIND == 1 ? 5 : 1); ++j) {
            const int i = tid + j * 128, c = i >> 3, m = i & 7;
            if (i >= nch) break;
            uint4 v = xv[j];
            if (xraw_on && rt == 0) *(uint4*)(xraw + m * KS + c * 8) = v;
            if (F::gamma && has_gamma && m < M) {
                const float r = rspre ? rs[m] : 1.f;
                bf16* h = (bf16*)&v;
                const uint4 g = *(const uint4*)(gs + c * 8);
                const bf16* gh = (const bf16*)&g;
#pragma unroll
                for (int e = 0; e < 8; ++e) h[e] = __float2bfloat16(__bfloat162float(gh[e]) * (__bfloat162float(h[e]) * r));
            }
            *(uint4*)(sB + c * 128 + m * 16) = v;
        }
        tcfence_proxy();
        __syncthreads();
    } else if (xq) {
        if (has_gamma) mb_wait_t(&gbar, 0);
        const int c = lane & 3;
#pragma unroll
        for (int i = 0; i < (F::xq ? 8 : 1); ++i) {
            const int b0 = warp * 8 + 32 * i;
            if (b0 >= xtotal) break;
            const int b = b0 + (lane >> 2);
            const bool ok = b < xtotal;
            const int bb = ok ? b : 0;
            const int m = F::gen ? bb / nkb : __float2int_rd(((float)bb + 0.5f) * rnkb), kl = (bb - m * nkb) * 32 + c * 8;
            const uint4 q = xr[i];
            const bf16* h = (const bf16*)&q;
            float v[8];
            if (has_gamma) {
                const uint4 g = *(const uint4*)(gs + kl);
                const bf16* gh = (const bf16*)&g;
#pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = __bfloat162float(gh[e]) * __bfloat162float(h[e]);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = __bfloat162float(h[e]);
            }
            float a = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) a = fmaxf(a, fabsf(v[e]));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
            const float sc = round_scale_pow2(fmaxf(a, 1e-4f), 1.0f / 448.0f);
            const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
            uint32_t w[2];
#pragma unroll
            for (int j = 0; j < 2; ++j)
                w[j] = (uint32_t)f32x2_to_e4m3x2(v[4 * j] * inv, v[4 * j + 1] * inv) | ((uint32_t)f32x2_to_e4m3x2(v[4 * j + 2] * inv, v[4 * j + 3] * inv) << 16);
            if (ok) {
                const int t = kl >> 7, kk = kl & 127;
                *(uint2*)(sB + t * 1024 + (kk >> 4) * 128 + m * 16 + (kk & 15)) = make_uint2(w[0], w[1]);
                if (c == 0) sSFB[t * 512 + m * 16 + (kk >> 5)] = (uint8_t)(__float_as_uint(sc) >> 23);
            }
        }
        tcfence_proxy();
        __syncthreads();
    }
    if (F::img && bimg_n) {
#pragma unroll
        for (int j = 0; j < (F::img ? 4 : 1); ++j) {
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
        if (KIND == 0) {
            if (img) {
                if (!bload) mb_wait_t(&xbar, 0);
            }
            tcafter();
            const uint64_t a0 = sdesc_t(sA, 16, 1024, 2), b0 = sdesc_t(sB, 128, 1024, 0);
            if (elect_one()) {
                const uint64_t sfb0 = sdesc_t(sSFB, 512, 128, 0);
                for (int t = 0; t < KT; ++t) tccpsf(tm + 64 + 4 * t, sfb0 + (uint64_t)((t * 512) >> 4));
            }
            __syncwarp();
            for (int t = 0; t < KT; ++t) {
                mb_wait_t(&wbar[t], 0);
                tcafter();
                if (elect_one()) {
#pragma unroll
                    for (int kc = 0; kc < 4; ++kc)
                        tcmma(tm, a0 + (uint64_t)((t * 16384 + kc * 32) >> 4), b0 + (uint64_t)((t * 1024 + kc * 256) >> 4), idesc_e4m3(8, kc), tm + 32 + 4 * t,
                              tm + 64 + 4 * t, (t | kc) != 0);
                }
                __syncwarp();
            }
        } else {
            const uint64_t a0 = sdesc_t(sA, 16, 1024, 2), b0 = sdesc_t(sB, 128, 1024, 0);
            const uint32_t id = idesc_bf16(8);
            for (int t = 0; t < KT; ++t) {
                mb_wait_t(&wbar[t], 0);
                tcafter();
                if (elect_one()) {
#pragma unroll
                    for (int kc = 0; kc < 4; ++kc)
                        tcmma_f16(tm, a0 + (uint64_t)((t * 16384 + kc * 32) >> 4), b0 + (uint64_t)((t * 1024 + kc * 256) >> 4), id, (t | kc) != 0);
                }
                __syncwarp();
            }
        }
        if (elect_one()) tccommit(&mbar);
        __syncwarp();
    }
    if (rspost) row_stat();
    if (F::xqout && p.xq && rt == 0 && warp != 1) {
        if (rspost) asm volatile("bar.sync 2, 96;" ::: "memory");
        const int qw = warp == 0 ? 0 : warp - 1;
        quant_blocks8(qw, 3, lane, 8, M, KS / 32, p.xq + k0, p.K + 32, p.xsq + (k0 / 32) * 8, [&](int m, int kb, int c, float* v) {
            if (rspost) {
                const uint4 q = *(const uint4*)(xraw + m * KS + kb * 32 + c * 8);
                const uint4 g = *(const uint4*)(gs + kb * 32 + c * 8);
                const bf16* h = (const bf16*)&q;
                const bf16* gh = (const bf16*)&g;
#pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = __bfloat162float(__float2bfloat16(__bfloat162float(gh[e]) * (__bfloat162float(h[e]) * rs[m])));
            } else {
                const uint4 q = *(const uint4*)(sB + (kb * 4 + c) * 128 + m * 16);
                const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    v[2 * e] = __uint_as_float(w[e] << 16);
                    v[2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u);
                }
            }
        });
    }
    if (rsdef && warp == 3) {
        float a[8];
#pragma unroll
        for (int m = 0; m < 8; ++m) a[m] = rsa[m] + rsb[m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int m = 0; m < 8; ++m) a[m] += __shfl_xor_sync(0xffffffffu, a[m], o);
        float sm = a[0];
#pragma unroll
        for (int m = 1; m < 8; ++m)
            if (lane == m) sm = a[m];
        if (lane < 8) rs[lane] = rsqrtf(sm / p.K + p.eps);
    }
    mb_wait_t(&mbar, 0);
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
    __syncthreads();
    tcafter();
    if (!late && warp == 0) tmdealloc(tm, TMCOLS);
    const bool use_rs = rsdef || rspost;
    if (partial) {
        const int row = rt * 128 + 32 * warp + lane;
#pragma unroll
        for (int m = 0; m < 8; ++m)
            if (m < M) ((float*)p.out)[((size_t)m * cs + crank) * p.ldo + row] = acc[m] * p.alpha;
    } else if (F::gen && !push) {
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m >= M) break;
            const float sc = use_rs ? p.alpha * rs[m] : p.alpha;
            tg2_unit<F>(p, m, rt * 128 + 32 * warp + lane, acc[m] * sc, tg2_sel4(qg, warp), rcos, rsin);
        }
    } else {
#pragma unroll
        for (int m = 0; m < 8; ++m) red[m * 128 + 32 * warp + lane] = acc[m];
        __syncwarp();
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int m = h * 4 + (lane >> 3), ch = lane & 7;
            if (m < M) {
                const int u = m * 4 + warp, o = u % (int)cs, slot = u / (int)cs;
                const float4 v = *(const float4*)(red + m * 128 + 32 * warp + ch * 4);
                tg2_st_async4(tg2_mapa(recv + (slot * (int)cs + (int)crank) * 32 + ch * 4, (uint32_t)o), v, tg2_mapa(&rbar, (uint32_t)o));
            }
        }
        mb_wait_t(&rbar, 0);
        for (int s = warp; s < nown; s += 4) {
            const int u = (int)crank + s * (int)cs, m = u >> 2, b = u & 3;
            constexpr int CSMAX = F::gen ? 16 : 8;
            float t[CSMAX];
#pragma unroll
            for (int r = 0; r < CSMAX; ++r) t[r] = r < (int)cs ? recv[(s * (int)cs + r) * 32 + lane] : 0.f;
            float v = 0.f;
#pragma unroll
            for (int r = 0; r < CSMAX; ++r)
                if (r < (int)cs) v += t[r];
            const float sc = use_rs ? p.alpha * rs[m] : p.alpha;
            tg2_unit<F>(p, m, rt * 128 + 32 * b + lane, v * sc, tg2_sel4(qg, b), rcos, rsin);
        }
    }
    if (late && warp == 0) tmdealloc(tm, TMCOLS);
}

extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv2_fp8(const __grid_constant__ Tg2Params p) { tg2_body<0, V_GEN>(p); }
extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv2_bf16(const __grid_constant__ Tg2Params p) { tg2_body<1, V_GEN>(p); }
extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv2_qkv(const __grid_constant__ Tg2Params p) { tg2_body<0, V_QKV>(p); }
extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv2_qb(const __grid_constant__ Tg2Params p) { tg2_body<0, V_QB>(p); }
extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv2_part(const __grid_constant__ Tg2Params p) { tg2_body<0, V_PART>(p); }
extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv2_oa(const __grid_constant__ Tg2Params p) { tg2_body<1, V_OA>(p); }
extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv2_gate(const __grid_constant__ Tg2Params p) { tg2_body<1, V_GATE>(p); }
