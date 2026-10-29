#include "tcgemv2.cu"

enum { TG3_UNI = 16, TG3_CHAIN = 32, TG3_EPI = 64, TG3_GSIDE = 128, TG3_REL = 256, TG3_XTMA = 512, TG3_GLATE = 1024, TG3_M64 = 2048 };

DEV void tg3_wait(uint32_t ba, uint32_t par) {
    asm volatile(
        "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\tbra.uni LAB_WAIT;\n\tDONE:\n\t}" ::"r"(ba),
        "r"(par), "r"(0x989680u)
        : "memory");
}
DEV void tg3_cp(uint32_t taddr, uint64_t desc) {
    asm volatile("{\n\t.reg .pred e;\n\telect.sync _|e, -1;\n\t@e tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}" ::"r"(taddr), "l"(desc) : "memory");
}
DEV void tg3_mma_f8(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
    asm volatile(
        "{\n\t.reg .pred e, p;\n\telect.sync _|e, -1;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb)
        : "memory");
}
DEV void tg3_mma_f16(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
    asm volatile(
        "{\n\t.reg .pred e, p;\n\telect.sync _|e, -1;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(acc)
        : "memory");
}
DEV void tg3_commit(uint32_t ba) {
    asm volatile("{\n\t.reg .pred e;\n\telect.sync _|e, -1;\n\t@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}" ::"r"(ba)
                 : "memory");
}
DEV void tg3_arrive(uint32_t ba) {
    asm volatile("{\n\t.reg .pred e;\n\telect.sync _|e, -1;\n\t@e mbarrier.arrive.shared::cta.b64 _, [%0];\n\t}" ::"r"(ba) : "memory");
}
DEV void tg3_st_async1(uint32_t raddr, float v, uint32_t rbar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(raddr), "r"(__float_as_uint(v)), "r"(rbar) : "memory");
}
DEV uint32_t tg3_mapa(uint32_t saddr, uint32_t rank) {
    uint32_t r;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(saddr), "r"(rank));
    return r;
}

template <class F>
DEV void tg3_units_qb(const Tg2Params& p, const int (&mm)[4], int nu, int r, const float (&v)[4], const float* rcos, const float* rsin) {
    static_assert(!F::gen && !F::qimg && !F::ssp && !F::f32o, "tg3_units_qb: the per-use wq_b paths only");
    constexpr uint32_t FULL = 0xffffffffu;
    if (F::split && p.n_split && r >= p.n_split) {
        const int rr = r - p.n_split;
        if (F::iq && p.iq_head) {
            const int d = rr & (p.iq_head - 1);
            float x[4], pt[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) x[j] = tg2_bfr(v[j]);
#pragma unroll
            for (int j = 0; j < 4; ++j) pt[j] = __shfl_xor_sync(FULL, x[j], 1);
            if (d >= p.iq_off) {
                const int f = (d - p.iq_off) >> 1;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (j < nu) {
                        const float c = rcos[mm[j] * 32 + f], sn = rsin[mm[j] * 32 + f];
                        x[j] = (r & 1) ? __fmaf_rn(pt[j], sn, __fmul_rn(x[j], c)) : __fmaf_rn(x[j], c, -__fmul_rn(pt[j], sn));
                        x[j] = tg2_bfr(x[j]);
                    }
            }
            float am[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) am[j] = fabsf(x[j]);
#pragma unroll
            for (int o = 1; o < 32; o <<= 1)
#pragma unroll
                for (int j = 0; j < 4; ++j) am[j] = fmaxf(am[j], __shfl_xor_sync(FULL, am[j], o));
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (j < nu) {
                    const float a = fmaxf(am[j], 6.f * 1.1754943508222875e-38f);
                    const float sc = round_scale_pow2(a, 1.0f / 6.0f);
                    const float y = fp4_round(fminf(fmaxf(__fdiv_rn(x[j], sc), -6.f), 6.f)) * sc;
                    ((bf16*)p.out2)[(size_t)mm[j] * p.ldo2 + rr] = __float2bfloat16(y);
                }
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (j < nu) ((bf16*)p.out2)[(size_t)mm[j] * p.ldo2 + rr] = __float2bfloat16(v[j]);
        }
        return;
    }
    const int rh = F::rope ? r & (p.rope_head - 1) : 0;
    if (F::rope && p.rope_cos && rh >= p.rope_off) {
        float a[4], o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) a[j] = tg2_bfr(v[j]);
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __shfl_xor_sync(FULL, a[j], 1);
        const int f = (rh - p.rope_off) >> 1;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (j < nu) {
                const float c = rcos[mm[j] * 32 + f], sn = rsin[mm[j] * 32 + f];
                const float y = (r & 1) ? __fmaf_rn(o[j], sn, __fmul_rn(a[j], c)) : __fmaf_rn(a[j], c, -__fmul_rn(o[j], sn));
                ((bf16*)p.out)[(size_t)mm[j] * p.ldo + r] = __float2bfloat16(y);
            }
        return;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        if (j < nu) ((bf16*)p.out)[(size_t)mm[j] * p.ldo + r] = __float2bfloat16(v[j]);
}

template <int KIND, int VAR, int OPT>
DEV void tg3_body(const Tg2Params& p) {
    static_assert(VAR != V_GEN, "per-use entries only");
    using F = TgF<KIND, VAR>;
    constexpr bool REL = (OPT & TG3_REL) != 0, M64 = (OPT & TG3_M64) != 0;
    static_assert(!M64 || (KIND == 1 && VAR == V_OA), "TG3_M64: wo_a");
    constexpr int RH = M64 ? 64 : 128, ABOX = KIND == 0 ? 16384 : RH * 128, NWB = M64 ? 16 : TG2_KT;
    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
    if (F::kvr && p.kvr_on && (int)blockIdx.x >= (int)gridDim.x - p.kvr_on) {
        const int grp = tid >> 6;
        const int nb = p.kvr_on * (int)gridDim.y;
        const int b = ((int)blockIdx.x - ((int)gridDim.x - p.kvr_on)) * (int)gridDim.y + (int)blockIdx.y;
        for (int row = b * 2 + grp; row < 16 * p.kvr.rows_per_token; row += 2 * nb) rope_body(p.meta, p.kvr, row, tid & 63, grp);
        return;
    }
    if (tid == 0) asm volatile("prefetch.tensormap [%0];" ::"l"(p.tmap) : "memory");
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* sm = align1k(smem_raw);
    uint32_t crank, cs;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cs));
    const int rt = blockIdx.x;
    const int M = p.M;
    constexpr int TKB = KIND == 0 ? 128 : 64;
    const int KS = p.K / (int)cs, KT = KS / TKB, k0 = (int)crank * KS;
    const bool img = F::img || (KIND == 0 && VAR == V_QKV && p.bimg != 0);
    const bool xq = F::xq && !img;
    const bool has_gamma = F::gamma && p.gamma;
    uint8_t* sA = sm;
    uint8_t* sB = sA + KT * ABOX;
    uint8_t* sSFA = sB + (KIND == 0 ? KT * 1024 : KS * 16);
    uint8_t* sSFB = sSFA + (KIND == 0 ? KT * 512 : 0);
    bf16* gs = (bf16*)(sSFB + (KIND == 0 ? KT * 512 : 0));
    const bool xraw_on = F::xqout && p.rs_pre == 2 && p.xq;
    bf16* xraw = gs + (has_gamma ? KS : 0);
    float* recv = (float*)(xraw + (xraw_on ? 8 * KS : 0));
    __shared__ __align__(8) uint64_t wbar[NWB], xbar, gbar, mbar, rbar, tbar, sbar;
    __shared__ uint32_t tbase;
    __shared__ float rs[8];
    __shared__ float rcos[8 * 32], rsin[8 * 32];
    __shared__ __align__(16) float red[4];
    const bool partial = F::part;
    const bool push = cs > 1 && !partial;
    const int nunits = (M64 ? 2 : 4) * M;
    const int nown = (int)crank < nunits ? (nunits - (int)crank + (int)cs - 1) / (int)cs : 0;
    constexpr uint32_t TMCOLS = KIND == 0 ? 128 : 32;
    const bool bload = img && p.bload;
    const bool rspre = F::rspre && p.rs_part && p.rs_pre == 1;
    const bool rspost = F::rspre && p.rs_part && p.rs_pre == 2;
    const bool rsdef = F::rsdef && p.rs_part && !(F::rspre && p.rs_pre);
    const bool rope_on = F::rope && p.rope_cos != nullptr;
    const int nsb = (rope_on ? 1 : 0) + ((rsdef || rspost) ? 1 : 0);
    if (warp == 0) tmalloc(&tbase, TMCOLS);
    if constexpr (REL) {
        if (warp == 0) asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    if (tid == 0) {
        for (int t = 0; t < NWB; ++t) mb_init_t(&wbar[t], 1);
        mb_init_t(&xbar, 1);
        mb_init_t(&gbar, 1);
        mb_init_t(&mbar, 1);
        mb_init_t(&rbar, 1);
        mb_init_t(&tbar, 4);
        if (nsb) mb_init_t(&sbar, nsb);
        if (push) mb_expect_t(&rbar, (uint32_t)(nown * (int)cs * 128));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (push) asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    const uint32_t tm = tbase;
    if (tid == 0) {
        if (has_gamma) {
            mb_expect_t(&gbar, KS * 2);
            bulk1d(gs, p.gamma + k0, KS * 2, &gbar);
        }
        if (KIND == 0) {
            for (int t = 0; t < KT; ++t) {
                mb_expect_t(&wbar[t], 16384 + (t == 0 ? KT * 512 : 0));
                tma2d_ef(sA + t * 16384, p.tmap, k0 + t * 128, rt * 128, &wbar[t]);
                if (t == 0) bulk1d(sSFA, p.sfa + ((size_t)rt * p.kpt + k0 / 128) * 512, KT * 512, &wbar[0]);
            }
        } else {
            for (int t = 0; t < KT; ++t) {
                mb_expect_t(&wbar[t], ABOX);
                tma2d_ef(sA + t * ABOX, p.tmap, k0 + t * 64, rt * RH, &wbar[t]);
            }
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
    if (tid == 0 && img && !bload) {
        mb_expect_t(&xbar, KT * 1024 + KT * 512);
        bulk1d(sB, p.bimg + (size_t)(k0 / 128) * 1024, KT * 1024, &xbar);
        bulk1d(sSFB, p.bimg + (size_t)(p.K / 128) * 1024 + (size_t)(k0 / 128) * 512, KT * 512, &xbar);
    }
    const int nkb = KS / 32, xtotal = xq ? M * nkb : 0;
    const float rnkb = 1.0f / (float)nkb;
    uint4 xr[F::xq ? 8 : 1];
    if (F::xq && xq) {
#pragma unroll
        for (int i = 0; i < (F::xq ? 8 : 1); ++i) {
            const int b = warp * 8 + (lane >> 2) + 32 * i;
            xr[i] = make_uint4(0, 0, 0, 0);
            if (b < xtotal) {
                const int m = __float2int_rd(((float)b + 0.5f) * rnkb);
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
    if (rope_on && warp == 2) {
        const int pos0 = p.meta[0];
        for (int m = 0; m < M; ++m) {
            rcos[m * 32 + lane] = p.rope_cos[(size_t)(pos0 + m) * 32 + lane];
            rsin[m * 32 + lane] = p.rope_sin[(size_t)(pos0 + m) * 32 + lane];
        }
        __syncwarp();
        tg3_arrive(s32a(&sbar));
    }
    const size_t xoff = F::group && p.group_rows ? (size_t)((rt * RH) / p.group_rows) * p.K : 0;
    constexpr int XVN = KIND == 1 ? (M64 ? 8 : 5) : 1;
    uint4 xv[XVN];
    const int nch = KIND == 1 ? (KS / 8) * 8 : 0;
    if (KIND == 1) {
#pragma unroll
        for (int j = 0; j < XVN; ++j) {
            const int i = tid + j * 128, m = i & 7;
            xv[j] = make_uint4(0, 0, 0, 0);
            if (i < nch && m < M) xv[j] = *(const uint4*)(p.x + (size_t)m * p.ldx + xoff + k0 + (i >> 3) * 8);
        }
    }
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
        for (int j = 0; j < XVN; ++j) {
            const int i = tid + j * 128, c = i >> 3, m = i & 7;
            if (i >= nch) break;
            uint4 v = xv[j];
            if (xraw_on && (rt == 0)) *(uint4*)(xraw + m * KS + c * 8) = v;
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
            const int m = __float2int_rd(((float)bb + 0.5f) * rnkb), kl = (bb - m * nkb) * 32 + c * 8;
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
        __syncwarp();
        const uint32_t tmu = __shfl_sync(0xffffffffu, tm, 0);
        const uint32_t wb = s32a(&wbar[0]);
        if (KIND == 0) {
            if (img && !bload) tg3_wait(s32a(&xbar), 0);
            tcafter();
            const uint64_t sfb0 = sdesc_t(sSFB, 512, 128, 0);
            for (int t = 0; t < KT; ++t) tg3_cp(tmu + 64 + 4 * t, sfb0 + (uint64_t)(t * 32));
        }
        const uint64_t a0 = sdesc_t(sA, 16, 1024, 2), b0 = sdesc_t(sB, 128, 1024, 0);
        for (int t = 0; t < KT; ++t) {
            tg3_wait(wb + 8u * (uint32_t)t, 0);
            tcafter();
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) {
                const uint64_t ad = a0 + (uint64_t)(t * (ABOX / 16) + kc * 2), bd = b0 + (uint64_t)(t * 64 + kc * 16);
                const uint32_t acc = (t | kc) != 0;
                if (KIND == 0) tg3_mma_f8(tmu, ad, bd, idesc_e4m3(8, kc), tmu + 32 + 4 * t, tmu + 64 + 4 * t, acc);
                else tg3_mma_f16(tmu, ad, bd, M64 ? idesc_bf16(8) - ((128u >> 4) << 24) + ((64u >> 4) << 24) : idesc_bf16(8), acc);
            }
        }
        tg3_commit(s32a(&mbar));
    }
    if (rspost) row_stat();
    const int kb_lo = 0, kb_hi = KS / 32;
    if (F::xqout && p.xq && (rt == 0) && kb_hi > kb_lo && warp != 1) {
        if (rspost) asm volatile("bar.sync 2, 96;" ::: "memory");
        const int qw = warp == 0 ? 0 : warp - 1;
        quant_blocks8(qw, 3, lane, 8, M, kb_hi - kb_lo, p.xq + k0 + kb_lo * 32, p.K + 32, p.xsq + (k0 / 32 + kb_lo) * 8, [&](int m, int kb, int c, float* v) {
            kb += kb_lo;
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
        float sm_ = a[0];
#pragma unroll
        for (int m = 1; m < 8; ++m)
            if (lane == m) sm_ = a[m];
        if (lane < 8) rs[lane] = rsqrtf(sm_ / p.K + p.eps);
    }
    if ((rsdef || rspost) && warp == 3) {
        __syncwarp();
        tg3_arrive(s32a(&sbar));
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
    const bool use_rs = rsdef || rspost;
    tcbefore();
    tg3_arrive(s32a(&tbar));
    if (partial) {
        const int row = rt * 128 + 32 * warp + lane;
#pragma unroll
        for (int m = 0; m < 8; ++m)
            if (m < M) ((float*)p.out)[((size_t)m * cs + crank) * p.ldo + row] = acc[m] * p.alpha;
    } else {
        const uint32_t rcv = s32a(recv), rbs = s32a(&rbar);
        if constexpr (M64) {
            const int qv = lane & 3, mv = lane >> 2;
            float4 vv = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                if (m < M) {
                    const float a0 = __shfl_sync(0xffffffffu, acc[m], 4 * qv), a1 = __shfl_sync(0xffffffffu, acc[m], 4 * qv + 1);
                    const float a2 = __shfl_sync(0xffffffffu, acc[m], 4 * qv + 2), a3 = __shfl_sync(0xffffffffu, acc[m], 4 * qv + 3);
                    if (m == mv) vv = make_float4(a0, a1, a2, a3);
                }
            }
            if (mv < M) {
                const int u = 2 * mv + (warp >> 1), o = u % (int)cs, slot = u / (int)cs;
                asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(tg3_mapa(rcv + (uint32_t)(((slot * (int)cs + (int)crank) * 32 + 16 * (warp & 1) + 4 * qv) * 4), (uint32_t)o)),
                             "f"(vv.x), "f"(vv.y), "f"(vv.z), "f"(vv.w), "r"(tg3_mapa(rbs, (uint32_t)o))
                             : "memory");
            }
        } else {
        const int qv = lane & 7, gv = lane >> 3;
        float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < M) {
                const float a0 = __shfl_sync(0xffffffffu, acc[m], 4 * qv), a1 = __shfl_sync(0xffffffffu, acc[m], 4 * qv + 1);
                const float a2 = __shfl_sync(0xffffffffu, acc[m], 4 * qv + 2), a3 = __shfl_sync(0xffffffffu, acc[m], 4 * qv + 3);
                if (m == gv) v0 = make_float4(a0, a1, a2, a3);
                if (m == gv + 4) v1 = make_float4(a0, a1, a2, a3);
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int m = gv + 4 * h;
            if (m < M) {
                const int u = 4 * m + warp, o = u % (int)cs, slot = u / (int)cs;
                const float4 vv = h ? v1 : v0;
                asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(tg3_mapa(rcv + (uint32_t)(((slot * (int)cs + (int)crank) * 32 + 4 * qv) * 4), (uint32_t)o)),
                             "f"(vv.x), "f"(vv.y), "f"(vv.z), "f"(vv.w), "r"(tg3_mapa(rbs, (uint32_t)o))
                             : "memory");
            }
        }
        }
    }
    __syncwarp();
    if (warp == 0) {
        tg3_wait(s32a(&tbar), 0);
        tcafter();
        tmdealloc(tm, TMCOLS);
    }
    if (!partial) {
        tg3_wait(s32a(&rbar), 0);
        if (nsb) tg3_wait(s32a(&sbar), 0);
        for (int s = (warp + 3) & 3; s < nown; s += 4) {
            const int u = (int)crank + s * (int)cs, m = M64 ? u >> 1 : u >> 2, b = M64 ? u & 1 : u & 3;
            constexpr int CSMAX = 8;
            float t[CSMAX];
#pragma unroll
            for (int r = 0; r < CSMAX; ++r) t[r] = r < (int)cs ? recv[(s * (int)cs + r) * 32 + lane] : 0.f;
            float v = 0.f;
#pragma unroll
            for (int r = 0; r < CSMAX; ++r)
                if (r < (int)cs) v += t[r];
            const float sc = use_rs ? p.alpha * rs[m] : p.alpha;
            tg2_unit<F>(p, m, rt * RH + 32 * b + lane, v * sc, tg2_sel4(qg, b), rcos, rsin);
        }
    }
}

#define TG3_ENTRIES(SUF, O)                                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv3##SUF##_qkv(const __grid_constant__ Tg2Params p) { tg3_body<0, V_QKV, O>(p); }             \
    extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv3##SUF##_qb(const __grid_constant__ Tg2Params p) { tg3_body<0, V_QB, O>(p); }               \
    extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv3##SUF##_part(const __grid_constant__ Tg2Params p) { tg3_body<0, V_PART, O>(p); }           \
    extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv3##SUF##_oa(const __grid_constant__ Tg2Params p) { tg3_body<1, V_OA, O>(p); }
TG3_ENTRIES(b, TG3_UNI | TG3_CHAIN)
extern "C" __global__ void __launch_bounds__(128, 1) tc_gemv3m_oa(const __grid_constant__ Tg2Params p) { tg3_body<1, V_OA, TG3_UNI | TG3_CHAIN | TG3_M64>(p); }
