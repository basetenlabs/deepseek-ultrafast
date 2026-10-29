#define TCMOE_RM 1
#define TCMOE_NAT 1
#include "tcmoe.cu"
#define GEMV_NO_ENTRIES 1
#include "gemv.cu"

extern "C" __global__ void __launch_bounds__(512, 1) tc_moe_w13_rm2(const __grid_constant__ TmParams p) { tc_moe_body<TM_SWIGLU>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) tc_moe_w2_rm(const __grid_constant__ TmParams p) { tc_moe_body<TM_STORE>(p); }

struct TmoeParams {
    const void* w13_map;
    const uint8_t* w13;
    long long w13_stride;
    const void* w2_map;
    const uint8_t* w2;
    long long w2_stride;
    const void* sh13_map;
    const uint8_t* sh13_sf;
    const void* sh2_map;
    const uint8_t* sh2_sf;
    uint8_t* ximg;
    uint8_t* aimg;
    float* out;
    unsigned* done;
    float* part;
    unsigned* tcnt;
    unsigned* sctr;
    int R13, K2, N2, flags;
    int kq, _pad;
};
#ifndef TM_CPS
#define TM_CPS 4
#endif
#ifndef TM_ST
#define TM_ST 3
#endif
constexpr int TM_O_SFA = TM_CPS * 16384, TM_O_B = TM_O_SFA + TM_CPS * 512;
constexpr int TM_STAGE = TM_O_B + TM_CPS * 1024;
constexpr int TM_NACC = 4;
constexpr int TM_W2T = 1;
constexpr int TM_IMG = 1536;
constexpr int TM_KT13_MAX = 40, TM_KT2_MAX = 5;
constexpr int TM_T_SFA = 8 * TM_W2T * TM_NACC, TM_T_SFX = TM_T_SFA + 4 * TM_CPS * TM_ST, TM_T_SFA2 = TM_T_SFX + 4 * TM_KT13_MAX;
static_assert(TM_T_SFA2 + 2 * 4 * TM_KT2_MAX <= 512, "TMEM columns");
static_assert(TM_KT13_MAX * 512 <= TM_STAGE, "the x scale staging fits a stage");
constexpr int TM_NQ = 8;

DEV void mb_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(s32a(b)) : "memory"); }
DEV bool mb_test(uint64_t* b, uint32_t par) {
    uint32_t r;
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(r) : "r"(s32a(b)), "r"(par) : "memory");
    return r != 0;
}
constexpr int TM_EARLY = 1 << 30;
DEV int tm_next_stage(int s, bool early) { return early ? (s ^ 1) : (s + 1 == TM_ST ? 0 : s + 1); }

DEV int tm_bimg(int row, int k) { return (k >> 4) * 128 + row * 16 + (k & 15); }

template <int CONSUMERS>
DEV void tmoe_x_item(const GemvParams& p, const GateParams& gt, int gks, int KSg, uint16_t* xg, const bf16* gw, const float* rsr, uint64_t* gbar, uint8_t* ximg, int KT) {
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
    const int nkb = KSg / 32, total = p.M * nkb;
    for (int b = warp * 8 + (lane >> 2); b - (lane >> 2) < total; b += CONSUMERS * 8) {
        const int c = lane & 3;
        const bool ok = b < total;
        const int m = ok ? b / nkb : 0, kb = ok ? b - m * nkb : 0;
        float v[8];
        if (ok) {
            const uint4 q4 = *(const uint4*)(xg + m * LDSg + kb * 32 + 8 * c);
            const uint32_t w[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = bf16x2_to_f2(w[e]);
                v[2 * e] = f.x;
                v[2 * e + 1] = f.y;
            }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = 0.f;
        }
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) a = fmaxf(a, fabsf(v[e]));
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 1));
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 2));
        if (ok) {
            const float sc = round_scale_pow2(fmaxf(a, 1e-4f), 1.0f / 448.0f);
            const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
            const int k = gks * KSg + kb * 32 + 8 * c, ch = k >> 7, kk = k & 127;
            uint32_t w0 = (uint32_t)f32x2_to_e4m3x2(v[0] * inv, v[1] * inv) | ((uint32_t)f32x2_to_e4m3x2(v[2] * inv, v[3] * inv) << 16);
            uint32_t w1 = (uint32_t)f32x2_to_e4m3x2(v[4] * inv, v[5] * inv) | ((uint32_t)f32x2_to_e4m3x2(v[6] * inv, v[7] * inv) << 16);
            *(uint2*)(ximg + (size_t)ch * 1024 + tm_bimg(m, kk)) = make_uint2(w0, w1);
            if (c == 0) ximg[(size_t)KT * 1024 + (size_t)ch * 512 + m * 16 + ((k >> 5) & 3)] = (uint8_t)((__float_as_uint(sc) >> 23) & 0xff);
        }
    }
    asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    if (tid == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(gt.ctr + 1) : "memory");
}

template <int MT>
DEV void tmoe_scores(const GemvParams& p, const GateParams& gt, int ng, float* s_scores) {
    const int tid = threadIdx.x;
    if (tid == 0) {
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

DEV void tm_gbar(int bar, int n) { asm volatile("bar.sync %0, %1;" ::"r"(bar), "r"(n) : "memory"); }
template <int MT>
DEV void tmoe_scores_g(const GemvParams& p, const GateParams& gt, int ng, float* s_scores, int gtid, int gn, int bar) {
    if (gtid == 0) {
        for (;;) {
            unsigned d;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(d) : "l"(gt.ctr) : "memory");
            if (d >= (unsigned)ng) break;
            __nanosleep(64);
        }
    }
    tm_gbar(bar, gn);
    const int nexp = p.r_nexp;
    for (int i = gtid; i < p.M * (nexp / 4); i += gn) {
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
DEV void tm_route_plan_g(const GemvParams& p, SPlan& sp, int M, const float* scores, int gtid, int gn, int bar) {
    const int lane = gtid & 31, gw = gtid >> 5;
    const int topk = p.r_topk, P = M * topk;
    tm_gbar(bar, gn);
    if (gw < M)
        route_token(gw, lane, scores, sp.bias, p.r_nexp, topk, p.r_scale, sp.sel + gw * topk, sp.selw + gw * topk,
                    nullptr, p.r_w, 1);
    tm_gbar(bar, gn);
    int first = -1, rank = 0, tot = 0, e = 0;
    if (gtid < P) {
        e = sp.sel[gtid];
        atomicOr(&sp.emask[e][gtid >> 5], 1u << (gtid & 31));
    }
    tm_gbar(bar, gn);
    bool isf = false;
    if (gtid < P) {
        const unsigned m0 = sp.emask[e][0], m1 = sp.emask[e][1], m2 = sp.emask[e][2];
        const unsigned bl = (1u << (gtid & 31)) - 1;
        const int w = gtid >> 5;
        rank = (w > 0 ? __popc(m0) : __popc(m0 & bl)) * 1 + (w == 1 ? __popc(m1 & bl) : (w > 1 ? __popc(m1) : 0)) + (w == 2 ? __popc(m2 & bl) : 0);
        tot = __popc(m0) + __popc(m1) + __popc(m2);
        first = m0 ? __ffs(m0) - 1 : (m1 ? 32 + __ffs(m1) - 1 : 64 + __ffs(m2) - 1);
        isf = first == gtid;
    }
    if (gw < 3) {
        const unsigned b = __ballot_sync(0xffffffffu, isf);
        if (lane == 0) sp.fmask[gw] = b;
    }
    tm_gbar(bar, gn);
    if (gtid < P) {
        const int fw = first >> 5;
        const unsigned fb = (1u << (first & 31)) - 1;
        int sl = 1 + __popc(sp.fmask[fw] & fb);
        for (int q = 0; q < fw; ++q) sl += __popc(sp.fmask[q]);
        sp.pair[sl][rank] = (unsigned char)gtid;
        if (isf) { sp.expert[sl] = (short)e; sp.cnt[sl] = (unsigned char)tot; }
    }
    if (gtid == 0) {
        sp.n_slots = 1 + __popc(sp.fmask[0]) + __popc(sp.fmask[1]) + __popc(sp.fmask[2]);
        sp.P = P;
    }
    tm_gbar(bar, gn);
}

DEV void tm_tma2d(void* dst, const void* tmap, int c0, int c1, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(s32a(dst)),
                 "l"(tmap), "r"(c0), "r"(c1), "r"(s32a(bar)), "l"(L2_EVICT_FIRST)
                 : "memory");
}
DEV void tm_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(s32a(dst)), "l"(src), "r"(bytes), "r"(s32a(bar)) : "memory");
}

extern "C" __global__ void __launch_bounds__(18 * 32, 1) tmoe_fused(const __grid_constant__ GemvParams p, const __grid_constant__ GateParams g, const __grid_constant__ TmoeParams t) {
    if (threadIdx.x == 0) {
        asm volatile("prefetch.tensormap [%0];" ::"l"(t.w13_map) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(t.w2_map) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(t.sh13_map) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(t.sh2_map) : "memory");
    }
    constexpr int NT = 1, PH = 4, CONSUMERS = 4 * PH, MT = 8 * NT;
    constexpr int TK = 64, STEP = 2048 + 128;
    extern __shared__ __align__(1024) uint8_t tm_smem[];
    uint8_t* smem = align1k(tm_smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ __align__(8) uint64_t full[TM_ST], empty[TM_ST], accf[TM_NACC], acce[TM_NACC], qfull[TM_NQ], qempty[TM_NQ];
    __shared__ __align__(8) uint64_t xfull, xsfdone, afull2[2], aempty2[2], planbar;
    __shared__ int sitem[TM_NQ];
    __shared__ int s_last;
    __shared__ uint32_t tbase;
    __shared__ float sup[2][32][9];
    const GemvParams* const pp_ = &p;
    const GateParams* const gt = &g;
    (void)pp_;
    const int SS = p.stage_steps;
    const int nsteps = p.K / TK;
    (void)nsteps;
    uint8_t* ring = smem;
    __shared__ __align__(8) uint64_t afull[STAGES], aempty[STAGES];
    const int LDS = p.K + 32;
    uint16_t* xs = (uint16_t*)(ring + (size_t)STAGES * SS * STEP);
    const int nkb = p.K / 32;
    float* sa = (float*)(xs + MT * LDS);
    float* red = sa + MT * nkb;
    __shared__ __align__(8) uint64_t gbar;
    __shared__ SPlan sp;
    __shared__ float rsr[16];
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) { mbar_init(&afull[s], 1); mbar_init(&aempty[s], CONSUMERS); }
        mbar_init(&gbar, 1);
        for (int s = 0; s < TM_ST; ++s) { mb_init_t(&full[s], 2); mb_init_t(&empty[s], 1); }
        for (int s = 0; s < TM_NACC; ++s) { mb_init_t(&accf[s], 1); mb_init_t(&acce[s], 4); }
        for (int s = 0; s < TM_NQ; ++s) { mb_init_t(&qfull[s], 1); mb_init_t(&qempty[s], 6); }
        mb_init_t(&xfull, 1);
        mb_init_t(&xsfdone, 1);
        mb_init_t(&planbar, 1);
        for (int s = 0; s < 2; ++s) { mb_init_t(&afull2[s], 1); mb_init_t(&aempty2[s], 1); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        tmalloc(&tbase, 512);
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    const int ng = gt->quads * gt->nks, nitg = ng + gt->nks;
    const bool gcta = (int)blockIdx.x < nitg;
    const bool xcta = (int)blockIdx.x >= ng && (int)blockIdx.x < nitg;
    const int gq = gcta ? (int)blockIdx.x / gt->nks : 0, gks = gcta ? (int)blockIdx.x % gt->nks : 0;
    const bool dots = gt->dots_fn != nullptr;
    const int nd = dots ? gt->mx.nslices * gt->dots_z : 0;
    const bool dcta = dots && (int)blockIdx.x >= nitg && (int)blockIdx.x < nitg + nd;
    const bool mcta = dots && (int)blockIdx.x == nitg + nd;
    const int dslice = dcta ? ((int)blockIdx.x - nitg) % gt->mx.nslices : 0, dz = dcta ? ((int)blockIdx.x - nitg) / gt->mx.nslices : 0;
    const bool fcta = !gcta && !dcta && !mcta;
    const int KSg = p.K / gt->nks, LDSg = KSg + 8, ntg = KSg / 16;
    const int SS16 = SS * STEP / 2048;
    const int nst_g = gcta && !xcta ? (ntg + SS16 - 1) / SS16 : 0;
    if (gcta && !xcta && warp == CONSUMERS && lane == 0) {
        const uint8_t* src = gt->w + ((size_t)gq * (p.K / 16) + (size_t)gks * ntg) * 2048;
        for (int st = 0; st < nst_g; ++st) {
            const int tiles = min(SS16, ntg - st * SS16);
            mbar_expect_tx(&afull[st], (uint32_t)tiles * 2048);
            tma_load_1d(ring + (size_t)st * SS * STEP, src + (size_t)st * SS16 * 2048, (uint32_t)tiles * 2048, &afull[st]);
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
    if (tid < p.r_nexp) sp.bias[tid] = p.r_bias[tid];
    if (tid < p.M) {
        if (tid == 0) { sp.expert[0] = -1; sp.cnt[0] = (unsigned char)p.M; }
        sp.pair[0][tid] = (unsigned char)(p.M * p.r_topk + tid);
    }
    for (int i = tid; i < 384 * 3; i += blockDim.x) (&sp.emask[0][0])[i] = 0u;
    if (tid < 3) sp.fmask[tid] = 0u;
    if (mcta) moe_mixf_arm(*gt, p.M);
    pdl_wait();
    pdl_launch();
    bf16* pw = (bf16*)(red + 4 * MT * 64);
    if (gcta && warp == CONSUMERS && lane == 0) {
        mbar_expect_tx(&gbar, (uint32_t)(p.M + 1) * KSg * 2);
        for (int m = 0; m < p.M; ++m) tma_load_1d(pw + m * LDSg, p.x + (size_t)m * p.ldx + gks * KSg, KSg * 2, &gbar);
        tma_load_1d(red, p.prenorm_w + gks * KSg, KSg * 2, &gbar);
    }
    if (dcta && warp < CONSUMERS) moe_dots_item<CONSUMERS>(*gt, p.M, dslice, dz, (const float*)xs, (float*)(xs + 2 * (MIX / gt->dots_z) * HC * SLICE), &gbar);
    if (mcta) moe_mix_item(*gt, p.M, nd, red);
    if (gcta && warp < CONSUMERS) {
        if (warp < MT) {
            float ss = 0.f;
            if (warp < p.M) {
                const int ns = p.norm_nslices;
                float v = 0.f;
                for (int sl = lane; sl < ns; sl += 32) v = __ldcg(p.norm_part + ((size_t)warp * ns + sl) * 26 + 25);
                for (int sl = 0; sl < ns; ++sl) ss += __shfl_sync(0xffffffffu, v, sl);
            }
            if (lane == 0) rsr[warp] = rsqrtf(ss / p.K + p.prenorm_eps);
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        if (xcta) tmoe_x_item<CONSUMERS>(p, *gt, gks, KSg, (uint16_t*)pw, (const bf16*)red, rsr, &gbar, t.ximg, p.K / 128);
        else moe_gate_item<CONSUMERS, MT>(p, *gt, false, gq, gks, KSg, ntg, SS16, nst_g, SS * STEP, (uint16_t*)pw, (const bf16*)red, rsr, ring, afull, aempty, &gbar, red);
    }
    if (fcta) {
        if (warp >= 4 && warp < 15) {
            float* s_scores = (float*)(smem + (TM_ST - 1) * TM_STAGE + TM_KT13_MAX * 512);
            tmoe_scores_g<MT>(p, *gt, ng, s_scores, tid - 128, 352, 8);
            tm_route_plan_g(p, sp, p.M, s_scores, tid - 128, 352, 8);
            if (tid == 128) mb_arrive(&planbar);
            return;
        }
    } else {
        float* s_scores = (float*)pw;
        tmoe_scores<MT>(p, *gt, ng, s_scores);
        route_plan(p, sp, (MoePlan*)p.plan, p.M, 1, s_scores, 1);
        if (tid == 0) mb_arrive(&planbar);
    }
    const int topk = p.r_topk, P = p.M * topk;
    const int RT13 = t.R13 / 128, KT13 = p.K / 128, RT2 = t.N2 / 128, KT2 = (t.K2 + 127) / 128;
    const int KQ = t.kq, CPQ = KT13 / KQ;
    const int RG2 = RT2 / TM_W2T;
    const int I13 = RT13 * KQ, IPS = I13 + RG2;
    (void)P;
    uint8_t* const asf = smem + TM_ST * TM_STAGE;
    uint8_t* const xstage = smem + (TM_ST - 1) * TM_STAGE;

    const bool role = warp < 4 || warp >= 15;
    if (!role) return;
    const uint32_t tm = tbase;
    auto item_info = [&](int id, bool& w2, int& sl, int& tile, int& k0, int& nch) {
        sl = id >> 16;
        const int lid = id & 0xffff;
        w2 = lid >= I13;
        if (w2) {
            tile = (lid - I13) * TM_W2T;
            k0 = 0;
            nch = TM_W2T * KT2;
        } else {
            tile = lid / KQ;
            k0 = (lid - tile * KQ) * CPQ;
            nch = CPQ;
        }
    };
    auto chunk_kc = [&](bool w2, int c) { return w2 ? min(4, (t.K2 - c * 128) >> 5) : 4; };
    auto chunk_at = [&](bool w2, int tile, int k0, int u, int& ct, int& kc) {
        ct = tile;
        kc = k0 + u;
        if (TM_W2T > 1 && w2)
            while (kc >= KT2) { kc -= KT2; ++ct; }
    };
    if (warp == 16) {
        if (lane == 0) {
            int s = 0, g = 0;
            unsigned ph = 0, used = 0;
            bool planned = !fcta, sh13 = true;
            int nsl = 0, n13r = 0;
            for (int j = 0;; ++j) {
                const int qs = j % TM_NQ;
                if (j >= TM_NQ) mb_wait_t(&qempty[qs], ((j / TM_NQ) - 1) & 1);
                int id = -1;
                bool early = false;
                if (sh13) {
                    const unsigned lid = atomicAdd(t.sctr, 1u);
                    if (lid < (unsigned)I13) {
                        id = (int)lid;
                        early = !planned && !mb_test(&planbar, 0);
                    } else {
                        sh13 = false;
                    }
                }
                if (id < 0) {
                    if (!planned) { mb_wait_t(&planbar, 0); planned = true; }
                    if (nsl == 0) { nsl = sp.n_slots; n13r = (nsl - 1) * I13; }
                    const unsigned gid = atomicAdd(p.sched, 1u);
                    if (gid < (unsigned)n13r) {
                        const int r = (int)gid / I13;
                        id = ((r + 1) << 16) | ((int)gid - r * I13);
                    } else if (gid < (unsigned)(n13r + nsl * RG2)) {
                        const int r = (int)gid - n13r, sl = r / RG2;
                        id = (sl << 16) | (I13 + r - sl * RG2);
                    }
                }
                sitem[qs] = id < 0 ? id : (id | (early ? TM_EARLY : 0));
                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(s32a(&qfull[qs])) : "memory");
                if (id < 0) break;
                bool w2; int sl, tile, k0, nch;
                item_info(id, w2, sl, tile, k0, nch);
                const int e = sp.expert[sl];
                const bool sh = e < 0;
                const int ktile = w2 ? KT2 : KT13;
                for (int c0 = 0; c0 < nch; c0 += TM_CPS) {
                    if ((used >> s) & 1) mb_wait_t(&empty[s], ((ph >> s) & 1) ^ 1);
                    else if (s == TM_ST - 1) mb_wait_t(&xsfdone, 0);
                    uint8_t* st = smem + (size_t)s * TM_STAGE;
                    const int nc = min(TM_CPS, nch - c0);
                    mb_expect_t(&full[s], (uint32_t)(nc * ((sh ? 16384 : 8192) + 512)));
                    for (int c = 0; c < nc; ++c) {
                        int ct, kc;
                        chunk_at(w2, tile, k0, c0 + c, ct, kc);
                        if (sh) {
                            tm_tma2d(st + c * 16384, w2 ? t.sh2_map : t.sh13_map, kc * 128, ct * 128, &full[s]);
                            const uint8_t* sf = (w2 ? t.sh2_sf : t.sh13_sf) + ((size_t)ct * ktile + kc) * 512;
                            bulk1d_ef(st + TM_O_SFA + c * 512, sf, 512, &full[s]);
                        } else {
                            tma3d_ef(st + c * 16384, w2 ? t.w2_map : t.w13_map, kc * 128, ct * 128, e, &full[s]);
                            const uint8_t* base = w2 ? t.w2 + (size_t)e * t.w2_stride + (size_t)t.N2 * (KT2 * 64)
                                                     : t.w13 + (size_t)e * t.w13_stride + (size_t)t.R13 * (p.K / 2);
                            bulk1d_ef(st + TM_O_SFA + c * 512, base + ((size_t)ct * ktile + kc) * 512, 512, &full[s]);
                        }
                    }
                    ph ^= 1u << s;
                    used |= 1u << s;
                    ++g;
                    s = tm_next_stage(s, early);
                }
            }
        }
    } else if (warp == 15) {
        if (lane == 0) {
            for (;;) {
                unsigned d;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(d) : "l"(gt->ctr + 1) : "memory");
                if (d >= (unsigned)gt->nks) break;
                __nanosleep(32);
            }
            asm volatile("fence.proxy.async.global;" ::: "memory");
            mb_expect_t(&xfull, (uint32_t)(KT13 * 512));
            tm_bulk(xstage, t.ximg + (size_t)KT13 * 1024, KT13 * 512, &xfull);
            int s = 0, g = 0, na = 0;
            unsigned ph = 0, used = 0;
            for (int j = 0;; ++j) {
                const int qs = j % TM_NQ;
                mb_wait_t(&qfull[qs], (j / TM_NQ) & 1);
                int id = sitem[qs];
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&qempty[qs])) : "memory");
                if (id < 0) break;
                const bool early = (id & TM_EARLY) != 0;
                id &= ~TM_EARLY;
                bool w2; int sl, tile, k0, nch;
                item_info(id, w2, sl, tile, k0, nch);
                const uint8_t* img = t.ximg;
                if (w2) {
                    const int ab = na & 1;
                    for (;;) {
                        unsigned d;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(d) : "l"(t.done + sl) : "memory");
                        if (d >= (unsigned)RT13) break;
                        __nanosleep(32);
                    }
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    img = t.aimg + (size_t)sl * KT2 * TM_IMG;
                    if (na >= 1) mb_wait_t(&aempty2[0], (na - 1) & 1);
                    mb_expect_t(&afull2[ab], (uint32_t)(KT2 * 512));
                    tm_bulk(asf, img + (size_t)KT2 * 1024, KT2 * 512, &afull2[ab]);
                    ++na;
                }
                for (int c0 = 0; c0 < nch; c0 += TM_CPS) {
                    if ((used >> s) & 1) mb_wait_t(&empty[s], ((ph >> s) & 1) ^ 1);
                    uint8_t* st = smem + (size_t)s * TM_STAGE;
                    const int nc = min(TM_CPS, nch - c0);
                    mb_expect_t(&full[s], (uint32_t)(nc * 1024));
                    for (int c = 0; c < nc;) {
                        int ct, kc;
                        chunk_at(w2, tile, k0, c0 + c, ct, kc);
                        int n = 1;
                        while (c + n < nc) {
                            int ct2, kc2;
                            chunk_at(w2, tile, k0, c0 + c + n, ct2, kc2);
                            if (kc2 != kc + n) break;
                            ++n;
                        }
                        tm_bulk(st + TM_O_B + c * 1024, img + (size_t)kc * 1024, n * 1024, &full[s]);
                        c += n;
                    }
                    ph ^= 1u << s;
                    used |= 1u << s;
                    ++g;
                    s = tm_next_stage(s, early);
                }
            }
        }
    } else if (warp == 17) {
        const uint32_t tmu = uni(tm);
        mb_wait_t(&xfull, 0);
        tcafter();
        if (elect_one()) {
            const uint64_t fx = sdesc_t(xstage, 512, 128, 0);
            for (int kc = 0; kc < KT13; ++kc) tccpsf(tmu + TM_T_SFX + 4 * kc, fx + (uint64_t)((kc * 512) >> 4));
            tccommit(&xsfdone);
        }
        __syncwarp();
        int s = 0, gm = 0, na = 0;
        unsigned ph = 0;
        for (int j = 0;; ++j) {
            const int qs = j % TM_NQ;
            mb_wait_t(&qfull[qs], (j / TM_NQ) & 1);
            int id = sitem[qs];
            __syncwarp();
            if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&qempty[qs])) : "memory");
            if (id < 0) break;
            const bool early = (id & TM_EARLY) != 0;
            id &= ~TM_EARLY;
            bool w2; int sl, tile, k0, nch;
            item_info(id, w2, sl, tile, k0, nch);
            const bool sh = sp.expert[sl] < 0;
            const uint32_t id0 = sh ? idesc_e4m3(8, 0) : idesc_e2m1_e4m3(8, 0);
            const int da = j % TM_NACC;
            const int ab = na & 1;
            uint32_t tsfb0 = tmu + TM_T_SFX;
            if (w2) {
                mb_wait_t(&afull2[ab], (na >> 1) & 1);
                tcafter();
                tsfb0 = tmu + TM_T_SFA2 + 4 * KT2 * ab;
                if (elect_one()) {
                    const uint64_t fa = sdesc_t(asf, 512, 128, 0);
                    for (int kc = 0; kc < KT2; ++kc) tccpsf(tsfb0 + 4 * kc, fa + (uint64_t)((kc * 512) >> 4));
                    tccommit(&aempty2[0]);
                }
                __syncwarp();
            }
            if (j >= TM_NACC) mb_wait_t(&acce[da], ((j / TM_NACC) - 1) & 1);
            tcafter();
            const uint32_t d0 = tmu + 8 * TM_W2T * da;
            for (int c0 = 0; c0 < nch; c0 += TM_CPS) {
                mb_wait_t(&full[s], (ph >> s) & 1);
                tcafter();
                const int nc = min(TM_CPS, nch - c0);
                const uint32_t tsfa = tmu + TM_T_SFA + 4 * TM_CPS * s;
                if (elect_one()) {
                    uint8_t* st = smem + (size_t)s * TM_STAGE;
                    const uint64_t a0 = sdesc_t(st, 16, 1024, 2), b0 = sdesc_t(st + TM_O_B, 128, 1024, 0);
                    const uint64_t fa = sdesc_t(st + TM_O_SFA, 512, 128, 0);
                    if (nc == TM_CPS && (!w2 || (k0 + c0 + TM_CPS) * 128 <= t.K2)) {
#pragma unroll
                        for (int c = 0; c < TM_CPS; ++c) {
                            tccpsf(tsfa + 4 * c, fa + (uint64_t)((c * 512) >> 4));
                            const uint32_t sfb = tsfb0 + 4 * (k0 + c0 + c);
#pragma unroll
                            for (int kc = 0; kc < 4; ++kc)
                                tcmma(d0, a0 + (uint64_t)((c * 16384 + kc * 32) >> 4), b0 + (uint64_t)((c * 1024 + kc * 256) >> 4),
                                      id0 | ((uint32_t)kc << 29) | ((uint32_t)kc << 4), tsfa + 4 * c, sfb, (c0 | c | kc) != 0);
                        }
                    } else {
                        for (int c = 0; c < nc; ++c) {
                            tccpsf(tsfa + 4 * c, fa + (uint64_t)((c * 512) >> 4));
                            int ct, kcx;
                            chunk_at(w2, tile, k0, c0 + c, ct, kcx);
                            const uint32_t d = d0 + 8 * (ct - tile);
                            const int nkc = chunk_kc(w2, kcx);
                            const bool first = w2 ? kcx == 0 : c0 + c == 0;
                            for (int kc = 0; kc < nkc; ++kc)
                                tcmma(d, a0 + (uint64_t)((c * 16384 + kc * 32) >> 4), b0 + (uint64_t)((c * 1024 + kc * 256) >> 4),
                                      id0 | ((uint32_t)kc << 29) | ((uint32_t)kc << 4), tsfa + 4 * c, tsfb0 + 4 * kcx, !(first && kc == 0));
                        }
                    }
                    tccommit(&empty[s]);
                }
                __syncwarp();
                ph ^= 1u << s;
                ++gm;
                s = tm_next_stage(s, early);
            }
            if (elect_one()) tccommit(&accf[da]);
            __syncwarp();
            if (w2) ++na;
        }
    } else {
        const int q = warp;
        const float L = p.swiglu_limit;
        for (int j = 0;; ++j) {
            const int qs = j % TM_NQ;
            mb_wait_t(&qfull[qs], (j / TM_NQ) & 1);
            const int id0 = sitem[qs];
            __syncwarp();
            if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&qempty[qs])) : "memory");
            if (id0 < 0) break;
            const int id = id0 & ~TM_EARLY;
            bool w2; int sl, tile, k0, nch;
            item_info(id, w2, sl, tile, k0, nch);
            const int da = j % TM_NACC;
            mb_wait_t(&accf[da], (j / TM_NACC) & 1);
            tcafter();
            const int cnt = sp.cnt[sl];
            const uint32_t dq = tm + 8 * TM_W2T * da + ((uint32_t)(32 * q) << 16);
            if (w2) {
                uint32_t r[TM_W2T][8];
#pragma unroll
                for (int ti = 0; ti < TM_W2T; ++ti)
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                                 : "=r"(r[ti][0]), "=r"(r[ti][1]), "=r"(r[ti][2]), "=r"(r[ti][3]), "=r"(r[ti][4]), "=r"(r[ti][5]), "=r"(r[ti][6]), "=r"(r[ti][7])
                                 : "r"(dq + 8 * ti)
                                 : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                tcbefore();
                __syncwarp();
                if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&acce[da])) : "memory");
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    if (k < cnt) {
                        float* o = t.out + (size_t)sp.pair[sl][k] * t.N2 + tile * 128 + 32 * q + lane;
#pragma unroll
                        for (int ti = 0; ti < TM_W2T; ++ti) o[ti * 128] = __uint_as_float(r[ti][k]);
                    }
                continue;
            }
            uint32_t r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                         : "r"(dq)
                         : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            tcbefore();
            __syncwarp();
            if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&acce[da])) : "memory");
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
            if (KQ > 1) {
                const int kq = k0 / CPQ;
                const size_t tb = ((size_t)sl * RT13 + tile) * KQ;
                float4* mp = (float4*)(t.part + ((tb + kq) * 128 + 32 * q + lane) * 8);
                mp[0] = make_float4(v[0], v[1], v[2], v[3]);
                mp[1] = make_float4(v[4], v[5], v[6], v[7]);
                asm volatile("bar.sync 6, 128;" ::: "memory");
                if (q == 0 && lane == 0) {
                    unsigned old;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(t.tcnt + (size_t)sl * RT13 + tile) : "memory");
                    s_last = old == (unsigned)(KQ - 1);
                }
                asm volatile("bar.sync 6, 128;" ::: "memory");
                if (!s_last) {
                    continue;
                }
                float acc8[8];
                for (int k = 0; k < KQ; ++k) {
                    float w8[8];
                    if (k == kq) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) w8[i] = v[i];
                    } else {
                        const float4* op = (const float4*)(t.part + ((tb + k) * 128 + 32 * q + lane) * 8);
                        const float4 a0 = __ldcg(op), a1 = __ldcg(op + 1);
                        w8[0] = a0.x; w8[1] = a0.y; w8[2] = a0.z; w8[3] = a0.w;
                        w8[4] = a1.x; w8[5] = a1.y; w8[6] = a1.z; w8[7] = a1.w;
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc8[i] = k == 0 ? w8[i] : acc8[i] + w8[i];
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) v[i] = acc8[i];
            }
            if (q & 1) {
#pragma unroll
                for (int i = 0; i < 8; ++i) sup[q >> 1][lane][i] = v[i];
            }
            asm volatile("bar.sync 6, 128;" ::: "memory");
            if (!(q & 1)) {
                const int col = tile * 64 + (q >> 1) * 32 + lane;
                const int ch = col >> 7, kk = col & 127;
                uint8_t* img = t.aimg + (size_t)sl * KT2 * TM_IMG;
                for (int k = 0; k < cnt; ++k) {
                    const int pr = sp.pair[sl][k];
                    const int tok = pr < P ? pr / topk : pr - P;
                    const float wgt = pr < P ? sp.selw[pr] : 1.0f;
                    float gv = v[0], uv = sup[q >> 1][lane][0];
#pragma unroll
                    for (int i = 1; i < 8; ++i)
                        if (i == tok) { gv = v[i]; uv = sup[q >> 1][lane][i]; }
                    float gte = __bfloat162float(__float2bfloat16(gv));
                    float up = __bfloat162float(__float2bfloat16(uv));
                    if (L > 0.f) { up = fminf(fmaxf(up, -L), L); gte = fminf(gte, L); }
                    const float a = __bfloat162float(__float2bfloat16(gte / (1.0f + expf(-gte)) * up * wgt));
                    const float sc = round_scale_pow2(fmaxf(warp_max(fabsf(a)), 1e-4f), 1.0f / 448.0f);
                    const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
                    img[(size_t)ch * 1024 + tm_bimg(k, kk)] = (uint8_t)(f32x2_to_e4m3x2(a * inv, 0.f) & 0xff);
                    if (lane == 0) img[(size_t)KT2 * 1024 + (size_t)ch * 512 + k * 16 + ((col >> 5) & 3)] = (uint8_t)((__float_as_uint(sc) >> 23) & 0xff);
                }
            }
            asm volatile("bar.sync 6, 128;" ::: "memory");
            if (q == 0 && lane == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(t.done + sl) : "memory");
        }
    }
    __syncwarp();
    tcbefore();
    asm volatile("bar.sync 7, 224;" ::: "memory");
    tcafter();
    if (warp == 0) tmdealloc(tm, 512);
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(p.sched + 1, 1u) == gridDim.x - 1) {
            p.sched[0] = 0;
            p.sched[1] = 0;
            gt->ctr[0] = 0;
            gt->ctr[1] = 0;
            if (dots) *gt->dots_ctr = 0;
            const int nsl = __ldcg(&((const MoePlan*)p.plan)->n_slots);
            for (int i = 0; i < nsl; ++i) { t.done[i] = 0; t.sctr[i] = 0; }
            if (t.kq > 1)
                for (int i = 0; i < nsl * (t.R13 / 128); ++i) t.tcnt[i] = 0;
            __threadfence();
        }
    }
}
