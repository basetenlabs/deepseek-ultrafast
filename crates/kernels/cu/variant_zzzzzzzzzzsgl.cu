#include "tc.cuh"

namespace sglm {
constexpr int TILE = 8, HID = 5120, SLICE = 576, PADDED = 640, SLICE_TP8 = 320, PADDED_TP8 = 384;
constexpr int A_CHUNK = 16384, SF_ATOM = 512, B_CHUNK = 1024;
struct alignas(64) CUtensorMap { unsigned long long opaque[16]; };

struct SgmParams {
    const CUtensorMap* tm_a;
    const CUtensorMap* tm_b;
    const CUtensorMap* tm_sh;
    const uint8_t* sfa_sh;
    float sh_clamp;
    int has_sh;
    int m_rows;
    const uint8_t* sfa;
    long long sfa_stride;
    const uint8_t* sfb;
    void* out;
    uint8_t* hs;
    const int* record;
    const float* clamp;
    int rows;
    int cta_words;
    int trigger_at_start;
    int _pad;
};

DEV void pdl_wait_() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEV void pdl_launch_() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
DEV void mb_arrive_(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(b)) : "memory"); }
DEV void tma3d_ef_(void* dst, const CUtensorMap* tmap, int c0, int c1, int c2, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(s32a(dst)),
                 "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(s32a(bar)), "l"(L2_EVICT_FIRST)
                 : "memory");
}
DEV void tma2d_ef_(void* dst, const CUtensorMap* tmap, int c0, int c1, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(s32a(dst)),
                 "l"(tmap), "r"(c0), "r"(c1), "r"(s32a(bar)), "l"(L2_EVICT_FIRST)
                 : "memory");
}
DEV void tma2d_(void* dst, const CUtensorMap* tmap, int c0, int c1, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(s32a(dst)), "l"(tmap),
                 "r"(c0), "r"(c1), "r"(s32a(bar))
                 : "memory");
}
DEV void prefetch_tmap(const CUtensorMap* t) { asm volatile("prefetch.tensormap [%0];" ::"l"(t) : "memory"); }

DEV void clamped_swiglu_x2(float up0, float up1, float gate0, float gate1, float limit, float neg_limit, float& h0, float& h1) {
    asm(
        "{\n"
        ".reg .pred sw_p;\n"
        ".reg .f32 sw_u0, sw_u1, sw_g0, sw_g1, sw_x0, sw_x1;\n"
        ".reg .b32 sw_r0, sw_r1;\n"
        ".reg .b64 sw_up, sw_gate, sw_one, sw_zero, sw_l2e, sw_neg, sw_f, sw_a, sw_d;\n"
        "setp.lt.ftz.f32 sw_p, %6, %2;\n"
        "selp.f32 sw_u0, %6, %2, sw_p;\n"
        "setp.lt.ftz.f32 sw_p, %2, %7;\n"
        "selp.f32 sw_u0, %7, sw_u0, sw_p;\n"
        "setp.lt.ftz.f32 sw_p, %6, %3;\n"
        "selp.f32 sw_u1, %6, %3, sw_p;\n"
        "setp.lt.ftz.f32 sw_p, %3, %7;\n"
        "selp.f32 sw_u1, %7, sw_u1, sw_p;\n"
        "setp.lt.ftz.f32 sw_p, %6, %4;\n"
        "selp.f32 sw_g0, %6, %4, sw_p;\n"
        "setp.lt.ftz.f32 sw_p, %4, 0fFF7FFFFF;\n"
        "selp.f32 sw_g0, 0fFF7FFFFF, sw_g0, sw_p;\n"
        "setp.lt.ftz.f32 sw_p, %6, %5;\n"
        "selp.f32 sw_g1, %6, %5, sw_p;\n"
        "setp.lt.ftz.f32 sw_p, %5, 0fFF7FFFFF;\n"
        "selp.f32 sw_g1, 0fFF7FFFFF, sw_g1, sw_p;\n"
        "mov.b32 sw_r0, sw_u0;\n"
        "mov.b32 sw_r1, sw_u1;\n"
        "mov.b64 sw_up, {sw_r0, sw_r1};\n"
        "mov.b32 sw_r0, sw_g0;\n"
        "mov.b32 sw_r1, sw_g1;\n"
        "mov.b64 sw_gate, {sw_r0, sw_r1};\n"
        "mov.b64 sw_one, 4575657222473777152;\n"
        "mov.b64 sw_zero, 0;\n"
        "mov.b64 sw_l2e, 4591606991553079867;\n"
        "mov.b64 sw_neg, -4647714812233515008;\n"
        "fma.rn.f32x2 sw_f, sw_up, sw_one, sw_zero;\n"
        "mul.rn.f32x2 sw_a, sw_gate, sw_l2e;\n"
        "mul.rn.f32x2 sw_a, sw_neg, sw_a;\n"
        "mov.b64 {sw_r0, sw_r1}, sw_a;\n"
        "mov.b32 sw_x0, sw_r0;\n"
        "mov.b32 sw_x1, sw_r1;\n"
        "ex2.approx.ftz.f32 sw_x0, sw_x0;\n"
        "ex2.approx.ftz.f32 sw_x1, sw_x1;\n"
        "mov.b32 sw_r0, sw_x0;\n"
        "mov.b32 sw_r1, sw_x1;\n"
        "mov.b64 sw_d, {sw_r0, sw_r1};\n"
        "add.f32x2 sw_d, sw_one, sw_d;\n"
        "mov.b64 {sw_r0, sw_r1}, sw_d;\n"
        "mov.b32 sw_x0, sw_r0;\n"
        "mov.b32 sw_x1, sw_r1;\n"
        "rcp.approx.ftz.f32 sw_x0, sw_x0;\n"
        "rcp.approx.ftz.f32 sw_x1, sw_x1;\n"
        "mov.b32 sw_r0, sw_x0;\n"
        "mov.b32 sw_r1, sw_x1;\n"
        "mov.b64 sw_d, {sw_r0, sw_r1};\n"
        "mul.rn.f32x2 sw_d, sw_gate, sw_d;\n"
        "mul.rn.f32x2 sw_d, sw_f, sw_d;\n"
        "mov.b64 {sw_r0, sw_r1}, sw_d;\n"
        "mov.b32 %0, sw_r0;\n"
        "mov.b32 %1, sw_r1;\n"
        "}\n"
        : "=f"(h0), "=f"(h1)
        : "f"(up0), "f"(up1), "f"(gate0), "f"(gate1), "f"(limit), "f"(neg_limit));
}
DEV float abs_max_x2(float lo, float hi) {
    float r;
    asm("{\n.reg .f32 am_a, am_b;\nabs.ftz.f32 am_a, %1;\nabs.ftz.f32 am_b, %2;\n"
        "max.ftz.f32 am_a, am_a, am_b;\nabs.ftz.f32 am_a, am_a;\n"
        "max.ftz.f32 %0, am_a, 0f00000000;\n}"
        : "=f"(r)
        : "f"(lo), "f"(hi));
    return r;
}
DEV void scale_bits_x2(float mine0, float partner0, float mine1, float partner1, uint32_t& b0, uint32_t& b1) {
    asm("{\n"
        ".reg .pred sc_p;\n"
        ".reg .f32 sc_a, sc_b;\n"
        ".reg .b32 sc_r0, sc_r1;\n"
        ".reg .b64 sc_x, sc_c;\n"
        "max.ftz.f32 sc_a, %2, %3;\n"
        "setp.eq.ftz.f32 sc_p, sc_a, 0f00000000;\n"
        "selp.f32 sc_a, 0f43E00000, sc_a, sc_p;\n"
        "max.ftz.f32 sc_b, %4, %5;\n"
        "setp.eq.ftz.f32 sc_p, sc_b, 0f00000000;\n"
        "selp.f32 sc_b, 0f43E00000, sc_b, sc_p;\n"
        "mov.b32 sc_r0, sc_a;\n"
        "mov.b32 sc_r1, sc_b;\n"
        "mov.b64 sc_x, {sc_r0, sc_r1};\n"
        "mov.b64 sc_c, 4256545022072211749;\n"
        "mul.rn.f32x2 sc_x, sc_x, sc_c;\n"
        "mov.b64 {%0, %1}, sc_x;\n"
        "}"
        : "=r"(b0), "=r"(b1)
        : "f"(mine0), "f"(partner0), "f"(mine1), "f"(partner1));
}
DEV uint16_t quantize_e4m3x2(float lo, float hi, uint32_t inverse_bits) {
    uint16_t r;
    asm("{\n.reg .f32 qz_i, qz_a, qz_b;\nmov.b32 qz_i, %3;\n"
        "mul.ftz.f32 qz_a, %1, qz_i;\nmul.ftz.f32 qz_b, %2, qz_i;\n"
        "cvt.rn.satfinite.e4m3x2.f32 %0, qz_b, qz_a;\n}"
        : "=h"(r)
        : "f"(lo), "f"(hi), "r"(inverse_bits));
    return r;
}
DEV void ue8m0_code(uint32_t bits, uint32_t& code, uint32_t& inverse) {
    const uint32_t exponent = (bits >> 23) & 255u, mantissa = bits & 0x7FFFFFu;
    code = exponent;
    if (mantissa != 0u && exponent != 254u) {
        if (exponent != 0u) code = exponent + 1u;
        else if (mantissa > 0x400000u) code = 1u;
    }
    if (exponent == 255u) code = 255u;
    inverse = (254u - code) << 23;
    if (code == 254u) inverse = 0x400000u;
}
DEV int unshuffled_row(int row) { return (row & -32) | ((row & 7) << 2) | ((row >> 3) & 3); }
DEV float fmax_ftz(float a, float b) {
    float r;
    asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

template <bool G1, int SL = SLICE, int PD = PADDED>
DEV void sgm_body(const SgmParams& p) {
    static_assert(PD % 128 == 0 && (2 * SL) % 128 == 0 && SL + 64 == PD, "GEMM1 zeroes exactly 64 padding columns");
    constexpr int CPS = G1 ? 2 : (PD == PADDED_TP8 ? PD / 128 : 1);
    constexpr int STAGES = G1 ? 6 : (PD == PADDED_TP8 ? 4 : 10);
    constexpr int K = G1 ? HID : PD;
    constexpr int KT = K / (128 * CPS);
    constexpr int M_TILES = G1 ? 2 * SL / 128 : HID / 128;
    constexpr int ATOMS_PER_MTILE = K / 128;
    constexpr int O_SFA = CPS * A_CHUNK, O_B = O_SFA + CPS * SF_ATOM, O_SFB = O_B + CPS * B_CHUNK;
    constexpr int STAGE = O_SFB + CPS * SF_ATOM;
    static_assert(STAGE % 1024 == 0, "stage alignment (128B swizzle atoms)");
    constexpr uint32_t T_SF = 16, T_SF_STAGE = 8 * CPS;
    static_assert(T_SF + STAGES * T_SF_STAGE <= 128, "TMEM columns");
    extern __shared__ __align__(1024) uint8_t sgm_smem[];
    uint8_t* smem = (uint8_t*)(((uintptr_t)sgm_smem + 1023) & ~(uintptr_t)1023);
    __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES], accf[2], acce[2];
    __shared__ uint32_t tbase;
    __shared__ float exch[2 * 4 * TILE];
    const int tid = threadIdx.x, lane = tid & 31, warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
    const int bid = blockIdx.x, grid = gridDim.x;
    if (warp == 0 && lane == 0) {
        prefetch_tmap(p.tm_a);
        prefetch_tmap(p.tm_b);
        if (p.has_sh) prefetch_tmap(p.tm_sh);
        for (int s = 0; s < STAGES; ++s) { mb_init_t(&full[s], 2); mb_init_t(&empty[s], 1); }
        for (int s = 0; s < 2; ++s) { mb_init_t(&accf[s], 1); mb_init_t(&acce[s], 128); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 2) {
        tmalloc(&tbase, 128);
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (G1) pdl_wait_();
    const int* rec = p.record;
    const int C = p.cta_words;
    const int T = rec[1];
    const int total = (T + p.has_sh) * M_TILES;
    const int count = bid < total ? (total - 1 - bid) / grid + 1 : 0;
    const uint32_t tm = tbase;

    if (warp == 0) {
        if (p.trigger_at_start) pdl_launch_();
        if (lane == 0) {
            int s = 0;
            uint32_t ph = 0;
            for (int i = 0; i < count; ++i) {
                const int item = bid + i * grid, n_tile = item / M_TILES, m_tile = item - n_tile * M_TILES;
                const bool sh = n_tile == T;
                const int expert = sh ? 0 : rec[4 + n_tile];
                const uint8_t* sfa = (sh ? p.sfa_sh : p.sfa + (size_t)expert * p.sfa_stride) + (size_t)m_tile * ATOMS_PER_MTILE * SF_ATOM;
                for (int kt = 0; kt < KT; ++kt) {
                    mb_wait_t(&empty[s], ph ^ 1);
                    uint8_t* st = smem + (size_t)s * STAGE;
                    mb_expect_t(&full[s], (uint32_t)(CPS * ((sh ? 16384 : 8192) + SF_ATOM)));
#pragma unroll
                    for (int c = 0; c < CPS; ++c) {
                        const int kc = kt * CPS + c;
                        if (sh) tma2d_ef_(st + c * A_CHUNK, p.tm_sh, kc * 128, m_tile * 128, &full[s]);
                        else tma3d_ef_(st + c * A_CHUNK, p.tm_a, kc * 128, m_tile * 128, expert, &full[s]);
                        bulk1d_ef(st + O_SFA + c * SF_ATOM, sfa + (size_t)kc * SF_ATOM, SF_ATOM, &full[s]);
                    }
                    if (++s == STAGES) { s = 0; ph ^= 1; }
                }
            }
            if (!p.trigger_at_start) pdl_launch_();
            for (int j = 0; j < STAGES; ++j) {
                mb_wait_t(&empty[s], ph ^ 1);
                if (++s == STAGES) { s = 0; ph ^= 1; }
            }
        }
    } else if (warp == 1) {
        if (!G1) pdl_wait_();
        __shared__ uint32_t xsc[G1 ? TILE * (HID / 128) : 1];
        if (G1) {
            constexpr int NW = TILE * (HID / 128) / 32;
            uint32_t v[NW];
#pragma unroll
            for (int j = 0; j < NW; ++j) {
                const int w = lane + 32 * j;
                v[j] = w / (HID / 128) < p.rows ? __ldg((const uint32_t*)p.sfb + w) : 0x7F7F7F7Fu;
            }
#pragma unroll
            for (int j = 0; j < NW; ++j) xsc[lane + 32 * j] = v[j];
            __syncwarp();
        }
        int s = 0;
        uint32_t ph = 0;
        for (int i = 0; i < count; ++i) {
            const int item = bid + i * grid, n_tile = item / M_TILES;
            const int b_tile = G1 ? 0 : n_tile;
            for (int kt = 0; kt < KT; ++kt) {
                mb_wait_t(&empty[s], ph ^ 1);
                uint8_t* st = smem + (size_t)s * STAGE;
                if (G1) {
                    if (lane < TILE * CPS) {
                        const int row = lane / CPS, atom = lane - row * CPS;
                        const uint32_t value = xsc[row * (HID / 128) + kt * CPS + atom];
                        *(uint32_t*)(st + O_SFB + atom * SF_ATOM + row * 16) = value;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                }
                if (lane == 0) {
                    mb_expect_t(&full[s], (uint32_t)(CPS * (B_CHUNK + (G1 ? 0 : SF_ATOM))));
#pragma unroll
                    for (int c = 0; c < CPS; ++c) {
                        const int kc = kt * CPS + c;
                        tma2d_(st + O_B + c * B_CHUNK, p.tm_b, kc * 128, b_tile * TILE, &full[s]);
                        if (!G1) bulk1d(st + O_SFB + c * SF_ATOM, p.sfb + (size_t)n_tile * (PD / 128) * SF_ATOM + (size_t)kc * SF_ATOM, SF_ATOM, &full[s]);
                    }
                }
                __syncwarp();
                if (++s == STAGES) { s = 0; ph ^= 1; }
            }
        }
        if (lane == 0)
            for (int j = 0; j < STAGES; ++j) {
                mb_wait_t(&empty[s], ph ^ 1);
                if (++s == STAGES) { s = 0; ph ^= 1; }
            }
    } else if (warp == 2) {
        int s = 0;
        uint32_t ph = 0;
        for (int i = 0; i < count; ++i) {
            const int a = i & 1;
            const uint32_t idesc = (bid + i * grid) / M_TILES == T ? idesc_e4m3(8, 0) : idesc_e2m1_e4m3(8, 0);
            if (i >= 2) mb_wait_t(&acce[a], ((i >> 1) - 1) & 1);
            tcafter();
            const uint32_t d = tm + 8 * a;
            for (int kt = 0; kt < KT; ++kt) {
                mb_wait_t(&full[s], ph);
                tcafter();
                if (elect_one()) {
                    uint8_t* st = smem + (size_t)s * STAGE;
                    const uint64_t a0 = sdesc_t(st, 16, 1024, 2), b0 = sdesc_t(st + O_B, 16, 1024, 2);
                    const uint64_t fa = sdesc_t(st + O_SFA, 512, 128, 0), fb = sdesc_t(st + O_SFB, 512, 128, 0);
                    const uint32_t tsf = tm + T_SF + T_SF_STAGE * s;
#pragma unroll
                    for (int c = 0; c < CPS; ++c) {
                        tccpsf(tsf + 4 * c, fa + (uint64_t)((c * SF_ATOM) >> 4));
                        tccpsf(tsf + 4 * CPS + 4 * c, fb + (uint64_t)((c * SF_ATOM) >> 4));
                    }
#pragma unroll
                    for (int c = 0; c < CPS; ++c)
#pragma unroll
                        for (int kc = 0; kc < 4; ++kc)
                            tcmma(d, a0 + (uint64_t)((c * A_CHUNK + kc * 32) >> 4), b0 + (uint64_t)((c * B_CHUNK + kc * 32) >> 4),
                                  idesc | ((uint32_t)kc << 29) | ((uint32_t)kc << 4), tsf + 4 * c, tsf + 4 * CPS + 4 * c, (kt | c | kc) != 0);
                    tccommit(&empty[s]);
                }
                __syncwarp();
                if (++s == STAGES) { s = 0; ph ^= 1; }
            }
            if (elect_one()) tccommit(&accf[a]);
            __syncwarp();
        }
    } else if (warp >= 4) {
        const int q = warp - 4;
        for (int i = 0; i < count; ++i) {
            const int item = bid + i * grid, n_tile = item / M_TILES, m_tile = item - n_tile * M_TILES;
            const int a = i & 1;
            const int row0 = n_tile * TILE;
            float limit = 0.f;
            int valid = 0, row_a = -1, row_b = -1;
            const int token = (lane & 3) * 2;
            const bool sh = n_tile == T;
            if (G1) {
                if (sh) {
                    limit = p.sh_clamp;
                    valid = p.m_rows;
                    if (token < valid) row_a = token;
                    if (token + 1 < valid) row_b = token + 1;
                } else {
                    limit = p.clamp[rec[4 + n_tile]];
                    valid = rec[4 + C + n_tile] - row0;
#pragma unroll
                    for (int r = 0; r < TILE; ++r)
                        if (r < valid) {
                            const int routed = rec[4 + 2 * C + row0 + r];
                            if (routed == token) row_a = r;
                            if (routed == token + 1) row_b = r;
                        }
                }
            } else {
                valid = sh ? row0 + p.m_rows : rec[4 + C + n_tile];
            }
            mb_wait_t(&accf[a], (i >> 1) & 1);
            tcafter();
            uint32_t r[8];
            if (G1) {
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                             : "r"(tm + 8 * a + ((uint32_t)(32 * q) << 16))
                             : "memory");
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                             : "r"(tm + 8 * a + ((uint32_t)(32 * q + 16) << 16))
                             : "memory");
            } else {
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                             : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                             : "r"(tm + 8 * a + ((uint32_t)(32 * q) << 16))
                             : "memory");
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            tcbefore();
            mb_arrive_(&acce[a]);
            float v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] = __uint_as_float(r[k]);
            if (!G1) {
                const int hrow = unshuffled_row(m_tile * 128 + 32 * q + lane);
                __nv_bfloat16* out = (__nv_bfloat16*)p.out;
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    if (row0 + k < valid) out[(size_t)(row0 + k) * HID + hrow] = __float2bfloat16_rn(v[k]);
                continue;
            }
            float lo_a, lo_b, hi_a, hi_b;
            clamped_swiglu_x2(v[0], v[1], v[2], v[3], limit, -limit, lo_a, lo_b);
            clamped_swiglu_x2(v[4], v[5], v[6], v[7], limit, -limit, hi_a, hi_b);
            float amax_a = abs_max_x2(lo_a, hi_a), amax_b = abs_max_x2(lo_b, hi_b);
#pragma unroll
            for (int sh = 4; sh > 1; --sh) {
                amax_a = fmax_ftz(amax_a, __shfl_xor_sync(0xffffffffu, amax_a, 1 << sh));
                amax_b = fmax_ftz(amax_b, __shfl_xor_sync(0xffffffffu, amax_b, 1 << sh));
            }
            const int base = (i % 2) * (4 * TILE);
            if (lane < 4) {
                exch[base + q * TILE + token] = amax_a;
                exch[base + q * TILE + token + 1] = amax_b;
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
            const int partner = base + (q ^ 1) * TILE + token;
            uint32_t bits_a, bits_b, code_a, inv_a, code_b, inv_b;
            scale_bits_x2(amax_a, exch[partner], amax_b, exch[partner + 1], bits_a, bits_b);
            ue8m0_code(bits_a, code_a, inv_a);
            ue8m0_code(bits_b, code_b, inv_b);
            constexpr int H_ROW = PD / 2;
            const int pair = q * 16 + (lane >> 2) * 2;
            const int column = (m_tile * 64 + pair) / 2;
            uint16_t* H = (uint16_t*)p.out;
            if (row_a >= 0) H[(size_t)(row0 + row_a) * H_ROW + column] = quantize_e4m3x2(lo_a, hi_a, inv_a);
            if (row_b >= 0) H[(size_t)(row0 + row_b) * H_ROW + column] = quantize_e4m3x2(lo_b, hi_b, inv_b);
            constexpr int TILE_BYTES = PD / 128 * 512;
            if ((q & 1) == 0 && lane < TILE) {
                const int block = m_tile * 2 + q / 2;
                const int row = lane < 4 ? row_a : row_b;
                const uint32_t code = lane < 4 ? code_a : code_b;
                if (row >= 0) p.hs[(size_t)n_tile * TILE_BYTES + (block / 4) * 512 + row * 16 + block % 4] = (uint8_t)code;
            }
            if (m_tile == M_TILES - 1) {
                const int thread = q * 32 + lane;
                int pad_row = thread / 16;
                if (pad_row < valid) {
                    const size_t pad = (size_t)(row0 + pad_row) * H_ROW + (SL + (thread % 16) * 4) / 2;
                    H[pad] = 0;
                    H[pad + 1] = 0;
                }
                if (thread < 2 * TILE) {
                    pad_row = thread / 2;
                    const int pad_block = SL / 32 + thread % 2;
                    if (pad_row < valid) p.hs[(size_t)n_tile * TILE_BYTES + (pad_block / 4) * 512 + pad_row * 16 + pad_block % 4] = 127;
                }
            }
        }
    }
    tcbefore();
    __syncthreads();
    tcafter();
    if (warp == 2) tmdealloc(tm, 128);
}
}

extern "C" __global__ void __launch_bounds__(256, 1) sgl_gemm1(const __grid_constant__ sglm::SgmParams p) { sglm::sgm_body<true>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) sgl_gemm2(const __grid_constant__ sglm::SgmParams p) { sglm::sgm_body<false>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) sgl_gemm1_tp8(const __grid_constant__ sglm::SgmParams p) { sglm::sgm_body<true, sglm::SLICE_TP8, sglm::PADDED_TP8>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) sgl_gemm2_tp8(const __grid_constant__ sglm::SgmParams p) { sglm::sgm_body<false, sglm::SLICE_TP8, sglm::PADDED_TP8>(p); }

namespace sglr {
constexpr uint32_t N = 384, K = 5120, NSPLIT = 3, BS = K * 2 / 32, NWARPS = BS / 32, WORDS = N / 32, TILE8 = 8;
constexpr unsigned long long kEmpty = ~0ull;

struct SgrParams {
    const __nv_bfloat16* x;
    long long ldx;
    const float* norm_part;
    int nslices;
    float eps;
    const __nv_bfloat16* norm_w;
    const __nv_bfloat16* w;
    const float* bias;
    unsigned long long* slots;
    unsigned* counter;
    float* out_w;
    int* out_i;
    int* permuted;
    int* record;
    uint8_t* xq;
    uint8_t* xs;
    float scale;
    int _pad;
};

DEV float fadd(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
DEV float fmul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
DEV bool fgt(float a, float b) { uint32_t r; asm("{\n\t.reg .pred p;\n\tsetp.gt.f32 p, %1, %2;\n\tselp.u32 %0, 1, 0, p;\n\t}" : "=r"(r) : "f"(a), "f"(b)); return r != 0; }
DEV bool feq(float a, float b) { uint32_t r; asm("{\n\t.reg .pred p;\n\tsetp.eq.f32 p, %1, %2;\n\tselp.u32 %0, 1, 0, p;\n\t}" : "=r"(r) : "f"(a), "f"(b)); return r != 0; }
DEV float div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
DEV float sqrtsoftplus(float s) {
    const float sp = fgt(s, 20.0f) ? s : log1pf(expf(s));
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(sp));
    return r;
}
DEV float fma_bf16(uint16_t a, uint16_t b, float acc) {
    float r;
    asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(r) : "h"(a), "h"(b), "f"(acc));
    return r;
}
DEV unsigned long long order_key(float key, uint32_t n) {
    uint32_t kb = __float_as_uint(key);
    kb = (kb == 0x80000000u) ? 0u : kb;
    const uint32_t ku = (kb & 0x80000000u) ? ~kb : (kb | 0x80000000u);
    return ((unsigned long long)ku << 32) | (unsigned long long)(0xffffffffu - n);
}
template <uint32_t V, uint32_t S, uint32_t kMask>
DEV void warp_sum_scatter(float (&a)[V], uint32_t (&id)[V], uint32_t lane, float* out) {
    if constexpr (kMask == 0) {
#pragma unroll
        for (uint32_t i = 0; i < S; ++i) out[id[i]] = a[i];
    } else {
        constexpr uint32_t Q = S / 2;
        const bool hi = (lane & kMask) != 0;
#pragma unroll
        for (uint32_t i = 0; i < Q; ++i) {
            const float send = hi ? a[i] : a[Q + i];
            const float keep = hi ? a[Q + i] : a[i];
            id[i] = hi ? id[Q + i] : id[i];
            a[i] = fadd(keep, __shfl_xor_sync(0xffffffffu, send, kMask, 32));
        }
        if constexpr (S % 2 == 1) {
            a[Q] = fadd(a[2 * Q], __shfl_xor_sync(0xffffffffu, a[2 * Q], kMask, 32));
            id[Q] = id[2 * Q];
        }
        warp_sum_scatter<V, S - Q, kMask / 2>(a, id, lane, out);
    }
}
DEV void ld32_ef(uint32_t (&r)[8], const void* p, uint64_t pol) {
    asm("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
        : "l"(p), "l"(pol));
}
DEV void ld32(uint32_t (&r)[8], const void* p) {
    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "l"(p));
}

template <uint32_t M, uint32_t TOPK, uint32_t NE = N, uint32_t NS = NSPLIT>
DEV void router_body(const SgrParams& p) {
    static_assert(M >= 1 && M <= 8 && TOPK >= 1 && TOPK <= 8 && NE % 32 == 0 && NE % NS == 0 && NE <= N, "sgl_router: at most 8 rows, top-k <= 8");
    constexpr uint32_t NW = NE / 32;
    constexpr uint32_t kSlots = M * TOPK, kTiles = kSlots, kTileWords = (kTiles + 3) / 4 * 4;
    constexpr uint32_t kRecordWords = 4 + 2 * kTileWords + TILE8 * kTiles;
    const uint32_t bx = blockIdx.x, tx = threadIdx.x, lane = tx & 31, warp_id = tx >> 5;
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    uint32_t wv[NS][8];
#pragma unroll
    for (uint32_t n = 0; n < NS; ++n) ld32_ef(wv[n], p.w + (size_t)(bx * NS + n) * K + tx * 16, pol);
    uint32_t gv[8];
    ld32(gv, p.norm_w + tx * 16);
    float my_bias = 0.f;
    if (tx < M * NS) my_bias = p.bias[bx * NS + tx % NS];
    __shared__ uint32_t s_present[NW];
    __shared__ uint32_t s_rows_of[NE];
    __shared__ int32_t s_ids[kSlots];
    __shared__ float s_rstd[8];
    __shared__ uint2 s_sel[NWARPS][TOPK];
    __shared__ float s_acc[NWARPS][M * NS];
    __shared__ uint32_t s_last;
    for (uint32_t i = tx; i < NE; i += BS) {
        s_rows_of[i] = 0;
        if (i < NW) s_present[i] = 0;
    }
    {
        uint32_t dep = 0;
#pragma unroll
        for (uint32_t n = 0; n < NS; ++n) dep ^= wv[n][0];
        asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.u32 q, %0, 0;\n\t@q griddepcontrol.wait;\n\t@!q griddepcontrol.wait;\n\t}" ::"r"(dep) : "memory");
    }
    uint32_t ticket = 0;
    if (tx == 0) asm volatile("atom.add.relaxed.gpu.global.u32 %0, [%1], 1;" : "=r"(ticket) : "l"(p.counter) : "memory");
    uint32_t xv[M][8];
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) ld32(xv[m], p.x + (size_t)m * p.ldx + tx * 16);
    if (warp_id < M) {
        const int ns = p.nslices;
        float v = 0.f;
        for (int sl = (int)lane; sl < ns; sl += 32) v = __ldcg(p.norm_part + ((size_t)warp_id * ns + sl) * 26 + 25);
        float ss = 0.f;
        for (int sl = 0; sl < ns; ++sl) ss += __shfl_sync(0xffffffffu, v, sl);
        if (lane == 0) s_rstd[warp_id] = rsqrtf(ss / (float)K + p.eps);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) {
        const float rs = s_rstd[m];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t xw = xv[m][i], gw = gv[i];
            const float lo = __uint_as_float(gw << 16) * (__uint_as_float(xw << 16) * rs);
            const float hi = __uint_as_float(gw & 0xffff0000u) * (__uint_as_float(xw & 0xffff0000u) * rs);
            xv[m][i] = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(lo)) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
        }
    }
    float part[M * NS];
    uint32_t part_id[M * NS];
#pragma unroll
    for (uint32_t m = 0; m < M; ++m)
#pragma unroll
        for (uint32_t n = 0; n < NS; ++n) {
            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc = fma_bf16((uint16_t)(xv[m][i] & 0xffff), (uint16_t)(wv[n][i] & 0xffff), acc);
                acc = fma_bf16((uint16_t)(xv[m][i] >> 16), (uint16_t)(wv[n][i] >> 16), acc);
            }
            part[m * NS + n] = acc;
            part_id[m * NS + n] = m * NS + n;
        }
    warp_sum_scatter<M * NS, M * NS, 16>(part, part_id, lane, s_acc[warp_id]);
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __syncthreads();
    if (tx < M * NS) {
        float acc = s_acc[0][tx];
#pragma unroll
        for (uint32_t i = 1; i < NWARPS; ++i) acc = fadd(acc, s_acc[i][tx]);
        const uint32_t m = tx / NS, n = tx % NS, idx = m * NE + bx * NS + n;
        const float act = sqrtsoftplus(acc);
        const float b = fadd(act, my_bias);
        const float key = feq(b, b) ? b : __int_as_float(0x7f800000);
        const unsigned long long slot = ((unsigned long long)__float_as_uint(act) << 32) | __float_as_uint(key);
        asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p.slots + idx), "l"(slot) : "memory");
    }
    {
        const uint32_t j = tx >> 1;
        const bool mine = j % gridDim.x == bx;
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) {
            float f[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) { f[2 * i] = __uint_as_float(xv[m][i] << 16); f[2 * i + 1] = __uint_as_float(xv[m][i] & 0xffff0000u); }
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) a = sglm::fmax_ftz(a, fabsf(f[i]));
            a = sglm::fmax_ftz(a, __shfl_xor_sync(0xffffffffu, a, 1));
            if (mine) {
                uint32_t bits, bits2, code, inv, c2, i2;
                sglm::scale_bits_x2(a, a, a, a, bits, bits2);
                sglm::ue8m0_code(bits, code, inv);
                (void)bits2; (void)c2; (void)i2;
                uint32_t q[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    q[i] = (uint32_t)sglm::quantize_e4m3x2(f[4 * i], f[4 * i + 1], inv) | ((uint32_t)sglm::quantize_e4m3x2(f[4 * i + 2], f[4 * i + 3], inv) << 16);
                *(uint4*)(p.xq + (size_t)m * K + tx * 16) = make_uint4(q[0], q[1], q[2], q[3]);
                if ((tx & 1) == 0) p.xs[(size_t)m * (K / 32) + j] = (uint8_t)code;
            }
        }
    }
    if (tx == 0) s_last = ticket == gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;
    if (tx == 0) asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" ::"l"(p.counter) : "memory");
    for (uint32_t w = tx; w < kRecordWords; w += BS) p.record[w] = 0;
    constexpr uint32_t kPer = NE / 32;
    const uint32_t m = warp_id;
    if (m < M) {
        unsigned long long sv[kPer];
        unsigned long long* row = p.slots + (size_t)m * NE;
#pragma unroll
        for (uint32_t j = 0; j < kPer; ++j) asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(sv[j]) : "l"(row + lane + j * 32) : "memory");
        bool wait;
        do {
            wait = false;
#pragma unroll
            for (uint32_t j = 0; j < kPer; ++j) {
                if ((uint32_t)sv[j] == (uint32_t)kEmpty) asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(sv[j]) : "l"(row + lane + j * 32) : "memory");
                wait |= (uint32_t)sv[j] == (uint32_t)kEmpty;
            }
        } while (wait);
#pragma unroll
        for (uint32_t j = 0; j < kPer; ++j) asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(row + lane + j * 32), "l"(kEmpty) : "memory");
        unsigned long long v[kPer], b0 = 0, b1 = 0;
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (uint32_t j = 0; j < kPer; ++j) v[j] = order_key(__uint_as_float((uint32_t)sv[j]), lane + j * 32);
        auto top2 = [&](unsigned long long cut) {
            b0 = b1 = 0;
#pragma unroll
            for (uint32_t j = 0; j < kPer; ++j) {
                const unsigned long long x = v[j] < cut ? v[j] : 0ull;
                const float xa = __uint_as_float((uint32_t)(sv[j] >> 32));
                if (x > b0) { b1 = b0; a1 = a0; b0 = x; a0 = xa; }
                else if (x > b1) { b1 = x; a1 = xa; }
            }
        };
        top2(~0ull);
#pragma unroll
        for (uint32_t k = 0; k < TOPK; ++k) {
            const uint32_t hi = (uint32_t)(b0 >> 32), lo = (uint32_t)b0;
            const uint32_t mh = __reduce_max_sync(0xffffffffu, hi);
            const uint32_t tie = __ballot_sync(0xffffffffu, hi == mh);
            bool won;
            if (__popc(tie) == 1) {
                won = (tie >> lane) & 1u;
            } else {
                const uint32_t ml = __reduce_max_sync(0xffffffffu, hi == mh ? lo : 0u);
                won = hi == mh && lo == ml;
            }
            if (won) {
                s_sel[warp_id][k] = make_uint2(0xffffffffu - lo, __float_as_uint(a0));
                const unsigned long long taken = b0;
                b0 = b1; a0 = a1; b1 = 0;
                if (b0 == 0) top2(taken);
            }
        }
        __syncwarp();
        float sel_w[TOPK];
        int32_t sel_i[TOPK];
#pragma unroll
        for (uint32_t k = 0; k < TOPK; ++k) { sel_i[k] = (int32_t)s_sel[warp_id][k].x; sel_w[k] = __uint_as_float(s_sel[warp_id][k].y); }
        float vv[8];
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) vv[k] = k < TOPK ? sel_w[k] : 0.f;
        const float routed_sum = fadd(fadd(fadd(vv[0], vv[2]), fadd(vv[1], vv[3])), fadd(fadd(vv[4], vv[6]), fadd(vv[5], vv[7])));
        const float norm = fgt(routed_sum, 0.f) ? routed_sum : 1.f;
        if (lane < TOPK) {
            float wk = 0.f;
            int32_t ik = 0;
#pragma unroll
            for (uint32_t k = 0; k < TOPK; ++k) if (lane == k) { wk = sel_w[k]; ik = sel_i[k]; }
            wk = fmul(div_full(wk, norm), p.scale);
            p.out_w[m * TOPK + lane] = wk;
            p.out_i[m * TOPK + lane] = ik;
            s_ids[m * TOPK + lane] = ik;
            if (ik >= 0 && ik < (int32_t)NE) {
                atomicOr(&s_present[ik >> 5], 1u << (ik & 31));
                atomicOr(&s_rows_of[ik], 1u << m);
            }
        }
    }
    __syncthreads();
    if (tx < kSlots) {
        const int32_t row = (int32_t)(tx / TOPK);
        int32_t expert = s_ids[tx];
        if (expert < 0 || expert >= (int32_t)NE) expert = -1;
        int32_t tiles = 0, tile = 0;
#pragma unroll
        for (int32_t word = 0; word < (int32_t)NW; ++word) {
            const int32_t bits = __popc(s_present[word]);
            tiles += bits;
            if (word < (expert >> 5)) tile += bits;
        }
        if (tx == 0) { p.record[0] = tiles * (int32_t)TILE8; p.record[1] = tiles; }
        if (expert < 0) {
            p.permuted[tx] = -1;
        } else {
            const uint32_t below = (1u << (expert & 31)) - 1u;
            tile += __popc(s_present[expert >> 5] & below);
            const uint32_t chosen_by = s_rows_of[expert];
            const int32_t count = __popc(chosen_by), rank = __popc(chosen_by & ((1u << row) - 1u)), first = tile * (int32_t)TILE8;
            p.permuted[tx] = first + rank;
            p.record[4 + 2 * kTileWords + first + rank] = row;
            if (rank == 0) {
                p.record[4 + tile] = expert;
                p.record[4 + kTileWords + tile] = count > (int32_t)TILE8 ? first + (int32_t)TILE8 : first + count;
            }
        }
    }
}
}

#define SGL_ROUTER_ENTRY(M) \
    extern "C" __global__ void __launch_bounds__(sglr::BS, 1) sgl_router_##M(const __grid_constant__ sglr::SgrParams p) { sglr::router_body<M, 6>(p); }
SGL_ROUTER_ENTRY(1)
SGL_ROUTER_ENTRY(2)
SGL_ROUTER_ENTRY(3)
SGL_ROUTER_ENTRY(4)
SGL_ROUTER_ENTRY(5)
SGL_ROUTER_ENTRY(6)
SGL_ROUTER_ENTRY(7)
SGL_ROUTER_ENTRY(8)
#define SGL_ROUTER_D_ENTRY(M) \
    extern "C" __global__ void __launch_bounds__(sglr::BS, 1) sgl_router_d_##M(const __grid_constant__ sglr::SgrParams p) { sglr::router_body<M, 3, 128, 1>(p); }
SGL_ROUTER_D_ENTRY(1)
SGL_ROUTER_D_ENTRY(2)
SGL_ROUTER_D_ENTRY(3)
SGL_ROUTER_D_ENTRY(4)
SGL_ROUTER_D_ENTRY(5)
SGL_ROUTER_D_ENTRY(6)
SGL_ROUTER_D_ENTRY(7)
SGL_ROUTER_D_ENTRY(8)
