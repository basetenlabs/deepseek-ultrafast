#include "tc.cuh"

constexpr int PNT = 128;
constexpr int TM_MAX_TILES = 1024;

struct MpParams {
    const int* ids;
    const float* w;
    int M, topk, n_exp;
    int* row_pair;
    float* row_w;
    int* pair_row;
    int* row_tile;
    int4* tiles;
    int* ntiles;
};

#ifndef TCMOE_RM
extern "C" __global__ void __launch_bounds__(1024) moe_perm(const __grid_constant__ MpParams p) {
    __shared__ int cnt[512], off[512], toff[512], fill[512];
    __shared__ int wsum[32], wtsum[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    pdl_wait();
    pdl_launch();
    for (int e = tid; e < 512; e += 1024) { cnt[e] = 0; fill[e] = 0; }
    __syncthreads();
    const int P = p.M * p.topk;
    for (int i = tid; i < P; i += 1024) atomicAdd(&cnt[p.ids[i]], 1);
    __syncthreads();
    if (warp == 0) {
        int rs = 0, ts = 0;
        for (int j = 0; j < 16; ++j) {
            const int e = lane * 16 + j;
            const int c = e < p.n_exp ? cnt[e] : 0;
            rs += c;
            ts += (c + PNT - 1) / PNT;
        }
        int ri = rs, ti = ts;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int a = __shfl_up_sync(0xffffffff, ri, o), b = __shfl_up_sync(0xffffffff, ti, o);
            if (lane >= o) { ri += a; ti += b; }
        }
        int r0 = ri - rs, t0 = ti - ts;
        for (int j = 0; j < 16; ++j) {
            const int e = lane * 16 + j;
            const int c = e < p.n_exp ? cnt[e] : 0;
            off[e] = r0;
            toff[e] = t0;
            r0 += c;
            t0 += (c + PNT - 1) / PNT;
        }
        if (lane == 31) *p.ntiles = ti;
    }
    __syncthreads();
    for (int e = tid; e < p.n_exp; e += 1024) {
        const int c = cnt[e];
        for (int j = 0; j * PNT < c; ++j) p.tiles[toff[e] + j] = make_int4(e, off[e] + j * PNT, min(PNT, c - j * PNT), 0);
    }
    for (int i = tid; i < P; i += 1024) {
        const int e = p.ids[i];
        const int k = atomicAdd(&fill[e], 1);
        const int r = off[e] + k;
        p.row_pair[r] = i;
        p.row_w[r] = p.w[i];
        p.pair_row[i] = r;
        p.row_tile[r] = (toff[e] + k / PNT) << 8 | (k % PNT);
    }
}
#endif

struct MqParams {
    const bf16* x;
    int ldx, M, K, topk;
    const bf16* pw;
    const float* norm_part;
    int nslices;
    float eps;
    const int* pair_row;
    const int* row_tile;
    uint8_t* xq;
    uint8_t* sf;
};

#ifdef TCMOE_RM
extern "C" __global__ void __launch_bounds__(256) moe_gather_quant_nat(const __grid_constant__ MqParams p) {
#else
extern "C" __global__ void __launch_bounds__(256) moe_gather_quant(const __grid_constant__ MqParams p) {
#endif
    pdl_wait();
    pdl_launch();
    const int nkb = p.K / 32;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.M * nkb) return;
    const int row = i / nkb, kb = i % nkb;
    const float rs = rsqrtf(norm_part_sum(p.norm_part, row, p.nslices) / p.K + p.eps);
    const bf16* xr = p.x + (size_t)row * p.ldx + kb * 32;
    float v[32];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint4 q = *(const uint4*)(xr + 8 * j);
        const uint4 wq = *(const uint4*)(p.pw + kb * 32 + 8 * j);
        const bf16* h = (const bf16*)&q;
        const bf16* hw = (const bf16*)&wq;
#pragma unroll
        for (int e = 0; e < 8; ++e) v[8 * j + e] = __bfloat162float(__float2bfloat16(__bfloat162float(hw[e]) * (__bfloat162float(h[e]) * rs)));
    }
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(v[j]));
    const float sc = round_scale_pow2(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
    const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
    uint16_t hw[16];
#ifdef TCMOE_RM
#pragma unroll
    for (int j = 0; j < 16; ++j) hw[j] = f32x2_to_e4m3x2(v[2 * j] * inv, v[2 * j + 1] * inv);
#else
#pragma unroll
    for (int j = 0; j < 16; ++j) hw[kpos32(2 * j) / 2] = f32x2_to_e4m3x2(v[2 * j] * inv, v[2 * j + 1] * inv);
#endif
    uint32_t w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) w[j] = (uint32_t)hw[2 * j] | ((uint32_t)hw[2 * j + 1] << 16);
    const uint8_t sb = (uint8_t)((__float_as_uint(sc) >> 23) & 0xff);
    for (int k = 0; k < p.topk; ++k) {
        const int r = p.pair_row[row * p.topk + k];
        uint4* o = (uint4*)(p.xq + (size_t)r * p.K + kb * 32);
        o[0] = make_uint4(w[0], w[1], w[2], w[3]);
        o[1] = make_uint4(w[4], w[5], w[6], w[7]);
        const int rt = p.row_tile[r], t = rt >> 8, n = rt & 255;
        p.sf[((size_t)t * (p.K / 128) + kb / 4) * 512 + (n % 32) * 16 + (n / 32) * 4 + kb % 4] = sb;
    }
}

enum { TM_SWIGLU = 1, TM_STORE = 0 };

struct TmParams {
    const uint8_t* w;
    unsigned long long w_stride;
    int K;
    int RT;
    const void* tb;
    const uint8_t* sfb;
    const int4* tiles;
    const int* ntiles;
    const int* row_pair;
    const float* row_w;
    uint8_t* aq;
    uint8_t* asf;
    int ldaq, akt;
    float limit;
    int nq;
    float* out;
    int ldo;
    int flags;
    unsigned long long sf_off;
#ifdef TCMOE_RM
    const void* wmap;
#endif
};
constexpr int TM_FLAG_SGL = 32;

constexpr int FP4_STEP_B = 2048 + 128;
constexpr int RAW_B = 2 * 4 * FP4_STEP_B;
#ifndef TM_RS
#define TM_RS 3
#define TM_WS 2
#define TM_BS 4
#endif
constexpr int RS = TM_RS, WS = TM_WS, BS = TM_BS;
constexpr int W_LBO = 2080, W_SF = 16 * W_LBO;
constexpr int W_BYTES = W_SF + 2 * 2048, B_BYTES = 16384;
constexpr int SFB_MAX = 40;
constexpr int SM_B = 0, SM_W = BS * B_BYTES, SM_RAW = SM_W + WS * W_BYTES, SM_SFB = SM_RAW + RS * RAW_B, SM_EPI = SM_SFB + SFB_MAX * 512;
#ifdef TCMOE_RM
constexpr int RMW_BYTES = 32768 + 1024 + 1024;
constexpr int RWS = 3;
static_assert(RWS * RMW_BYTES <= SM_SFB - SM_W, "rm weight stages");
constexpr int T_SFA = 256, T_SFB = T_SFA + 8 * RWS;
#else
constexpr int T_SFA = 256, T_SFB = T_SFA + 8 * WS;
#endif

template <int MODE>
DEV void tc_moe_body(const TmParams& p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* sm = align1k(smem_raw);
#ifdef TCMOE_RM
    __shared__ __align__(8) uint64_t raw_full[RS], raw_empty[RS], w_ready[RWS], w_empty[RWS], b_full[BS], b_empty[BS], acc_full[2], acc_empty[2], sfb_full, sfb_empty;
#else
    __shared__ __align__(8) uint64_t raw_full[RS], raw_empty[RS], w_ready[WS], w_empty[WS], b_full[BS], b_empty[BS], acc_full[2], acc_empty[2], sfb_full, sfb_empty;
#endif
    __shared__ uint32_t tbase;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    pdl_wait();
    pdl_launch();
    constexpr int CA = MODE == TM_SWIGLU ? 8 : 3, CB = 1;
    __shared__ int scum[TM_MAX_TILES + 1];
    __shared__ int wsum[16], s_i0, s_i1;
    {
        const int nt = *p.ntiles, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
        int c0 = 0, c1 = 0;
        if (2 * threadIdx.x < nt) c0 = (CA + CB * ((p.tiles[2 * threadIdx.x].z + 15) >> 4)) * p.RT;
        if (2 * threadIdx.x + 1 < nt) c1 = (CA + CB * ((p.tiles[2 * threadIdx.x + 1].z + 15) >> 4)) * p.RT;
        int v = c0 + c1;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int u = __shfl_up_sync(0xffffffff, v, o); if (lane >= o) v += u; }
        if (lane == 31) wsum[w] = v;
        __syncthreads();
        if (w == 0) {
            int x = lane < 16 ? wsum[lane] : 0;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) { const int u = __shfl_up_sync(0xffffffff, x, o); if (lane >= o) x += u; }
            if (lane < 16) wsum[lane] = x;
        }
        __syncthreads();
        const int ex = v - (c0 + c1) + (w > 0 ? wsum[w - 1] : 0);
        if (2 * threadIdx.x <= nt) scum[2 * threadIdx.x] = ex;
        if (2 * threadIdx.x + 1 <= nt) scum[2 * threadIdx.x + 1] = ex + c0;
        __syncthreads();
        if (threadIdx.x == 0) {
            const long long C = scum[nt];
            auto item_at = [&](long long x) -> int {
                if (x >= C) return nt * p.RT;
                int lo = 0, hi = nt - 1;
                while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (scum[mid] <= x) lo = mid; else hi = mid - 1; }
                const int cpi = scum[lo + 1] - scum[lo];
                const long long j = (x - scum[lo] + cpi / p.RT - 1) / (cpi / p.RT);
                return lo * p.RT + (int)min((long long)p.RT, j);
            };
            s_i0 = item_at(C * blockIdx.x / gridDim.x);
            s_i1 = item_at(C * (blockIdx.x + 1) / gridDim.x);
        }
        __syncthreads();
    }
    const int i0 = s_i0, i1 = s_i1;
    if (i0 >= i1) return;
    const int steps = p.K / 64, KT = (steps + 1) / 2, WT = (steps + 3) / 4;
    if (warp == 0) tmalloc(&tbase, 512);
    if (tid == 0) {
        for (int s = 0; s < RS; ++s) { mb_init_t(&raw_full[s], 1); mb_init_t(&raw_empty[s], 8); }
#ifdef TCMOE_RM
        for (int s = 0; s < RWS; ++s) { mb_init_t(&w_ready[s], 1); mb_init_t(&w_empty[s], 1); }
#else
        for (int s = 0; s < WS; ++s) { mb_init_t(&w_ready[s], 8); mb_init_t(&w_empty[s], 1); }
#endif
        for (int s = 0; s < BS; ++s) { mb_init_t(&b_full[s], 1); mb_init_t(&b_empty[s], 1); }
        for (int s = 0; s < 2; ++s) { mb_init_t(&acc_full[s], 1); mb_init_t(&acc_empty[s], 4); }
        mb_init_t(&sfb_full, 1);
        mb_init_t(&sfb_empty, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    tcbefore();
    __syncthreads();
    tcafter();
    const uint32_t tm = tbase;
    auto item = [&](int it, int& tile, int& rt, int4& ti) { tile = it / p.RT; rt = it % p.RT; ti = p.tiles[tile]; };
    if (warp == 0 && lane == 0) {
#ifdef TCMOE_RM
        int ws = 0, wph = 0, k = 0;
        for (int it = i0; it < i1; ++it) {
            int tile, rt; int4 ti; item(it, tile, rt, ti);
#ifdef TCMOE_NAT
            const size_t sf0 = p.sf_off ? (size_t)p.sf_off : (size_t)p.RT * 128 * (KT * 64);
#else
            const size_t sf0 = p.sf_off ? (size_t)p.sf_off : (size_t)p.RT * 128 * (p.K / 2);
#endif
            const uint8_t* sfa = p.w + (size_t)ti.x * p.w_stride + sf0 + (size_t)rt * KT * 512;
            for (int wt = 0; wt < WT; ++wt, ++k) {
                if (k >= RWS) mb_wait_sleep(&w_empty[ws], wph ^ 1, 64);
                uint8_t* st = sm + SM_W + ws * RMW_BYTES;
                const int nk = min(2, KT - 2 * wt);
                mb_expect_t(&w_ready[ws], nk * (8192 + 512));
                for (int kl = 0; kl < nk; ++kl) {
                    tma3d_ef(st + kl * 16384, p.wmap, (2 * wt + kl) * 128, rt * 128, ti.x, &w_ready[ws]);
                    bulk1d_ef(st + 32768 + kl * 512, sfa + (size_t)(2 * wt + kl) * 512, 512, &w_ready[ws]);
                }
                if (++ws == RWS) { ws = 0; wph ^= 1; }
            }
        }
#else
        int s = 0, ph = 0, k = 0;
        for (int it = i0; it < i1; ++it) {
            int tile, rt; int4 ti; item(it, tile, rt, ti);
            const uint8_t* we = p.w + (size_t)ti.x * p.w_stride;
            for (int wt = 0; wt < WT; ++wt, ++k) {
                if (k >= RS) mb_wait_sleep(&raw_empty[s], ph ^ 1, 64);
                const int nks = min(4, steps - 4 * wt);
                const int nqd = min(2, p.nq - rt * 2);
                mb_expect_t(&raw_full[s], nqd * nks * FP4_STEP_B);
                for (int qd = 0; qd < nqd; ++qd)
                    bulk1d(sm + SM_RAW + s * RAW_B + qd * 4 * FP4_STEP_B, we + ((size_t)(rt * 2 + qd) * steps + 4 * wt) * FP4_STEP_B, nks * FP4_STEP_B, &raw_full[s]);
                if (++s == RS) { s = 0; ph ^= 1; }
            }
        }
#endif
    } else if (warp == 2 && lane == 0) {
        int s = 0, ph = 0, sl = 0, prev = -1, nsf = 0;
        for (int it = i0; it < i1; ++it) {
            int tile, rt; int4 ti; item(it, tile, rt, ti);
            if (tile != prev) {
                if (nsf > 0) mb_wait_sleep(&sfb_empty, (nsf - 1) & 1, 64);
                mb_expect_t(&sfb_full, KT * 512);
                bulk1d(sm + SM_SFB, p.sfb + (size_t)tile * KT * 512, KT * 512, &sfb_full);
                prev = tile;
                ++nsf;
            }
            const int nb = (ti.z + 15) >> 4, kpb = 8 / nb;
            for (int kt0 = 0; kt0 < KT; kt0 += kpb, ++sl) {
                const int nk = min(kpb, KT - kt0);
                if (sl >= BS) mb_wait_sleep(&b_empty[s], ph ^ 1, 64);
                uint8_t* st = sm + SM_B + s * B_BYTES;
                mb_expect_t(&b_full[s], nk * nb * 2048);
                const void* tbm = (const uint8_t*)p.tb + (nb - 1) * 128;
                for (int j = 0; j < nk; ++j) tma2d(st + j * nb * 2048, tbm, (kt0 + j) * 128, ti.y, &b_full[s]);
                if (++s == BS) { s = 0; ph ^= 1; }
            }
        }
    } else if (warp == 1) {
        const uint32_t tmu = uni(tm);
        const uint64_t a0 = uni64(sdesc_t(sm + SM_W, W_LBO, 128, 0)), b0 = uni64(sdesc_t(sm + SM_B, 16, 1024, 2));
        const uint64_t sfa0 = uni64(sdesc_t(sm + SM_W + W_SF, 2048, 128, 0)), sfb0 = uni64(sdesc_t(sm + SM_SFB, 512, 128, 0));
#ifdef TCMOE_RM
        const uint64_t ar0 = uni64(sdesc_t(sm + SM_W, 16, 1024, 2)), sfr0 = uni64(sdesc_t(sm + SM_W + 32768, 512, 128, 0));
#endif
        int ws = 0, wph = 0, bs = 0, bph = 0, ai = 0, prev = -1, nsf = 0;
        for (int it = i0; it < i1; ++it, ++ai) {
            int tile, rt; int4 ti; item(it, tile, rt, ti);
            const int cnt = (int)uni((uint32_t)ti.z);
            const int nmma = (cnt + 15) & ~15, ab = ai & 1, nb = nmma >> 4, kpb = 8 / nb;
            const uint32_t id0 = idesc_e2m1_e4m3(nmma, 0);
            if (ai >= 2) mb_wait_t(&acc_empty[ab], ((ai >> 1) - 1) & 1);
            tcafter();
            if (tile != prev) {
                mb_wait_t(&sfb_full, nsf & 1);
                tcafter();
                if (elect_one()) {
                    for (int kt = 0; kt < KT; ++kt) tccpsf(tmu + T_SFB + 4 * kt, sfb0 + (uint64_t)((kt * 512) >> 4));
                    tccommit(&sfb_empty);
                }
                __syncwarp();
                prev = tile;
                ++nsf;
            }
            const uint32_t d = tmu + ab * 128;
            int j = 0;
            for (int wt = 0; wt < WT; ++wt) {
                mb_wait_t(&w_ready[ws], wph);
                tcafter();
                const uint32_t t_sfa = tmu + T_SFA + 8 * ws;
                for (int kl = 0, jj = j, bb = bs, bp = bph; kl < 2; ++kl) {
                    const int kt = 2 * wt + kl;
                    if (kt >= KT) break;
                    if (jj == 0) { mb_wait_t(&b_full[bb], bp); tcafter(); }
                    if (++jj == kpb || kt == KT - 1) { jj = 0; if (++bb == BS) { bb = 0; bp ^= 1; } }
                }
                const bool leader = elect_one();
#ifdef TCMOE_RM
                if (leader) {
                    tccpsf(t_sfa, sfr0 + (uint64_t)((ws * RMW_BYTES) >> 4));
                    tccpsf(t_sfa + 4, sfr0 + (uint64_t)((ws * RMW_BYTES + 512) >> 4));
                }
#else
                if (leader) tccpsf_128x256(t_sfa, sfa0 + (uint64_t)((ws * W_BYTES) >> 4));
#endif
#pragma unroll
                for (int kl = 0; kl < 2; ++kl) {
                    const int kt = 2 * wt + kl;
                    if (kt >= KT) break;
#ifdef TCMOE_RM
                    const uint64_t aw = ar0 + (uint64_t)((ws * RMW_BYTES + kl * 16384) >> 4);
#else
                    const uint64_t aw = a0 + (uint64_t)((ws * W_BYTES + kl * 8 * W_LBO) >> 4);
#endif
                    const uint64_t bw = b0 + (uint64_t)((bs * B_BYTES + j * nb * 2048) >> 4);
                    const bool full = 2 * kt + 1 < steps;
                    if (leader) {
#pragma unroll
                        for (int kc = 0; kc < 4; ++kc) {
                            if (kc >= 2 && !full) break;
#ifdef TCMOE_RM
                            tcmma(d, aw + (uint64_t)((kc * 32) >> 4), bw + (uint64_t)(kc * 2), id0 | ((uint32_t)kc << 29) | ((uint32_t)kc << 4), t_sfa + 4 * kl, tmu + T_SFB + 4 * kt, (kt | kc) != 0);
#else
                            tcmma(d, aw + (uint64_t)((kc * 2 * W_LBO) >> 4), bw + (uint64_t)(kc * 2), id0 | ((uint32_t)kc << 29) | ((uint32_t)kc << 4), t_sfa + 4 * kl, tmu + T_SFB + 4 * kt, (kt | kc) != 0);
#endif
                        }
                    }
                    if (++j == kpb || kt == KT - 1) {
                        if (leader) tccommit(&b_empty[bs]);
                        j = 0;
                        if (++bs == BS) { bs = 0; bph ^= 1; }
                    }
                }
                if (leader) tccommit(&w_empty[ws]);
                __syncwarp();
#ifdef TCMOE_RM
                if (++ws == RWS) { ws = 0; wph ^= 1; }
#else
                if (++ws == WS) { ws = 0; wph ^= 1; }
#endif
            }
            if (elect_one()) tccommit(&acc_full[ab]);
            __syncwarp();
        }
#ifndef TCMOE_RM
    } else if (warp >= 4 && warp < 12) {
        const int tw = warp - 4, g = lane >> 2, t = lane & 3, q4 = tw & 3, qd = tw >> 2;
        const int h = t & 1, u = t >> 1;
        const int si = tw * 32 + lane, sg = si & 7, sq4 = (si >> 3) & 3, sks = (si >> 5) & 3, sqd = si >> 7;
        const int sr0 = sqd * 64 + sq4 * 16 + sg, sr1 = sr0 + 8;
        int rs = 0, rph = 0, ws = 0, wph = 0, k = 0;
        for (int it = i0; it < i1; ++it) {
            for (int wt = 0; wt < WT; ++wt, ++k) {
                mb_wait_t(&raw_full[rs], rph);
                if (k >= WS) mb_wait_t(&w_empty[ws], wph ^ 1);
                const uint8_t* raw = sm + SM_RAW + rs * RAW_B;
                uint8_t* st = sm + SM_W + ws * W_BYTES;
                const int nks = min(4, steps - 4 * wt);
                uint4 Wk[4];
#pragma unroll
                for (int ks = 0; ks < 4; ++ks)
                    if (ks < nks) Wk[ks] = *(const uint4*)(raw + (qd * 4 + ks) * FP4_STEP_B + q4 * 512 + lane * 16);
                const uint32_t sv = sks < nks ? *(const uint32_t*)(raw + (sqd * 4 + sks) * FP4_STEP_B + 2048 + sq4 * 32 + sg * 4) : 0u;
#pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    if (ks >= nks) break;
                    const uint4 W = Wk[ks];
                    const uint32_t O0 = h ? W.z : W.x, O1 = h ? W.w : W.y;
                    const uint32_t P0 = __shfl_xor_sync(0xffffffff, h ? W.x : W.z, 1);
                    const uint32_t P1 = __shfl_xor_sync(0xffffffff, h ? W.y : W.w, 1);
                    const uint32_t m0 = __byte_perm(O0, O1, 0x5410), m1 = __byte_perm(O0, O1, 0x7632);
                    const uint32_t q0 = __byte_perm(P0, P1, 0x5410), q1 = __byte_perm(P0, P1, 0x7632);
                    const int r0 = qd * 64 + q4 * 16 + g, c = ks * 4 + h * 2 + u;
                    if (!(p.flags & 1)) {
                        *(uint2*)(st + c * W_LBO + r0 * 16) = h ? make_uint2(q0, m0) : make_uint2(m0, q0);
                        *(uint2*)(st + c * W_LBO + (r0 + 8) * 16) = h ? make_uint2(q1, m1) : make_uint2(m1, q1);
                    }
                }
                if (sks < nks && !(p.flags & 4)) {
                    uint8_t* img = st + W_SF + (sks >> 1) * 2048;
#pragma unroll
                    for (int rep = 0; rep < 4; ++rep) {
                        *(uint16_t*)(img + rep * 512 + (sr0 % 32) * 16 + (sr0 / 32) * 4 + 2 * (sks & 1)) = (uint16_t)(sv & 0xffff);
                        *(uint16_t*)(img + rep * 512 + (sr1 % 32) * 16 + (sr1 / 32) * 4 + 2 * (sks & 1)) = (uint16_t)(sv >> 16);
                    }
                }
                if (!(p.flags & 2)) tcfence_proxy();
                __syncwarp();
                if (lane == 0) {
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&raw_empty[rs])) : "memory");
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32a(&w_ready[ws])) : "memory");
                }
                if (++rs == RS) { rs = 0; rph ^= 1; }
                if (++ws == WS) { ws = 0; wph ^= 1; }
            }
        }
#endif
    } else if (warp >= 12) {
        const int q = warp & 3;
        float* epi = (float*)(sm + SM_EPI);
        int ai = 0;
        for (int it = i0; it < i1; ++it, ++ai) {
            int tile, rt; int4 ti; item(it, tile, rt, ti);
            const int row0 = ti.y, cnt = ti.z, nmma = (cnt + 15) & ~15, ab = ai & 1;
            mb_wait_sleep(&acc_full[ab], (ai / 2) & 1, 500);
            tcafter();
            const uint32_t d = tm + ab * 128 + ((uint32_t)(32 * q) << 16);
            if (MODE == TM_STORE) {
#ifdef TCMOE_RM
                const int r_ = rt * 128 + 32 * q + lane;
                const int dd = (p.flags & TM_FLAG_SGL) ? ((r_ & -32) | ((r_ & 7) << 2) | ((r_ >> 3) & 3)) : r_;
#else
                const int dd = rt * 128 + 32 * q + lane;
#endif
                for (int c = 0; c < nmma; c += 32) {
                    const int prl = c + lane < cnt ? p.row_pair[row0 + c + lane] : 0;
                    float v[32];
                    tcld32(d + c, v);
#pragma unroll
                    for (int jj = 0; jj < 32; ++jj) {
                        const int pr = __shfl_sync(0xffffffffu, prl, jj);
                        if (c + jj < cnt) {
                            if (p.flags & 16) ((bf16*)p.out)[(size_t)pr * p.ldo + dd] = __float2bfloat16(v[jj]);
                            else p.out[(size_t)pr * p.ldo + dd] = v[jj];
                        }
                    }
                }
            }
#ifdef TCMOE_RM
            else if (p.flags & TM_FLAG_SGL) {
                const float L = p.limit;
                const int s0 = lane & 7, s1 = lane >> 3, b = q >> 1;
                const bool isup = (s1 & 1) == 0;
                const int lcol = 16 * (q & 1) + 2 * s0 + (s1 >> 1);
                const int kb = rt * 2 + b;
                float* xm = epi;
                for (int c = 0; c < nmma; c += 32) {
                    const float rwl = c + lane < cnt ? p.row_w[row0 + c + lane] : 0.f;
                    float v[32];
                    tcld32(d + c, v);
                    float a[32];
#pragma unroll
                    for (int jj = 0; jj < 32; ++jj) {
                        const float gv = __shfl_down_sync(0xffffffffu, v[jj], 8);
                        float gte = __bfloat162float(__float2bfloat16(gv));
                        float up = __bfloat162float(__float2bfloat16(v[jj]));
                        if (L > 0.f) { up = fminf(fmaxf(up, -L), L); gte = fminf(gte, L); }
                        const float rw = __shfl_sync(0xffffffffu, rwl, jj);
                        a[jj] = isup ? __bfloat162float(__float2bfloat16(gte / (1.0f + expf(-gte)) * up * rw)) : 0.f;
                        const unsigned am = __reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(a[jj])));
                        if (lane == 0) xm[(b * 2 + (q & 1)) * 32 + jj] = __uint_as_float(am);
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    if (kb < p.nq) {
#pragma unroll
                        for (int jj = 0; jj < 32; ++jj) {
                            const int n = c + jj;
                            if (n >= cnt) break;
                            const float amax = fmaxf(xm[(b * 2) * 32 + jj], xm[(b * 2 + 1) * 32 + jj]);
                            const float sc = round_scale_pow2(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
                            const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
                            if (isup) {
#ifdef TCMOE_NAT
                                p.aq[(size_t)(row0 + n) * p.ldaq + kb * 32 + lcol] = (uint8_t)(f32x2_to_e4m3x2(a[jj] * inv, 0.f) & 0xff);
#else
                                p.aq[(size_t)(row0 + n) * p.ldaq + kb * 32 + kpos32(lcol)] = (uint8_t)(f32x2_to_e4m3x2(a[jj] * inv, 0.f) & 0xff);
#endif
                            }
                            if ((q & 1) == 0 && lane == 0) p.asf[((size_t)tile * p.akt + kb / 4) * 512 + (n % 32) * 16 + (n / 32) * 4 + kb % 4] = (uint8_t)((__float_as_uint(sc) >> 23) & 0xff);
                        }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }
            }
#endif
            else {
                const int b = q >> 1;
                const float L = p.limit;
                float* eb = epi + b * 32 * 33;
                for (int c = 0; c < nmma; c += 32) {
                    const float rwl = c + lane < cnt ? p.row_w[row0 + c + lane] : 0.f;
                    float v[32];
                    tcld32(d + c, v);
                    if (q & 1) {
#pragma unroll
                        for (int jj = 0; jj < 32; ++jj) eb[lane * 33 + jj] = v[jj];
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    if (!(q & 1) && rt * 2 + b < p.nq) {
                        const int kb = rt * 2 + b;
#pragma unroll
                        for (int jj = 0; jj < 32; ++jj) {
                            const int n = c + jj;
                            if (n >= cnt) break;
                            float gte = __bfloat162float(__float2bfloat16(v[jj]));
                            float up = __bfloat162float(__float2bfloat16(eb[lane * 33 + jj]));
                            if (L > 0.f) { up = fminf(fmaxf(up, -L), L); gte = fminf(gte, L); }
                            const float a = __bfloat162float(__float2bfloat16(gte / (1.0f + expf(-gte)) * up * __shfl_sync(0xffffffffu, rwl, jj)));
                            const float amax = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(fabsf(a))));
                            const float sc = round_scale_pow2(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
                            const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
#ifdef TCMOE_NAT
                            p.aq[(size_t)(row0 + n) * p.ldaq + kb * 32 + lane] = (uint8_t)(f32x2_to_e4m3x2(a * inv, 0.f) & 0xff);
#else
                            p.aq[(size_t)(row0 + n) * p.ldaq + kb * 32 + kpos32(lane)] = (uint8_t)(f32x2_to_e4m3x2(a * inv, 0.f) & 0xff);
#endif
                            if (lane == 0) p.asf[((size_t)tile * p.akt + kb / 4) * 512 + (n % 32) * 16 + (n / 32) * 4 + kb % 4] = (uint8_t)((__float_as_uint(sc) >> 23) & 0xff);
                        }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
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

#ifndef TCMOE_RM
extern "C" __global__ void __launch_bounds__(512, 1) tc_moe_w13(const __grid_constant__ TmParams p) { tc_moe_body<TM_SWIGLU>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) tc_moe_w2(const __grid_constant__ TmParams p) { tc_moe_body<TM_STORE>(p); }

#endif
