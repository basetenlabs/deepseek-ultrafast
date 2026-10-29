#include "common.cuh"

struct Meta {
    int pos;
    int m;
    int tokens[16];
};

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

DEV uint32_t fkey(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
#ifndef TOPK_OLD
#define TOPK_OLD 0
#endif
DEV int topk_n(const Meta* meta, const TopkParams& t, int m) {
#if TOPK_OLD
    (void)m;
    int n = meta->pos / t.ratio;
#else
    int n = (meta->pos + m + 1) / t.ratio;
#endif
    if (t.block > 1) n = (n + t.block - 1) / t.block;
    return n;
}
#if TOPK_OLD
__shared__ uint32_t* s_okeys;
#define TKC_OKEY(slot, key) (s_okeys[(size_t)m * t.k + (slot)] = (key))
#else
#define TKC_OKEY(slot, key) ((void)0)
#endif
DEV uint32_t cl_rank_a() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEV void cl_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
DEV uint32_t sm_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sm_addr(dst)),
                 "l"(src), "r"(bytes), "r"(sm_addr(bar))
                 : "memory");
}
DEV uint32_t dsm_ldu(const uint32_t* local, uint32_t rank) {
    return *(const volatile uint32_t*)__cluster_map_shared_rank((void*)local, rank);
}
DEV uint2 dsm_ld2(const uint32_t* local, uint32_t rank) {
    uint32_t a = sm_addr(local), r;
    uint2 v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    asm volatile("ld.shared::cluster.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(r) : "memory");
    return v;
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

constexpr int TKC_NB = 4096;
constexpr uint32_t TKC_FINITE_BIN = 8;

template <int C, int T>
struct TkcShared {
    __align__(16) uint32_t hist[2][TKC_NB];
    __align__(16) uint32_t coarse[2][64];
    uint32_t tot[TKC_NB / C], rsum[C], xg[C], xe[C], wg[T / 32], we[T / 32];
    uint32_t s_sum, s_bin, s_need, s_cnt, s_some;
    int scid[T];
    __align__(8) uint64_t lbar;
};

DEV bool warp_pick64(uint2 v, uint32_t need, int& bin, uint32_t& rank, uint32_t& cnt) {
    const int lane = threadIdx.x & 31;
    const uint32_t p = v.x + v.y;
    uint32_t suf = p;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t u = __shfl_down_sync(0xffffffffu, suf, o);
        if (lane + o < 32) suf += u;
    }
    const uint32_t above = suf - p;
    const bool hi = above < need && need <= above + v.y;
    const bool lo = above + v.y < need && need <= above + p;
    const uint32_t ball = __ballot_sync(0xffffffffu, hi || lo);
    if (!ball) return false;
    const int src = __ffs(ball) - 1;
    bin = __shfl_sync(0xffffffffu, hi ? 2 * lane + 1 : 2 * lane, src);
    rank = __shfl_sync(0xffffffffu, hi ? need - above : need - above - v.y, src);
    cnt = __shfl_sync(0xffffffffu, hi ? v.y : v.x, src);
    return true;
}

template <int C, int T>
DEV bool tkc_resolve2(TkcShared<C, T>& s, int nb, uint32_t need, int r) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t* h = s.hist[r & 1];
    uint32_t* co = s.coarse[r & 1];
    const int FB = nb / 64;
    __syncthreads();
    {
        const int bpt = nb >= T ? nb / T : 1, nthr = nb / bpt, L = FB / bpt;
        if (warp * 32 < nthr) {
            uint32_t p = 0;
            for (int j = 0; j < bpt; ++j) p += h[tid * bpt + j];
            for (int o = L / 2; o >= 1; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if ((lane & (L - 1)) == 0) co[tid * bpt / FB] = p;
        }
    }
    cl_sync();
    if (warp == 0) {
        uint2 us[C];
#pragma unroll
        for (int q = 0; q < C; ++q) us[q] = dsm_ld2(co + 2 * lane, q);
        uint2 v = {0u, 0u};
#pragma unroll
        for (int q = 0; q < C; ++q) {
            v.x += us[q].x;
            v.y += us[q].y;
        }
        int sb = 0;
        uint32_t rk = 0, cnt = 0;
        const bool ok = warp_pick64(v, need, sb, rk, cnt);
        if (ok) {
            uint2 w = {0u, 0u};
            if (2 * lane < FB) {
                uint2 uw[C];
#pragma unroll
                for (int q = 0; q < C; ++q) uw[q] = dsm_ld2(h + sb * FB + 2 * lane, q);
#pragma unroll
                for (int q = 0; q < C; ++q) {
                    w.x += uw[q].x;
                    w.y += uw[q].y;
                }
            }
            int fb = 0;
            uint32_t rk2 = 0, cnt2 = 0;
            warp_pick64(w, rk, fb, rk2, cnt2);
            if (lane == 0) {
                s.s_bin = (uint32_t)(sb * FB + fb);
                s.s_need = rk2;
                s.s_cnt = cnt2;
            }
        }
        if (lane == 0) s.s_some = ok;
    } else {
        uint32_t* hn = s.hist[(r + 1) & 1];
        for (int i = tid - 32; i < TKC_NB; i += T - 32) hn[i] = 0;
    }
    __syncthreads();
    return s.s_some != 0;
}

template <int C, int T, typename Pos>
DEV void tkc_emit(TkcShared<C, T>& s, uint32_t c, const TopkParams& t, int m, const float* x, const uint32_t* tkeys,
                  bool cached, int ns, uint32_t pre, uint32_t pmask, uint32_t need, uint32_t eqall, Pos pos) {
    constexpr int NW = T / 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
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
    if (lane == 0) { s.wg[warp] = ng; s.we[warp] = ne; }
    __syncthreads();
    if (warp == 0) {
        const uint32_t g = lane < NW ? s.wg[lane] : 0u, e = lane < NW ? s.we[lane] : 0u;
        uint32_t gi = g, ei = e;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t u = __shfl_up_sync(0xffffffffu, gi, o), v = __shfl_up_sync(0xffffffffu, ei, o);
            if (lane >= o) { gi += u; ei += v; }
        }
        if (lane < NW) { s.wg[lane] = gi - g; s.we[lane] = ei - e; }
        const uint32_t gt = __shfl_sync(0xffffffffu, gi, 31), et = __shfl_sync(0xffffffffu, ei, 31);
        if (lane < C) { dsm_stu(&s.xg[c], lane, gt); dsm_stu(&s.xe[c], lane, et); }
    }
    cl_sync();
    const uint32_t nq = eqall ? 0xffffffffu : need;
    uint32_t gb = 0, eb0 = 0, gall = 0, eall = 0;
#pragma unroll
    for (int r = 0; r < C; ++r) {
        const uint32_t g = s.xg[r], e = s.xe[r];
        if (r < (int)c) { gb += g; eb0 += e; }
        gall += g;
        eall += e;
    }
    const uint32_t total = gall + min(nq, eall);
    uint32_t eqb = eb0 + s.we[warp];
    uint32_t off = gb + s.wg[warp] + min(nq, eqb);
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
                    const int id = pos(written + 4 * lane + j);
                    out[base + __popc(take & ((1u << j) - 1))] = id;
                    TKC_OKEY(base + __popc(take & ((1u << j) - 1)), j == 0 ? k.x : j == 1 ? k.y : j == 2 ? k.z : k.w);
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
            const int id = pos(i);
            out[off + __popc(tb & lt)] = id;
            TKC_OKEY(off + __popc(tb & lt), key);
            if (mk) mk[id] = 1;
        }
        off += __popc(tb);
        eqb += __popc(eb);
    }
    if (c == C - 1)
        for (int i = (int)total + tid; i < t.k; i += T) out[i] = -1;
}

template <int C, int T>
DEV void tkc_passes(TkcShared<C, T>& s, uint32_t c, const TopkParams& t, const float* x, const uint32_t* tkeys, bool cached,
                    int ns, int pass, int& r, uint32_t& pre, uint32_t& pmask, uint32_t& need, uint32_t& eqall) {
    const int tid = threadIdx.x;
    for (;; ++pass) {
        const int shift = pass == 0 ? 20 : pass == 1 ? 8 : 0, nb = pass < 2 ? TKC_NB : 256;
        uint32_t* hist = s.hist[(r & 1)];
        if (pass > 0) {
            int loaded = 0;
            if (!cached) {
                for (; loaded + 4 * T <= ns; loaded += 4 * T) {
                    const uint4 k = topk_load4(x, loaded + 4 * tid, t.drop_neg_inf);
                    if (k.x && (k.x & pmask) == pre) atomicAdd(&hist[(k.x >> shift) & (nb - 1)], 1u);
                    if (k.y && (k.y & pmask) == pre) atomicAdd(&hist[(k.y >> shift) & (nb - 1)], 1u);
                    if (k.z && (k.z & pmask) == pre) atomicAdd(&hist[(k.z >> shift) & (nb - 1)], 1u);
                    if (k.w && (k.w & pmask) == pre) atomicAdd(&hist[(k.w >> shift) & (nb - 1)], 1u);
                }
            }
            for (int i = loaded + tid; i < ns; i += T) {
                const uint32_t key = topk_cl_key(x, tkeys, i, cached, t.drop_neg_inf);
                if (key && (key & pmask) == pre) atomicAdd(&hist[(key >> shift) & (nb - 1)], 1u);
            }
        }
        bool some;
        some = tkc_resolve2<C, T>(s, nb, need, r);
        ++r;
        if (!some) { eqall = 1; break; }
        pre |= s.s_bin << shift;
        pmask |= (uint32_t)(nb - 1) << shift;
        need = s.s_need;
        const uint32_t cnt = s.s_cnt;
        if (need == cnt || shift == 0) { eqall = need == cnt; break; }
    }
}

template <int C, int T>
DEV void tkc_full(TkcShared<C, T>& s, uint32_t c, const Meta* meta, const TopkParams& t, int m, uint32_t* tkeys,
                  uint32_t cache_bytes, int r, bool reuse) {
    const int tid = threadIdx.x;
    const int n = topk_n(meta, t, m);
    const int S = ((n + C - 1) / C + 3) & ~3;
    const int b0 = (int)c * S, b1 = min(n, b0 + S), ns = max(0, b1 - b0);
    const bool cached = (uint32_t)S * 4 <= cache_bytes;
    const float* x = t.x + (size_t)m * t.ld + b0;
    uint32_t* hist = s.hist[(r & 1)];
    if (reuse) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
    }
    if (cached && tid == 0) {
#if TOPK_OLD
        const uint32_t bytes = (uint32_t)(ns & ~3) * 4;
#else
        const uint32_t bytes = (uint32_t)((ns + 3) & ~3) * 4;
#endif
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sm_addr(&s.lbar)), "r"(bytes) : "memory");
        if (bytes) bulk_g2s(tkeys, x, bytes, &s.lbar);
    }
    if (t.mask) {
        uint8_t* mk = t.mask + (size_t)m * t.ld_mask;
        const int e1 = c == C - 1 ? t.ld_mask : min(t.ld_mask, b0 + S);
        for (int i = b0 + tid; i < e1; i += T) mk[i] = 0;
    }
    const uint32_t ninf = fkey(-INFINITY);
    if (cached) {
        asm volatile("{ .reg .pred P; W_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W_%=; }" ::"r"(sm_addr(&s.lbar)) : "memory");
#if TOPK_OLD
        for (int i = (ns & ~3) + tid; i < ns; i += T) tkeys[i] = __float_as_uint(x[i]);
        __syncthreads();
#endif
    }
    int loaded = 0;
    if (!cached) {
        for (; loaded + 4 * T <= ns; loaded += 4 * T) {
            const uint4 k = topk_load4(x, loaded + 4 * tid, t.drop_neg_inf);
            topk_first_bin(hist, k.x);
            topk_first_bin(hist, k.y);
            topk_first_bin(hist, k.z);
            topk_first_bin(hist, k.w);
        }
    }
    for (int i0 = loaded; i0 < ns; i0 += T) {
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
    tkc_passes<C, T>(s, c, t, x, tkeys, cached, ns, 0, r, pre, pmask, need, eqall);
    tkc_emit<C, T>(s, c, t, m, x, tkeys, cached, ns, pre, pmask, need, eqall, [&](int i) { return b0 + i; });
}

template <int C, int T>
DEV void topk_clc_body(const Meta* meta, const TopkParams& t, const int* cand_ids, int ncand) {
    extern __shared__ __align__(16) uint32_t tkeys[];
    __shared__ TkcShared<C, T> s;
    const uint32_t c = cl_rank_a();
    const int m = blockIdx.y, tid = threadIdx.x;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    for (int i = tid; i < TKC_NB; i += T) s.hist[0][i] = 0;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sm_addr(&s.lbar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    pdl_wait();
    pdl_launch();
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (m >= meta->m) return;
    uint32_t cache_bytes;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(cache_bytes));
    const int nc = ncand * 8;
    const int S = ((nc + C - 1) / C + 7) & ~7;
    const int b0 = (int)c * S, ns = max(0, min(nc, b0 + S) - b0);
    bool cand = ncand > 0 && !t.mask && (uint32_t)S * 4 <= cache_bytes && S / 8 <= T;
    int r = 0;
    if (cand) {
        const int nlive = topk_n(meta, t, m);
        const float* xr = t.x + (size_t)m * t.ld;
        const int* cid = cand_ids + (size_t)m * ncand + b0 / 8;
        const uint32_t ninf = fkey(-INFINITY);
        for (int i0 = 0; i0 < ns; i0 += 2 * T) {
            const int i = i0 + 2 * tid;
            uint32_t k0 = 0, k1 = 0;
            if (i < ns) {
                const int blk = cid[i >> 3];
                if ((i & 7) == 0) s.scid[i >> 3] = blk;
                if (blk >= 0) {
                    const int p = blk * 8 + (i & 7);
                    const float2 v = *(const float2*)(xr + p);
                    k0 = p < nlive ? fkey(v.x) : 0u;
                    k1 = p + 1 < nlive ? fkey(v.y) : 0u;
                    if (t.drop_neg_inf) {
                        if (k0 == ninf) k0 = 0;
                        if (k1 == ninf) k1 = 0;
                    }
                }
                *(uint2*)&tkeys[i] = make_uint2(k0, k1);
            }
            topk_first_bin(s.hist[0], k0);
            topk_first_bin(s.hist[0], k1);
        }
        bool some;
        some = tkc_resolve2<C, T>(s, TKC_NB, (uint32_t)t.k, r);
        ++r;
        cand = some && s.s_bin >= TKC_FINITE_BIN;
        if (cand) {
            uint32_t pre = s.s_bin << 20, pmask = (uint32_t)(TKC_NB - 1) << 20, need = s.s_need, eqall = 0;
            if (need == s.s_cnt) {
                eqall = 1;
            } else {
                tkc_passes<C, T>(s, c, t, nullptr, tkeys, true, ns, 1, r, pre, pmask, need, eqall);
            }
            const int* scid = s.scid;
            tkc_emit<C, T>(s, c, t, m, nullptr, tkeys, true, ns, pre, pmask, need, eqall,
                           [&](int i) { return scid[i >> 3] * 8 + (i & 7); });
            return;
        }
    }
    tkc_full<C, T>(s, c, meta, t, m, tkeys, cache_bytes, r, ncand > 0);
}

extern "C" __global__ void __launch_bounds__(1024, 1) topk_clc2_8_1024(const Meta* meta, const __grid_constant__ TopkParams t, const int* cand_ids, int ncand) {
    topk_clc_body<8, 1024>(meta, t, cand_ids, ncand);
}

#if TOPK_OLD
extern "C" __global__ void __launch_bounds__(1024, 1) topk_old_8_1024(const Meta* meta, const __grid_constant__ TopkParams t, uint32_t* okeys) {
    if (threadIdx.x == 0) s_okeys = okeys;
    topk_clc_body<8, 1024>(meta, t, nullptr, 0);
}
#endif
