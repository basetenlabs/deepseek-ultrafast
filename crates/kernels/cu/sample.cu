#include "common.cuh"
#include "sample.cuh"

struct Meta {
    int pos;
    int m;
    int tokens[16];
};

constexpr int SNC = 63;
constexpr int PKW = 8 + 3 * SNC + 1;
constexpr int PKB = 4 + SNC + 32;
constexpr int RRW = 8 + 3 * 64;
constexpr int SROWS = 8;
constexpr int SO_PK = 0;

struct SampIo {
    const float* logits;
    int vloc, vpad, offset;
    int qn;
    const int* drafts;
    const float* vq;
    const float2* qstat;
    SampParams* sp;
    uint32_t* recv[8];
    int tp, rank;
    unsigned* seq;
    float* rowres;
    int* out;
    int* next;
    int force_fb;
    int last_only;
    const float* spart;
    int nquads;
    int so_s2, so_fr;
    float leps;
    int so_bv;
    int block;
};
constexpr int BV_ITEMS = 3;

DEV unsigned okey(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
DEV float okey_inv(unsigned k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }
#if SAMP_CL
DEV uint32_t samp_crank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEV uint32_t samp_csize() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r)); return r; }
DEV void samp_cl_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
DEV uint32_t samp_mapa(const void* local, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"((uint32_t)__cvta_generic_to_shared(local)), "r"(rank));
    return r;
}
DEV void samp_dst(const void* local, uint32_t rank, float v) { asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(samp_mapa(local, rank)), "f"(v) : "memory"); }
DEV float samp_dld(const void* local, uint32_t rank) {
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(samp_mapa(local, rank)) : "memory");
    return v;
}
#endif
DEV void st_v4(uint32_t* a, uint4 v) { asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
DEV uint4 ld_v4(const uint32_t* a) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    return v;
}
DEV float wmax(float v) {
    __syncwarp();
    return warp_max(v);
}
DEV float wsum(float v) {
    __syncwarp();
    return warp_sum(v);
}
DEV void lse_add(float& m, float& s, float m2, float s2) {
    const float mx = fmaxf(m, m2), mr = mx == -INFINITY ? 0.f : mx;
    s = s * expf(m - mr) + s2 * expf(m2 - mr);
    m = mx;
}
#ifndef SAMP_NT
#define SAMP_NT 1024
#endif
constexpr int NWARP = SAMP_NT / 32;
constexpr int SAMP_MAXQ = 1024;
DEV float blk_max(float v, float* sh) {
    v = wmax(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = -INFINITY;
    for (int i = 0; i < SAMP_NT / 32; ++i) r = fmaxf(r, sh[i]);
    return r;
}
DEV float blk_sum(float v, float* sh) {
    v = wsum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < SAMP_NT / 32; ++i) r += sh[i];
    return r;
}
constexpr int FB_LV = 3, FB_TOK = 2048;
constexpr int FB_LVW = SROWS * 8 * 256 * 4;
DEV unsigned long long fb_mass(float v, float gm) { return __float2ull_rz(expf(v - gm) * 17179869184.f); }

__noinline__ DEV void samp_radix_fallback(const SampIo& p, const float* z, float inv_t, float P,
                                         float lse, float gm, unsigned tag, int jr,
                                         float* cut, float* logz) {
    const int tid = threadIdx.x;
    __shared__ double mass[256], global_mass[256];
    __shared__ unsigned lo[256], hi[256], global_lo[256], global_hi[256];
    __shared__ unsigned prefix, mask;
    __shared__ double above, need;
    __shared__ int done;
    if (tid == 0) { prefix = mask = 0; above = 0.; done = 0; }
    __syncthreads();
    static_assert(4 * 256 * 8 <= FB_TOK * 4, "radix exchange must fit boundary-token scratch");
    const size_t base = (size_t)p.so_fr + (size_t)FB_LV * FB_LVW;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) { mass[tid] = 0.; lo[tid] = 0xffffffffu; hi[tid] = 0; }
        __syncthreads();
        for (int i = tid; i < ((p.vloc + 31) & ~31); i += SAMP_NT) {
            const float v = i < p.vloc ? z[i] * inv_t : -INFINITY;
            const unsigned key = okey(v == 0.f ? 0.f : v);
            const int bin = (key & mask) == prefix && v != -INFINITY ? (key >> shift) & 255 : 256;
            const unsigned group = __match_any_sync(0xffffffffu, bin);
            if (bin < 256) {
                const double w = (double)expf(v - gm);
                double sum = 0.;
                if (group == 0xffffffffu) {
                    sum = w;
#pragma unroll
                    for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(group, sum, delta);
                } else {
                    for (unsigned lanes = group; lanes; lanes &= lanes - 1) sum += __shfl_sync(group, w, __ffs(lanes) - 1);
                }
                const unsigned low = __reduce_min_sync(group, key), high = __reduce_max_sync(group, key);
                if ((tid & 31) == __ffs(group) - 1) {
                    atomicAdd(&mass[bin], sum);
                    atomicMin(&lo[bin], low);
                    atomicMax(&hi[bin], high);
                }
            }
        }
        __syncthreads();
        for (int i = tid; i < p.tp * 256; i += SAMP_NT) {
            const int peer = i / 256, bin = i % 256;
            const unsigned long long bits = (unsigned long long)__double_as_longlong(mass[bin]);
            uint32_t* dst = p.recv[peer] + base + (((size_t)pass * SROWS * 8 + jr * 8 + p.rank) * 256 + bin) * 8;
            st_v4(dst, make_uint4((unsigned)bits, tag, (unsigned)(bits >> 32), tag));
            st_v4(dst + 4, make_uint4(lo[bin], tag, hi[bin], tag));
        }
        if (tid < 256) {
            double sum = 0.;
            unsigned low = 0xffffffffu, high = 0;
            for (int src = 0; src < p.tp; ++src) {
                const uint32_t* addr = p.recv[p.rank] + base + (((size_t)pass * SROWS * 8 + jr * 8 + src) * 256 + tid) * 8;
                uint4 u, k;
                do { u = ld_v4(addr); } while (u.y != tag || u.w != tag);
                do { k = ld_v4(addr + 4); } while (k.y != tag || k.w != tag);
                sum += __longlong_as_double((unsigned long long)u.x | ((unsigned long long)u.z << 32));
                low = min(low, k.x); high = max(high, k.z);
            }
            global_mass[tid] = sum; global_lo[tid] = low; global_hi[tid] = high;
        }
        __syncthreads();
        if (tid == 0) {
            if (pass == 0) {
                double total = 0.;
                for (int bin = 255; bin >= 0; --bin) total += global_mass[bin];
                need = (double)P * total;
            }
            int bin = 255;
            for (; bin > 0; --bin) {
                if (above + global_mass[bin] >= need) break;
                above += global_mass[bin];
            }
            if (global_lo[bin] == global_hi[bin]) {
                *cut = okey_inv(global_lo[bin]);
                *logz = (float)(log(above + global_mass[bin]) + (double)gm - (double)lse);
                done = 1;
            } else {
                prefix |= (unsigned)bin << shift;
                mask |= 255u << shift;
            }
        }
        __syncthreads();
        if (done) return;
    }
}

__noinline__ DEV void samp_fallback(const SampIo& p, const float* z, float inv_t, float P, float lse, float gm_in, float cmax, unsigned tag, int jr,
                                    unsigned* hc, float* fbv, int* fbi, float* s_cut, float* s_logz, int s_k) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float gm = gm_in;
    __shared__ unsigned long long hm[256], gsm[256];
    __shared__ unsigned gsc[256];
    __shared__ unsigned long long s_tot, s_need, s_cab;
    __shared__ int s_bl[FB_LV], s_cnt, s_c, s_unresolved;
    __shared__ int s_src[9];
    if (tid == 0) {
        s_tot = (unsigned long long)((double)expf(lse - gm) * 17179869184.0);
        s_need = (unsigned long long)((double)P * (double)s_tot);
        s_cab = 0;
        s_c = 0;
        s_unresolved = 0;
    }
    auto lvbin = [&](float v, int l) -> int {
        float d = (gm - v) * 16.f;
        if (!(d < 256.f)) return -1;
        int b = (int)d;
        for (int q = 0; q < l; ++q) {
            if (b != s_bl[q]) return -1;
            d = (d - (float)b) * 256.f;
            b = min(255, (int)d);
        }
        return b;
    };
    int lv = 0;
    for (;; ++lv) {
        if (tid < 256) { hm[tid] = 0; hc[tid] = 0; gsm[tid] = 0; gsc[tid] = 0; }
        __syncthreads();
        for (int i = tid; i < p.vloc; i += SAMP_NT) {
            const float v = z[i] * inv_t;
            const int b = lvbin(v, lv);
            if (b >= 0) {
                atomicAdd(&hm[b], fb_mass(v, gm));
                atomicAdd(&hc[b], 1u);
            }
        }
        __syncthreads();
        for (int w = tid; w < 256 * p.tp; w += SAMP_NT) {
            const int q = w >> 8, b = w & 255;
            st_v4(p.recv[q] + p.so_fr + (size_t)lv * FB_LVW + ((size_t)(jr * 8 + p.rank) * 256 + b) * 4,
                  make_uint4((uint32_t)hm[b], tag, (uint32_t)(hm[b] >> 32) | (hc[b] << 17), tag));
        }
        for (int w = tid; w < 256 * p.tp; w += SAMP_NT) {
            const int src = w >> 8, b = w & 255;
            const uint32_t* a = p.recv[p.rank] + p.so_fr + (size_t)lv * FB_LVW + ((size_t)(jr * 8 + src) * 256 + b) * 4;
            uint4 u;
            do { u = ld_v4(a); } while (u.y != tag || u.w != tag);
            atomicAdd(&gsm[b], (unsigned long long)u.x | ((unsigned long long)(u.z & 0x1ffffu) << 32));
            atomicAdd(&gsc[b], u.z >> 17);
        }
        __syncthreads();
        if (warp == 0) {
            __syncwarp();
            unsigned long long loc[8], tot = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) { loc[i] = gsm[lane * 8 + i]; tot += loc[i]; }
            unsigned long long pre = tot;
#pragma unroll
            for (int o2 = 1; o2 < 32; o2 <<= 1) {
                const unsigned long long v = __shfl_up_sync(0xffffffffu, pre, o2);
                if (lane >= o2) pre += v;
            }
            pre -= tot;
            const unsigned long long cab = s_cab, need = s_need;
            const bool hit = cab + pre + tot >= need;
            int bb = lane * 8 + 7;
            unsigned long long ab = pre + tot - loc[7];
            if (hit) {
                unsigned long long c = cab + pre;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (c + loc[i] >= need) { bb = lane * 8 + i; ab = c - cab; break; }
                    c += loc[i];
                }
            }
            const unsigned bal = __ballot_sync(0xffffffffu, hit);
            const int src = bal ? __ffs(bal) - 1 : 31;
            bb = __shfl_sync(0xffffffffu, bb, src);
            ab = __shfl_sync(0xffffffffu, ab, src);
            if (lane == 0) {
                s_bl[lv] = bb;
                s_cab = cab + ab;
                s_cnt = (int)gsc[bb];
                s_unresolved = bal == 0;
            }
        }
        __syncthreads();
        if (s_unresolved || s_cnt <= FB_TOK || lv == FB_LV - 1) break;
    }
    if (s_unresolved || s_cnt > FB_TOK) {
        samp_radix_fallback(p, z, inv_t, P, lse, gm, tag, jr, s_cut, s_logz);
        return;
    }
    const size_t tb0 = (size_t)p.so_fr + (size_t)FB_LV * FB_LVW;
    for (int i = tid; i < p.vloc; i += SAMP_NT) {
        const float v = z[i] * inv_t;
        if (lvbin(v, lv) == s_bl[lv]) {
            const int slot = atomicAdd(&s_c, 1);
            if (slot < FB_TOK)
                for (int q = 0; q < p.tp; ++q)
                    st_v4(p.recv[q] + tb0 + ((size_t)(jr * 8 + p.rank) * FB_TOK + slot) * 4, make_uint4(__float_as_uint(v), tag, (uint32_t)(i + p.offset), tag));
        }
    }
    if (tid == 0) {
        int c = 0;
        for (int r = 0; r < p.tp; ++r) {
            const uint4 u = ld_v4(p.recv[p.rank] + p.so_fr + (size_t)lv * FB_LVW + ((size_t)(jr * 8 + r) * 256 + s_bl[lv]) * 4);
            s_src[r] = c;
            c = min(FB_TOK, c + min(FB_TOK, (int)(u.z >> 17)));
        }
        s_src[p.tp] = c;
    }
    __syncthreads();
    float* bv = fbv;
    int* bi = fbi;
    const int c = s_src[p.tp];
    for (int r = 0; r < p.tp; ++r)
        for (int i = s_src[r] + tid; i < s_src[r + 1]; i += SAMP_NT) {
            const uint32_t* a = p.recv[p.rank] + tb0 + ((size_t)(jr * 8 + r) * FB_TOK + (i - s_src[r])) * 4;
            uint4 u;
            do { u = ld_v4(a); } while (u.y != tag || u.w != tag);
            bv[i] = __uint_as_float(u.x);
            bi[i] = (int)u.z;
        }
    int np2 = 1;
    while (np2 < c) np2 <<= 1;
    for (int i = c + tid; i < np2; i += SAMP_NT) { bv[i] = -INFINITY; bi[i] = 0x7fffffff; }
    __syncthreads();
    for (int k = 2; k <= np2; k <<= 1)
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
            for (int i = tid; i < np2; i += SAMP_NT) {
                const int l = i ^ jj;
                if (l > i) {
                    const bool desc = (i & k) == 0;
                    const float a = bv[i], b = bv[l];
                    const int ia = bi[i], ib = bi[l];
                    const bool a_first = a > b || (a == b && ia < ib);
                    if (desc != a_first) { bv[i] = b; bv[l] = a; bi[i] = ib; bi[l] = ia; }
                }
            }
            __syncthreads();
        }
    if (tid == 0) {
        unsigned long long cum = s_cab;
        float cut = bv[c > 0 ? c - 1 : 0];
        int k = 0;
        for (; k < c; ++k) {
            cum += fb_mass(bv[k], gm);
            if (cum >= s_need) { cut = bv[k]; break; }
        }
        for (++k; k < c && bv[k] == cut; ++k) cum += fb_mass(bv[k], gm);
        const float cut0 = (*s_cut), logz0 = (*s_logz);
        (*s_cut) = cut;
        (*s_logz) = logf((float)((double)cum / (double)s_tot));
        if (p.force_fb == 2 && s_k <= 64 && cut0 > cmax && cut0 != -INFINITY) {
            atomicAdd(&p.out[2], 1);
            if (cut0 != (*s_cut) || fabsf(logz0 - (*s_logz)) > 1e-4f) atomicAdd(&p.out[3], 1);
        }
    }
    __syncthreads();
}

struct RowDraw {
    float kr, kp, mass;
    int ir, ip;
};
#if SAMP_CL
DEV void samp_quads(const SampIo& p, const float* z, const float* vq, float inv_t, float cut, float lz, float lseq, float w, bool resid,
                    int qa, int qb, float (*qmass)[SAMP_MAXQ], uint32_t dst, bool remote) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int QB = 4;
    for (int q0 = qa + warp; q0 < qb; q0 += NWARP * QB) {
        float2 zz[QB], vv[QB];
#pragma unroll
        for (int b = 0; b < QB; ++b) {
            const int q = q0 + b * NWARP, i = q * 64 + 2 * lane;
            zz[b] = make_float2(-INFINITY, -INFINITY);
            vv[b] = make_float2(-INFINITY, -INFINITY);
            if (q < qb && i < p.vloc) {
                zz[b] = *(const float2*)(z + i);
                if (resid) vv[b] = *(const float2*)(vq + i);
            }
        }
#pragma unroll
        for (int b = 0; b < QB; ++b) {
            const int q = q0 + b * NWARP;
            if (q >= qb) break;
            float wr = 0.f, wp = 0.f;
            const float zs[2] = {zz[b].x, zz[b].y}, vs[2] = {vv[b].x, vv[b].y};
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const float s = zs[u] * inv_t;
                if (s < cut) continue;
                const float pp = expf(s - lz);
                wp += pp;
                if (resid) wr += fmaxf(w * pp - expf(vs[u] * inv_t - lseq), 0.f);
            }
            __syncwarp();
            wr = wsum(wr);
            wp = wsum(wp);
            if (lane == 0) {
                if (remote) { samp_dst(&qmass[0][q], dst, wr); samp_dst(&qmass[1][q], dst, wp); }
                else { qmass[0][q] = wr; qmass[1][q] = wp; }
            }
        }
    }
}

DEV int samp_qsplit(int nq, uint32_t S, uint32_t r) { return (int)(((long long)nq * r) / S); }
__shared__ float samp_cpar[6];

#endif
DEV RowDraw samp_row_draw(const SampIo& p, const SampParams& sp, const float* z, const float* vq, float inv_t,
#if SAMP_CL
                          float cut, float lse, float logz, float lseq, float w, bool resid, unsigned sub, bool mass, bool helper = false, bool clustered = false) {
#else
                          float cut, float lse, float logz, float lseq, float w, bool resid, unsigned sub, bool mass) {
#endif
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float shv[2][32];
    __shared__ int shi[2][32];
    __shared__ float qmass[2][SAMP_MAXQ];
    __shared__ int s_qbest[2];
    __shared__ float s_mass, shm[32];
#if SAMP_CL
    const int nq = (p.vloc + 63) / 64;
    const uint32_t S = samp_csize(), cr = samp_crank();
    if (helper) {
        samp_cl_sync();
        const bool act = samp_dld(&samp_cpar[0], 0) != 0.f;
        if (act) {
            const float c = samp_dld(&samp_cpar[1], 0), l = samp_dld(&samp_cpar[2], 0), lq = samp_dld(&samp_cpar[3], 0), ww = samp_dld(&samp_cpar[4], 0);
            const bool rs = samp_dld(&samp_cpar[5], 0) != 0.f;
            samp_quads(p, z, vq, inv_t, c, l, lq, ww, rs, samp_qsplit(nq, S, cr), samp_qsplit(nq, S, cr + 1), qmass, 0, true);
        }
        samp_cl_sync();
        return RowDraw{};
    }
#endif
    const float lz = lse + logz;
#if SAMP_CL
    if (clustered && S > 1) {
        if (tid == 0) { samp_cpar[0] = 1.f; samp_cpar[1] = cut; samp_cpar[2] = lz; samp_cpar[3] = lseq; samp_cpar[4] = w; samp_cpar[5] = resid ? 1.f : 0.f; }
        samp_cl_sync();
        samp_quads(p, z, vq, inv_t, cut, lz, lseq, w, resid, 0, samp_qsplit(nq, S, 1), qmass, 0, false);
        samp_cl_sync();
    } else {
        samp_quads(p, z, vq, inv_t, cut, lz, lseq, w, resid, 0, nq, qmass, 0, false);
        __syncthreads();
    }
#else
    const int nq = (p.vloc + 63) / 64;
    {
        constexpr int QB = 4;
        for (int q0 = warp; q0 < nq; q0 += NWARP * QB) {
            float2 zz[QB], vv[QB];
#pragma unroll
            for (int b = 0; b < QB; ++b) {
                const int q = q0 + b * NWARP, i = q * 64 + 2 * lane;
                zz[b] = make_float2(-INFINITY, -INFINITY);
                vv[b] = make_float2(-INFINITY, -INFINITY);
                if (q < nq && i < p.vloc) {
                    zz[b] = *(const float2*)(z + i);
                    if (resid) vv[b] = *(const float2*)(vq + i);
                }
            }
#pragma unroll
            for (int b = 0; b < QB; ++b) {
                const int q = q0 + b * NWARP;
                if (q >= nq) break;
                float wr = 0.f, wp = 0.f;
                const float zs[2] = {zz[b].x, zz[b].y}, vs[2] = {vv[b].x, vv[b].y};
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const float s = zs[u] * inv_t;
                    if (s < cut) continue;
                    const float pp = expf(s - lz);
                    wp += pp;
                    if (resid) wr += fmaxf(w * pp - expf(vs[u] * inv_t - lseq), 0.f);
                }
                __syncwarp();
                wr = wsum(wr);
                wp = wsum(wp);
                if (lane == 0) { qmass[0][q] = wr; qmass[1][q] = wp; }
            }
        }
    }
    __syncthreads();
#endif
    if (mass) {
        const float m = blk_sum(tid < nq ? qmass[0][tid] : 0.f, shm);
        if (tid == 0) s_mass = m;
    }
    {
        float br = -INFINITY, bp = -INFINITY;
        int ir = 0x7fffffff, ip = 0x7fffffff;
        for (int q = tid; q < nq; q += SAMP_NT) {
            const int gq = p.rank * SAMP_MAXQ + q;
            const float g = rng_gumbel(rng_bits(sp, RNG_FINAL, (unsigned)gq, sub + 1u));
            const float wr = qmass[0][q], wp = qmass[1][q];
            if (wp > 0.f) { const float k = logf(wp) + g; if (k > bp || (k == bp && gq < ip)) { bp = k; ip = gq; } }
            if (resid && wr > 0.f) { const float k = logf(wr) + g; if (k > br || (k == br && gq < ir)) { br = k; ir = gq; } }
        }
        __syncwarp();
        for (int o = 16; o > 0; o >>= 1) {
            const float a = __shfl_xor_sync(0xffffffffu, br, o), b = __shfl_xor_sync(0xffffffffu, bp, o);
            const int ia = __shfl_xor_sync(0xffffffffu, ir, o), ib = __shfl_xor_sync(0xffffffffu, ip, o);
            if (a > br || (a == br && ia < ir)) { br = a; ir = ia; }
            if (b > bp || (b == bp && ib < ip)) { bp = b; ip = ib; }
        }
        if (lane == 0) { shv[0][warp] = br; shi[0][warp] = ir; shv[1][warp] = bp; shi[1][warp] = ip; }
        __syncthreads();
        if (warp == 0) {
            __syncwarp();
            const bool on = lane < NWARP;
            br = on ? shv[0][lane] : -INFINITY; ir = on ? shi[0][lane] : 0x7fffffff;
            bp = on ? shv[1][lane] : -INFINITY; ip = on ? shi[1][lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1) {
                const float a = __shfl_xor_sync(0xffffffffu, br, o), b = __shfl_xor_sync(0xffffffffu, bp, o);
                const int ia = __shfl_xor_sync(0xffffffffu, ir, o), ib = __shfl_xor_sync(0xffffffffu, ip, o);
                if (a > br || (a == br && ia < ir)) { br = a; ir = ia; }
                if (b > bp || (b == bp && ib < ip)) { bp = b; ip = ib; }
            }
            if (lane == 0) { shv[0][0] = br; shv[1][0] = bp; s_qbest[0] = ir; s_qbest[1] = ip; }
        }
        __syncthreads();
    }
    if (warp < 4) {
        const int which = warp >> 1;
        const int gq = s_qbest[which];
        float k = -INFINITY;
        int gi = 0x7fffffff;
        if (gq != 0x7fffffff && (gq - p.rank * SAMP_MAXQ) * 64 + (warp & 1) * 32 + lane < p.vloc) {
            const int i = (gq - p.rank * SAMP_MAXQ) * 64 + (warp & 1) * 32 + lane;
            gi = i + p.offset;
            const float s = z[i] * inv_t;
            if (s >= cut) {
                const float lp = s - lz;
                const float g = rng_gumbel(rng_bits(sp, RNG_FINAL, (unsigned)gi, sub + 2u));
                if (which == 1) k = lp + g;
                else {
                    const float r = w * expf(lp) - expf(vq[i] * inv_t - lseq);
                    if (r > 0.f) k = logf(r) + g;
                }
            }
        }
        __syncwarp();
        for (int o = 16; o > 0; o >>= 1) {
            const float a = __shfl_xor_sync(0xffffffffu, k, o);
            const int ia = __shfl_xor_sync(0xffffffffu, gi, o);
            if (a > k || (a == k && ia < gi)) { k = a; gi = ia; }
        }
        if (lane == 0) { shv[0][4 + warp] = k; shi[0][4 + warp] = gi; }
    }
    __syncthreads();
    if (tid == 0) {
        const float r0 = shv[0][4], r1 = shv[0][5], p0 = shv[0][6], p1 = shv[0][7];
        const int i0 = shi[0][4], i1 = shi[0][5], j0 = shi[0][6], j1 = shi[0][7];
        shi[1][0] = (r0 > r1 || (r0 == r1 && i0 < i1)) ? i0 : i1;
        shi[1][1] = (p0 > p1 || (p0 == p1 && j0 < j1)) ? j0 : j1;
        if (fmaxf(r0, r1) == -INFINITY) { shv[0][0] = -INFINITY; shi[1][0] = 0x7fffffff; }
        if (fmaxf(p0, p1) == -INFINITY) { shv[1][0] = -INFINITY; shi[1][1] = 0x7fffffff; }
    }
    __syncthreads();
    return RowDraw{shv[0][0], shv[1][0], mass ? s_mass : 0.f, shi[1][0], shi[1][1]};
}
#if SAMP_CL

DEV void samp_cluster_idle() {
    if (threadIdx.x == 0) samp_cpar[0] = 0.f;
    samp_cl_sync();
    samp_cl_sync();
}
#endif

DEV unsigned samp_seq_next(unsigned seq) { return seq >= 0xFFFFFFFDu ? 0u : seq + 1u; }

DEV void samp_pick(const SampIo& p, int pos, int M, const SampParams& sp, unsigned seq) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int L = p.last_only ? 0 : M - 1;
    const float inv_t = 1.f / sp.temp;
    const unsigned tag = seq + 1, par = seq & 1;
    __shared__ int s_js;
    __shared__ float s_rr[8];
    if (warp == 0) {
        __syncwarp();
        float acc = 1.f;
        if (!p.last_only && lane < L) acc = __ldcg(p.rowres + lane * RRW);
        const unsigned rej = __ballot_sync(0xffffffffu, acc == 0.f);
        const int js0 = rej ? __ffs(rej) - 1 : L;
        const float* rr0 = p.rowres + (p.last_only ? 0 : js0) * RRW;
        if (lane < 8) s_rr[lane] = __ldcg(rr0 + lane);
        if (lane == 0) s_js = js0;
    }
    __syncthreads();
    int js = s_js;
    const bool resid = !p.last_only && js < L;
    const float* rr = p.rowres + (p.last_only ? 0 : js) * RRW;
    const float cut = s_rr[1], lse = s_rr[2], logz = s_rr[3], lseq = s_rr[4];
    if (p.last_only) js = M - 1;
    if (s_rr[5] != 0.f) {
        if (warp == 0) {
            __syncwarp();
            const int k = (int)s_rr[6];
            float br = -INFINITY, bp = -INFINITY;
            int ir = 0x7fffffff, ip = 0x7fffffff;
            for (int i = lane; i < k; i += 32) {
                const float s = __ldcg(rr + 8 + i), v = __ldcg(rr + 8 + 128 + i);
                const int gi = __float_as_int(__ldcg(rr + 8 + 64 + i));
                const float g = rng_gumbel(rng_bits(sp, RNG_FINAL, (unsigned)gi, 0u));
                const float pp = expf(s - lse - logz);
                const float kp = logf(pp) + g;
                if (kp > bp || (kp == bp && gi < ip)) { bp = kp; ip = gi; }
                if (resid) {
                    const float w = pp - expf(v - lseq);
                    if (w > 0.f) {
                        const float kr = logf(w) + g;
                        if (kr > br || (kr == br && gi < ir)) { br = kr; ir = gi; }
                    }
                }
            }
            __syncwarp();
            for (int o = 16; o > 0; o >>= 1) {
                const float a = __shfl_xor_sync(0xffffffffu, br, o), b = __shfl_xor_sync(0xffffffffu, bp, o);
                const int ia = __shfl_xor_sync(0xffffffffu, ir, o), ib = __shfl_xor_sync(0xffffffffu, ip, o);
                if (a > br || (a == br && ia < ir)) { br = a; ir = ia; }
                if (b > bp || (b == bp && ib < ip)) { bp = b; ip = ib; }
            }
            if (lane == 0) {
                const int tok = (resid && br != -INFINITY) ? ir : ip;
                p.out[0] = p.last_only ? 0 : js;
                p.out[1] = tok;
                p.next[js] = tok;
                *p.seq = samp_seq_next(seq);
                p.sp->step = sp.step + 1;
            }
        }
        return;
    }
    const float* z = p.logits + (size_t)js * p.vpad;
    const float* vq = p.vq + (size_t)js * p.vpad;
    const RowDraw d = samp_row_draw(p, sp, z, vq, inv_t, cut, lse, logz, lseq, 1.f, resid, 0u, false);
    if (warp == 0) {
        __syncwarp();
        float br = d.kr, bp = d.kp;
        int ir = d.ir, ip = d.ip;
        if (lane < p.tp) {
            uint32_t* a = p.recv[lane] + p.so_s2 + (size_t)(par * 8 + p.rank) * 8;
            st_v4(a, make_uint4(__float_as_uint(br), tag, (uint32_t)ir, tag));
            st_v4(a + 4, make_uint4(__float_as_uint(bp), tag, (uint32_t)ip, tag));
        }
        float vr = -INFINITY, vp = -INFINITY;
        int xr = 0x7fffffff, xp = 0x7fffffff;
        if (lane < p.tp) {
            const uint32_t* a = p.recv[p.rank] + p.so_s2 + (size_t)(par * 8 + lane) * 8;
            uint4 u, w;
            do { u = ld_v4(a); } while (u.y != tag || u.w != tag);
            do { w = ld_v4(a + 4); } while (w.y != tag || w.w != tag);
            vr = __uint_as_float(u.x); xr = (int)u.z; vp = __uint_as_float(w.x); xp = (int)w.z;
        }
        __syncwarp();
        float gr = -INFINITY, gp = -INFINITY;
        int tr = 0x7fffffff, tq = 0x7fffffff;
        for (int r = 0; r < p.tp; ++r) {
            const float a = __shfl_sync(0xffffffffu, vr, r), b = __shfl_sync(0xffffffffu, vp, r);
            const int ia = __shfl_sync(0xffffffffu, xr, r), ib = __shfl_sync(0xffffffffu, xp, r);
            if (a > gr || (a == gr && ia < tr)) { gr = a; tr = ia; }
            if (b > gp || (b == gp && ib < tq)) { gp = b; tq = ib; }
        }
        if (lane == 0) {
            const int tok = (resid && gr != -INFINITY) ? tr : tq;
            p.out[0] = p.last_only ? 0 : js;
            p.out[1] = tok;
            p.next[js] = tok;
            *p.seq = samp_seq_next(seq);
            p.sp->step = sp.step + 1;
        }
    }
}

DEV int ld_acquire_gpu(const int* a) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
    return v;
}

DEV void samp_block(const SampIo& p, int j, int M, const SampParams& sp, unsigned seq, float inv_t, const float* z,
                    float cut, float lse, float logz, float lseq, bool fast, int k, const float* o) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int L = M - 1;
    const unsigned tag = seq + 1, par = seq & 1;
    __shared__ float s_w[SROWS + 1];
    __shared__ float s_m[SROWS], s_kr[SROWS], s_kp[SROWS];
    __shared__ int s_ir[SROWS], s_ip[SROWS], s_last;
    if (warp == 0) {
        if (lane == 0)
            while (ld_acquire_gpu(p.out + 8) < M) __nanosleep(64);
        __syncwarp();
        const float lr = lane < L ? __ldcg(p.rowres + lane * RRW + 7) : 0.f;
        __syncwarp();
        float c = 0.f, mx = 0.f;
        for (int i = 0; i < L; ++i) {
            c += __shfl_sync(0xffffffffu, lr, i);
            mx = fmaxf(mx, c);
            if (lane == 0) s_w[i + 1] = expf(c - mx);
        }
        if (lane == 0) s_w[0] = 1.f;
    }
    __syncthreads();
    const float w = s_w[j];
    const bool resid = j < L;
    RowDraw d;
    if (fast) {
        __shared__ float s_fd[3];
        __shared__ int s_fi[2];
        if (warp == 0) {
            __syncwarp();
            float br = -INFINITY, bp = -INFINITY, m = 0.f;
            int ir = 0x7fffffff, ip = 0x7fffffff;
            for (int i = lane; i < k; i += 32) {
                const float s = __ldcg(o + 8 + i), v = __ldcg(o + 8 + 128 + i);
                const int gi = __float_as_int(__ldcg(o + 8 + 64 + i));
                const float g = rng_gumbel(rng_bits(sp, RNG_FINAL, (unsigned)gi, 8u * j));
                const float pp = expf(s - lse - logz);
                const float kp = logf(pp) + g;
                if (kp > bp || (kp == bp && gi < ip)) { bp = kp; ip = gi; }
                if (resid) {
                    const float r = w * pp - expf(v - lseq);
                    if (r > 0.f) {
                        m += r;
                        const float kr = logf(r) + g;
                        if (kr > br || (kr == br && gi < ir)) { br = kr; ir = gi; }
                    }
                }
            }
            __syncwarp();
            m = warp_sum(m);
            for (int o2 = 16; o2 > 0; o2 >>= 1) {
                const float a = __shfl_xor_sync(0xffffffffu, br, o2), b = __shfl_xor_sync(0xffffffffu, bp, o2);
                const int ia = __shfl_xor_sync(0xffffffffu, ir, o2), ib = __shfl_xor_sync(0xffffffffu, ip, o2);
                if (a > br || (a == br && ia < ir)) { br = a; ir = ia; }
                if (b > bp || (b == bp && ib < ip)) { bp = b; ip = ib; }
            }
            if (lane == 0) { s_fd[0] = br; s_fd[1] = bp; s_fd[2] = p.rank == 0 ? m : 0.f; s_fi[0] = ir; s_fi[1] = ip; }
        }
        __syncthreads();
        d = RowDraw{s_fd[0], s_fd[1], s_fd[2], s_fi[0], s_fi[1]};
#if SAMP_CL
        if (samp_csize() > 1) samp_cluster_idle();
#endif
    } else {
#if SAMP_CL
        d = samp_row_draw(p, sp, z, p.vq + (size_t)j * p.vpad, inv_t, cut, lse, logz, lseq, w, resid, 8u * j, resid, false, true);
#else
        d = samp_row_draw(p, sp, z, p.vq + (size_t)j * p.vpad, inv_t, cut, lse, logz, lseq, w, resid, 8u * j, resid);
#endif
    }
    if (tid < p.tp) {
        uint32_t* a = p.recv[tid] + p.so_bv + (size_t)(((par * SROWS + j) * 8 + p.rank) * BV_ITEMS) * 4;
        st_v4(a, make_uint4(__float_as_uint(d.kr), tag, (uint32_t)d.ir, tag));
        st_v4(a + 4, make_uint4(__float_as_uint(d.kp), tag, (uint32_t)d.ip, tag));
        st_v4(a + 8, make_uint4(__float_as_uint(d.mass), tag, 0u, tag));
    }
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(p.out + 9, 1) + 1 == M;
    __syncthreads();
    if (!s_last) return;
    __shared__ float g_m[SROWS][8], g_kr[SROWS][8], g_kp[SROWS][8];
    __shared__ int g_ir[SROWS][8], g_ip[SROWS][8];
    if (tid < (L + 1) * p.tp) {
        const int i = tid / p.tp, r = tid % p.tp;
        const uint32_t* a = p.recv[p.rank] + p.so_bv + (size_t)(((par * SROWS + i) * 8 + r) * BV_ITEMS) * 4;
        uint4 u, v, x;
        do {
            u = ld_v4(a);
            v = ld_v4(a + 4);
            x = ld_v4(a + 8);
        } while (u.y != tag || u.w != tag || v.y != tag || v.w != tag || x.y != tag || x.w != tag);
        g_kr[i][r] = __uint_as_float(u.x); g_ir[i][r] = (int)u.z;
        g_kp[i][r] = __uint_as_float(v.x); g_ip[i][r] = (int)v.z;
        g_m[i][r] = __uint_as_float(x.x);
    }
    __syncthreads();
    if (warp == 0) {
        __syncwarp();
        if (lane <= L) {
            const int i = lane;
            float gm = 0.f, gr = -INFINITY, gp = -INFINITY;
            int tr = 0x7fffffff, tq = 0x7fffffff;
            for (int r = 0; r < p.tp; ++r) {
                gm += g_m[i][r];
                const float a = g_kr[i][r], b = g_kp[i][r];
                const int ia = g_ir[i][r], ib = g_ip[i][r];
                if (a > gr || (a == gr && ia < tr)) { gr = a; tr = ia; }
                if (b > gp || (b == gp && ib < tq)) { gp = b; tq = ib; }
            }
            s_m[i] = gm; s_kr[i] = gr; s_ir[i] = tr; s_kp[i] = gp; s_ip[i] = tq;
        }
        __syncwarp();
        bool pass = false;
        if (lane >= 1 && lane <= L) {
            const int i = lane;
            const float wi = s_w[i];
            const float h = i == L ? wi : (wi >= 1.f ? 1.f : s_m[i] / (s_m[i] + 1.f - wi));
            const float u = rng_u01(rng_bits(sp, RNG_ACCEPT, (unsigned)(i - 1), 0u));
            pass = u < h;
        }
        const unsigned passed = __ballot_sync(0xffffffffu, pass);
        if (lane == 0) {
            const int n = passed ? 31 - __clz(passed) : 0;
            const int tok = (n < L && s_kr[n] != -INFINITY) ? s_ir[n] : s_ip[n];
            p.out[0] = n;
            p.out[1] = tok;
            p.next[n] = tok;
            *p.seq = samp_seq_next(seq);
            p.sp->step = sp.step + 1;
            p.out[8] = 0;
            p.out[9] = 0;
        }
    }
}

extern "C" __global__ void __launch_bounds__(SAMP_NT) samp_stats(const Meta* meta, const __grid_constant__ SampIo p) {
    pdl_wait();
    pdl_launch();
#if SAMP_CL
    const uint32_t S = samp_csize();
    const int j = (int)(blockIdx.x / S), M = meta->m, L = p.last_only ? 0 : M - 1;
    if (j >= M || (p.last_only && j != M - 1)) return;
    if (S > 1 && samp_crank() > 0) {
        if (!p.block || p.last_only) return;
        const SampParams sp = *p.sp;
        samp_row_draw(p, sp, p.logits + (size_t)j * p.vpad, p.vq + (size_t)j * p.vpad, 1.f / sp.temp, 0.f, 0.f, 0.f, 0.f, 0.f, false, 0u, false, true);
        return;
    }
#else
    const int j = blockIdx.x, M = meta->m, L = p.last_only ? 0 : M - 1;
    if (j >= M || (p.last_only && j != M - 1)) return;
#endif
    const int jr = p.last_only ? 0 : j;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const SampParams sp = *p.sp;
    const float inv_t = 1.f / sp.temp;
    __shared__ float s_top_p;
    if (tid == 0) s_top_p = sp.top_p;
    const unsigned seq = *p.seq, tag = seq + 1, par = seq & 1;
    __shared__ float shf[32];
    __shared__ unsigned s_nc;
    __shared__ uint32_t pk[PKW];
    __shared__ uint32_t rx[8][PKW];
    __shared__ float s_misc[8];
    __shared__ float ss[512], sv2[512], scum[512];
    __shared__ int sid[512];
    __shared__ float gv[64], gq[64];
    __shared__ int gid[64];
    __shared__ int s_n, s_base[9];
    __shared__ unsigned hc[256];
    __shared__ int s_tb;
    __shared__ float wm[32], w9[32], we[32], wq_m[32], wq_e[32];
    __shared__ int wc[32];
    __shared__ float fbv[2048];
    __shared__ int fbi[2048];
    const float* z = p.logits + (size_t)j * p.vpad;
    const float* rec0 = p.spart + (size_t)j * p.nquads * 20;
    float sd = -INFINITY, vd = -INFINITY;
    if (tid == 0 && j < L) {
        const int d = p.drafts[j + 1] - p.offset;
        if (d >= 0 && d < p.vloc) {
            sd = z[d] * inv_t;
            vd = p.vq[(size_t)j * p.vpad + d] * inv_t;
        }
    }
    float qsm = -INFINITY, qse = 0.f;
    if (j < L && tid < p.qn) {
        const float2 q = p.qstat[(size_t)j * p.qn + tid];
        qsm = q.x;
        qse = q.y;
    }
    if (!p.spart) {
        float lm = -INFINITY, le = 0.f;
        for (int i = tid; i < p.vloc; i += SAMP_NT) lm = fmaxf(lm, z[i] * inv_t);
        lm = blk_max(lm, shf);
        if (lm != -INFINITY)
            for (int i = tid; i < p.vloc; i += SAMP_NT) le += expf(z[i] * inv_t - lm);
        le = blk_sum(le, shf);
        if (tid == 0) {
            pk[0] = __float_as_uint(lm); pk[1] = __float_as_uint(le); pk[2] = __float_as_uint(INFINITY); pk[3] = 0;
            pk[4] = pk[5] = pk[6] = __float_as_uint(-INFINITY); pk[7] = 0;
        }
    } else {
        constexpr int EPT = 4096 / SAMP_NT;
        const int ne = p.nquads * 8;
        float qm = -INFINITY, qe = 0.f, q9 = -INFINITY, ev[EPT];
        int eid[EPT];
        if (tid < p.nquads) { qm = rec0[tid * 20]; qe = rec0[tid * 20 + 1]; q9 = rec0[tid * 20 + 2]; }
#pragma unroll
        for (int u = 0; u < EPT; ++u) {
            const int i = u * SAMP_NT + tid;
            ev[u] = i < ne ? rec0[(i >> 3) * 20 + 3 + (i & 7)] : -INFINITY;
            eid[u] = i < ne ? __float_as_int(rec0[(i >> 3) * 20 + 11 + (i & 7)]) : 0;
        }
        if (tid < 256) hc[tid] = 0;
        if (tid == 0) s_nc = 0;
        {
            const float a = wmax(qm), b = wmax(q9);
            if (lane == 0) { wm[warp] = a; w9[warp] = b; }
        }
        {
            __syncwarp();
            float m = qsm, e = qse;
            for (int o = 1; o < 32; o <<= 1) {
                const float om = __shfl_xor_sync(0xffffffffu, m, o), oe = __shfl_xor_sync(0xffffffffu, e, o);
                lse_add(m, e, om, oe);
            }
            if (lane == 0) { wq_m[warp] = m; wq_e[warp] = e; }
            if (tid == 0) {
                pk[4] = __float_as_uint(sd);
                pk[5] = __float_as_uint(vd);
            }
        }
        __syncthreads();
        if (warp == 0) {
            __syncwarp();
            float m = lane < NWARP ? wq_m[lane] : -INFINITY, e = lane < NWARP ? wq_e[lane] : 0.f;
            for (int o = 1; o < 32; o <<= 1) {
                const float om = __shfl_xor_sync(0xffffffffu, m, o), oe = __shfl_xor_sync(0xffffffffu, e, o);
                lse_add(m, e, om, oe);
            }
            if (lane == 0) {
                pk[6] = __float_as_uint(m);
                pk[7] = __float_as_uint(e);
            }
        }
        __syncthreads();
        const float rm = wmax(lane < NWARP ? wm[lane] : -INFINITY), b9 = wmax(lane < NWARP ? w9[lane] : -INFINITY);
        {
            const float e = wsum(tid < p.nquads && qe > 0.f ? qe * expf(qm - rm) : 0.f);
            if (lane == 0) we[warp] = e;
        }
        __syncthreads();
        const float e = wsum(lane < NWARP ? we[lane] : 0.f);
        const float lo = s_top_p < 1.f ? fmaxf(fmaxf(b9, rm - 16.f), rm - logf(e) + p.leps) : INFINITY;
        {
            int c = 0;
#pragma unroll
            for (int u = 0; u < EPT; ++u) c += ev[u] > lo;
            __syncwarp();
            c = __reduce_add_sync(0xffffffffu, c);
            if (lane == 0) wc[warp] = c;
        }
        if (tid == 0) { pk[0] = __float_as_uint(rm); pk[1] = __float_as_uint(e); }
        __syncthreads();
        __syncwarp();
        int tb = 256;
        if ((int)__reduce_add_sync(0xffffffffu, lane < NWARP ? wc[lane] : 0) > SNC) {
#pragma unroll
            for (int u = 0; u < EPT; ++u)
                if (ev[u] > lo) atomicAdd(&hc[max(0, min(255, (int)((rm - ev[u]) * 16.f)))], 1u);
            __syncthreads();
            if (warp == 0) {
                __syncwarp();
                unsigned loc[8], tot = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) { loc[i] = hc[lane * 8 + i]; tot += loc[i]; }
                unsigned pre = tot;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const unsigned v = __shfl_up_sync(0xffffffffu, pre, o);
                    if (lane >= o) pre += v;
                }
                pre -= tot;
                int t = lane * 8 + 7;
                unsigned c = pre;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (c + loc[i] > SNC) { t = lane * 8 + i; break; }
                    c += loc[i];
                }
                const unsigned b = __ballot_sync(0xffffffffu, pre + tot > SNC);
                if (lane == 0) s_tb = __shfl_sync(0xffffffffu, t, __ffs(b) - 1);
                else __shfl_sync(0xffffffffu, t, __ffs(b) - 1);
            }
            __syncthreads();
            tb = s_tb;
        }
        float bmax = b9;
#pragma unroll
        for (int u = 0; u < EPT; ++u) {
            if (ev[u] > lo && (tb == 256 || max(0, min(255, (int)((rm - ev[u]) * 16.f))) < tb)) {
                const int slot = atomicAdd(&s_nc, 1);
                gv[slot] = ev[u];
                gid[slot] = eid[u];
                gq[slot] = j < L ? p.vq[(size_t)j * p.vpad + min(max(eid[u] - p.offset, 0), p.vloc - 1)] * inv_t : 0.f;
            } else if (ev[u] > bmax) {
                bmax = ev[u];
            }
        }
        bmax = wmax(bmax);
        if (lane == 0) wm[warp] = bmax;
        __syncthreads();
        const int nc = s_nc;
        if (tid < nc) {
            const float v = gv[tid];
            const int id = gid[tid];
            int r = 0;
#pragma unroll 8
            for (int k = 0; k < nc; ++k) r += gv[k] > v || (gv[k] == v && gid[k] < id);
            pk[8 + r] = __float_as_uint(v);
            pk[8 + SNC + r] = (uint32_t)id;
            pk[8 + 2 * SNC + r] = __float_as_uint(gq[tid]);
        } else if (warp == NWARP - 1) {
            const float b = wmax(lane < NWARP ? wm[lane] : -INFINITY);
            if (lane == 0) { pk[2] = __float_as_uint(b); pk[3] = nc; }
        }
    }
    __syncthreads();
    {
        const int nc = (int)pk[3], cnt = 4 + nc + (nc + 1) / 2;
        for (int w = tid; w < cnt * p.tp; w += SAMP_NT) {
            const int q = w / cnt, i = w % cnt;
            uint4 v;
            if (i < 4) v = make_uint4(pk[2 * i], tag, pk[2 * i + 1], tag);
            else if (i < 4 + nc) v = make_uint4(pk[8 + i - 4], tag, pk[8 + SNC + i - 4], tag);
            else { const int k = 2 * (i - 4 - nc); v = make_uint4(pk[8 + 2 * SNC + k], tag, pk[8 + 2 * SNC + k + 1], tag); }
            st_v4(p.recv[q] + SO_PK + ((size_t)((par * SROWS + jr) * 8 + p.rank) * PKB + i) * 4, v);
        }
    }
    for (int w = tid; w < 4 * p.tp; w += SAMP_NT) {
        const int src = w >> 2, i = w & 3;
        const uint32_t* a = p.recv[p.rank] + SO_PK + ((size_t)((par * SROWS + jr) * 8 + src) * PKB + i) * 4;
        uint4 v;
        do { v = ld_v4(a); } while (v.y != tag || v.w != tag);
        rx[src][2 * i] = v.x;
        rx[src][2 * i + 1] = v.z;
    }
    __syncthreads();
    for (int w = tid; w < (SNC + 32) * p.tp; w += SAMP_NT) {
        const int src = w / (SNC + 32), i = w % (SNC + 32), nc = (int)rx[src][3];
        const int it = i < SNC ? (i < nc ? 4 + i : -1) : (2 * (i - SNC) < nc ? 4 + nc + i - SNC : -1);
        if (it < 0) continue;
        const uint32_t* a = p.recv[p.rank] + SO_PK + ((size_t)((par * SROWS + jr) * 8 + src) * PKB + it) * 4;
        uint4 v;
        do { v = ld_v4(a); } while (v.y != tag || v.w != tag);
        if (i < SNC) { rx[src][8 + i] = v.x; rx[src][8 + SNC + i] = v.z; }
        else { const int k = 2 * (i - SNC); rx[src][8 + 2 * SNC + k] = v.x; rx[src][8 + 2 * SNC + k + 1] = v.z; }
    }
    __syncthreads();
    if (warp == 0) {
        const bool on = lane < p.tp;
        const float m = on ? __uint_as_float(rx[lane][0]) : -INFINITY, e = on ? __uint_as_float(rx[lane][1]) : 0.f;
        const float mq = on ? __uint_as_float(rx[lane][6]) : -INFINITY, eq = on ? __uint_as_float(rx[lane][7]) : 0.f;
        const float gmx = wmax(m), gmq = wmax(mq);
        const float se = wsum(on && e > 0.f ? e * expf(m - gmx) : 0.f), seq = wsum(on && eq > 0.f ? eq * expf(mq - gmq) : 0.f);
        const float cmax = wmax(on ? __uint_as_float(rx[lane][2]) : -INFINITY);
        const float sd = wmax(on ? __uint_as_float(rx[lane][4]) : -INFINITY), vd = wmax(on ? __uint_as_float(rx[lane][5]) : -INFINITY);
        const int cnt = on ? (int)rx[lane][3] : 0;
        int pre = cnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, pre, o);
            if (lane >= o) pre += v;
        }
        const int n = __shfl_sync(0xffffffffu, pre, 31);
        if (lane <= 8) s_base[lane] = lane < p.tp ? pre - cnt : n;
        if (lane == 0) {
            s_misc[0] = gmx + logf(se);
            s_misc[1] = cmax;
            s_misc[2] = gmq + logf(seq);
            s_misc[3] = sd;
            s_misc[4] = vd;
            s_misc[5] = gmx;
        }
        if (lane == 0) s_n = n;
    }
    __syncthreads();
    const float lse = s_misc[0];
    const int n = s_n;
    if (tid < n) {
        int r0 = 0;
#pragma unroll
        for (int r = 1; r < 8; ++r) r0 += r < p.tp && s_base[r] <= tid;
        const int k0 = tid - s_base[r0];
        const float v = __uint_as_float(rx[r0][8 + k0]);
        const int id = (int)rx[r0][8 + SNC + k0];
        int rank = k0;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= p.tp || r == r0) continue;
            int lo2 = 0, len = (int)rx[r][3];
            while (len > 0) {
                const int h = len >> 1, mid = lo2 + h;
                const float w = __uint_as_float(rx[r][8 + mid]);
                if (w > v || (w == v && (int)rx[r][8 + SNC + mid] < id)) { lo2 = mid + 1; len -= h + 1; } else len = h;
            }
            rank += lo2;
        }
        ss[rank] = v;
        sid[rank] = id;
        sv2[rank] = __uint_as_float(rx[r0][8 + 2 * SNC + k0]);
    }
    __syncthreads();
    __shared__ float s_cut, s_logz;
    __shared__ int s_fast, s_k, s_hit;
    float* o = p.rowres + jr * RRW;
    if (tid == 0) { s_hit = -1; s_k = n; }
    const float P = s_top_p;
    if (P < 1.f) {
        float c = tid < n ? expf(ss[tid] - lse) : 0.f;
        __syncwarp();
#pragma unroll
        for (int o2 = 1; o2 < 32; o2 <<= 1) {
            const float v = __shfl_up_sync(0xffffffffu, c, o2);
            if (lane >= o2) c += v;
        }
        if (lane == 31 && warp < 16) shf[warp] = c;
        __syncthreads();
        for (int w = 0; w < warp && w < 16; ++w) c += shf[w];
        if (tid < 512) scum[tid] = c;
        __syncthreads();
        if (tid < n && c >= P && (tid == 0 || scum[tid - 1] < P)) s_hit = tid;
        __syncthreads();
        if (s_hit >= 0) {
            const float cut = ss[s_hit];
            if (tid >= s_hit && tid < n && ss[tid] == cut && (tid + 1 == n || ss[tid + 1] != cut)) s_k = tid + 1;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int found = s_hit >= 0, k = s_k;
        const float cut = found ? ss[s_hit] : -INFINITY;
        s_fast = found && cut > s_misc[1] && k <= 64 && !p.force_fb;
        s_cut = cut;
        s_logz = found ? logf(scum[k - 1]) : 0.f;
    }
    __syncthreads();
    if (s_fast)
        for (int i = tid; i < s_k; i += SAMP_NT) {
            o[8 + i] = ss[i];
            o[8 + 64 + i] = __int_as_float(sid[i]);
            o[8 + 128 + i] = sv2[i];
        }
    if (!s_fast && P < 1.f) samp_fallback(p, z, inv_t, P, lse, s_misc[5], s_misc[1], tag, jr, hc, fbv, fbi, &s_cut, &s_logz, s_k);
    if (tid == 0 && p.rank == 0) {
        atomicAdd(&p.out[4], 1);
        if (!s_fast) atomicAdd(&p.out[5], 1);
    }
    if (tid == 0) {
        int acc = 0;
        float lr = -INFINITY;
        if (j < L) {
            const float sd = s_misc[3], vd = s_misc[4];
            const float pd = sd >= s_cut ? expf(sd - lse - s_logz) : 0.f;
            const float qd = expf(vd - s_misc[2]);
            const float u = rng_u01(rng_bits(sp, RNG_ACCEPT, (unsigned)j, 0u));
            acc = u * qd < pd;
            lr = sd >= s_cut ? (sd - lse - s_logz) - (vd - s_misc[2]) : -INFINITY;
        }
        o[0] = (float)acc;
        o[1] = s_cut;
        o[2] = lse;
        o[3] = s_logz;
        o[4] = s_misc[2];
        o[5] = (float)(s_fast != 0);
        o[6] = (float)s_k;
        o[7] = lr;
    }
    __shared__ int s_last;
    __syncthreads();
    if (p.block && !p.last_only) {
        if (tid == 0) {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            atomicAdd(&p.out[8], 1);
        }
        samp_block(p, j, M, sp, seq, inv_t, z, s_cut, lse, s_logz, s_misc[2], s_fast != 0, s_k, o);
        return;
    }
    if (tid == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        const int done = atomicAdd(&p.out[8], 1) + 1;
        s_last = done == (p.last_only ? 1 : M);
        if (s_last) {
            p.out[8] = 0;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }
    }
    __syncthreads();
    if (s_last) {
        samp_pick(p, meta->pos, M, sp, seq);
    }
}
