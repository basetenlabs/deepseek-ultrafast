#include "common.cuh"
#include "sample.cuh"

struct Meta {
    int pos;
    int m;
    int tokens[16];
};
DEV float bf(bf16 v) { return __bfloat162float(v); }

struct MkParams {
    const uint8_t* w;
    int nquads;
    int qpc;
    int vloc, vpad, offset;
    const bf16* table;
    const float* logits;
    int* drafts;
    int blk;
    float2* cand;
    unsigned* bar;
    uint32_t* recv[8];
    int tp, rank;
    unsigned* seq;
    const bf16* cx;
    const bf16* cw;
    int ldx, dim;
    float* xpart;
    float* conf;
    float theta;
    int _pad1;
    unsigned long long hcond;
    const SampParams* samp;
    float* vq;
    float2* qstat;
    float theta_r[8];
};
DEV void mk_lse(float& m, float& s, float m2, float s2) {
    const float mx = fmaxf(m, m2), mr = mx == -INFINITY ? 0.f : mx;
    s = s * expf(m - mr) + s2 * expf(m2 - mr);
    m = mx;
}
constexpr int MK_K = 256, MK_STEPS = MK_K / 16, MK_QUADS = 2;
DEV bool mk_better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

DEV unsigned long long mk_key(float v, int i) {
    const unsigned u = __float_as_uint(v == 0.f ? 0.f : v);
    const unsigned o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)o << 32) | (unsigned long long)(0xffffffffu - (unsigned)i);
}
DEV float mk_val(unsigned long long k) {
    const unsigned o = (unsigned)(k >> 32);
    return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o);
}
DEV int mk_idx(unsigned long long k) { return (int)(0xffffffffu - (unsigned)(k & 0xffffffffu)); }

extern "C" __global__ void __launch_bounds__(512, 2) markov_rounds_ld(const Meta* meta, const __grid_constant__ MkParams p, unsigned long long* kmax) {
    extern __shared__ __align__(16) uint8_t mk_w[];
    __shared__ __align__(16) uint16_t sx[MK_K];
    __shared__ float red[MK_QUADS][2][64];
    __shared__ float sv[16];
    __shared__ int si[16];
    __shared__ int s_tok;
    __shared__ int s_lead;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = gridDim.x;
    const int q0 = blockIdx.x * p.qpc;
    const int nq = min(p.qpc, p.nquads - q0);
    for (int i = tid; i < nq * MK_STEPS * 128; i += blockDim.x)
        ((uint4*)mk_w)[i] = ((const uint4*)(p.w + (size_t)q0 * MK_STEPS * 2048))[i];
    const unsigned seq0 = *p.seq;
    if (p.cx) {
        const int d0 = (int)((long long)blockIdx.x * p.dim / G), d1 = (int)((long long)(blockIdx.x + 1) * p.dim / G);
        if (warp < p.blk) {
            float a = 0.f;
            for (int d = d0 + lane; d < d1; d += 32) a += bf(p.cw[d]) * bf(p.cx[(size_t)warp * p.ldx + d]);
            a = warp_sum(a);
            if (lane == 0) p.xpart[blockIdx.x * 8 + warp] = a;
        }
    }
    pdl_wait();
    const bool sampling = p.samp && p.samp->temp > 0.f;
    const SampParams sp = sampling ? *p.samp : SampParams{1.f, 1.f, 0ull, 0ull, 0};
    const float inv_t = 1.f / sp.temp;
    const int R = q0 * 64 + tid;
    const bool live = tid < nq * 64 && R < p.vloc;
    float lg[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) lg[r] = live && r < p.blk ? p.logits[(size_t)r * p.vpad + R] : 0.f;
    if (tid == 0) {
        s_tok = meta->tokens[0];
        if (blockIdx.x == 0) p.drafts[0] = s_tok;
    }
    __syncthreads();
    const int lq = warp >> 3, rg = warp & 3, ph = (warp >> 2) & 1;
    const int g = lane >> 2, t = lane & 3;
    __shared__ float s_conf[8], s_cum;
    __shared__ int s_L, s_stop1;
    float cwr[MK_K / 32], xpr[8];
    if (p.cx && warp == 0)
#pragma unroll
        for (int k = 0; k < MK_K / 32; ++k) cwr[k] = bf(p.cw[p.dim + lane + 32 * k]);
    auto xp_load = [&](int i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) xpr[j] = lane + 32 * j < G ? __ldcg(p.xpart + (lane + 32 * j) * 8 + i) : 0.f;
    };
    auto conf_mk = [&]() {
        float a = 0.f;
#pragma unroll
        for (int k = 0; k < MK_K / 32; ++k) a += cwr[k] * bf(__ushort_as_bfloat16(sx[lane + 32 * k]));
        return a;
    };
    auto conf_fin = [&](float a) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (lane + 32 * j < G) a += xpr[j];
        return warp_sum(a);
    };
    float a0 = 0.f;
    if (tid == 0) { s_cum = 1.f; s_L = p.blk; s_stop1 = 0; }
    int rounds = 0;
    for (int r = 0; r < p.blk; ++r) {
        const unsigned sq = seq0 + r + 1;
        const int par = r & 1;
        if (p.cx && r == 1 && warp == 0) {
            xp_load(0);
            const float cf = conf_fin(a0);
            if (lane == 0) {
                s_conf[0] = cf;
                s_cum *= 1.f / (1.f + expf(-cf));
                if (s_cum < p.theta_r[0]) s_stop1 = 1;
            }
        }
        if (p.cx && r >= 1 && warp == 0) xp_load(r);
        if (tid < MK_K / 8) ((uint4*)sx)[tid] = ((const uint4*)(p.table + (size_t)s_tok * MK_K))[tid];
        __syncthreads();
        if (p.cx && r == 0 && warp == 0) a0 = conf_mk();
        if (p.cx && r >= 1) {
            if (warp == 0 && !s_stop1) {
                const float cf = conf_fin(conf_mk());
                if (lane == 0) {
                    s_conf[r] = cf;
                    s_cum *= 1.f / (1.f + expf(-cf));
                    if (s_cum < p.theta_r[r]) s_L = r;
                }
            }
            __syncthreads();
            if (s_stop1 || s_L < p.blk) break;
        }
        ++rounds;
        if (lq < nq) {
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            const uint8_t* wq = mk_w + (size_t)lq * MK_STEPS * 2048;
#pragma unroll
            for (int j = ph; j < MK_STEPS; j += 2) {
                const uint4 v = *(const uint4*)(wq + j * 2048 + rg * 512 + lane * 16);
                const uint32_t a[4] = {v.x, v.y, v.z, v.w};
                const uint32_t b0 = g == 0 ? *(const uint32_t*)(sx + j * 16 + 2 * t) : 0u;
                const uint32_t b1 = g == 0 ? *(const uint32_t*)(sx + j * 16 + 8 + 2 * t) : 0u;
                mma_bf16(acc, a, b0, b1);
            }
            if (t == 0) {
                red[lq][ph][rg * 16 + g] = acc[0];
                red[lq][ph][rg * 16 + g + 8] = acc[2];
            }
        }
        __syncthreads();
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        if (tid < 128 && live) {
            const float mk = red[tid >> 6][0][tid & 63] + red[tid >> 6][1][tid & 63];
            float l = lg[0];
#pragma unroll
            for (int k = 1; k < 8; ++k) if (k == r) l = lg[k];
            bv = l + mk;
            bi = R + p.offset;
        }
        if (sampling && warp < 4) {
            __syncwarp();
            float lm = bv * inv_t, ls = bv == -INFINITY ? 0.f : 1.f;
            if (live) p.vq[(size_t)r * p.vpad + R] = bv;
            for (int o = 16; o > 0; o >>= 1) {
                const float om = __shfl_xor_sync(0xffffffff, lm, o), os = __shfl_xor_sync(0xffffffff, ls, o);
                mk_lse(lm, ls, om, os);
            }
            if (lane == 0) p.qstat[(size_t)r * G * 4 + blockIdx.x * 4 + warp] = make_float2(lm, ls);
            const float gk = bv * inv_t + rng_gumbel(rng_bits(sp, RNG_DRAFT, (unsigned)bi, (unsigned)r));
            bv = bv == -INFINITY ? bv : gk;
        }
        if (warp < 4) {
            __syncwarp();
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, bv, o);
                const int oi = __shfl_xor_sync(0xffffffff, bi, o);
                const bool t = mk_better(ov, oi, bv, bi);
                bv = t ? ov : bv;
                bi = t ? oi : bi;
            }
            if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
        }
        __syncthreads();
        if (tid == 0) {
            for (int w = 1; w < 4; ++w)
                if (mk_better(sv[w], si[w], sv[0], si[0])) { sv[0] = sv[w]; si[0] = si[w]; }
            atomicMax(kmax + r, mk_key(sv[0], si[0]));
            unsigned old;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p.bar) : "memory");
            s_lead = old == (unsigned)G * (r + 1) - 1;
        }
        __syncthreads();
        if (r == 0) pdl_launch();
        if (warp == 0) {
            if (s_lead) {
                unsigned long long k = lane == 0 ? atomicAdd(kmax + r, 0ull) : 0ull;
                k = __shfl_sync(0xffffffff, k, 0);
                bv = mk_val(k);
                bi = mk_idx(k);
            }
            if (s_lead && lane < p.tp) {
                asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p.recv[lane] + (par * 8 + p.rank) * 4), "r"(__float_as_uint(bv)), "r"(sq),
                             "r"((unsigned)bi), "r"(sq) : "memory");
            }
            float v = -INFINITY;
            int i = 0x7fffffff;
            if (lane < p.tp) {
                uint4 a;
                do {
                    asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(p.recv[p.rank] + (par * 8 + lane) * 4) : "memory");
                } while (a.y != sq || a.w != sq);
                v = __uint_as_float(a.x);
                i = (int)a.z;
            }
            __syncwarp();
            float b = -INFINITY;
            int bj = 0x7fffffff;
            for (int q = 0; q < p.tp; ++q) {
                const float rv = __shfl_sync(0xffffffff, v, q);
                const int ri = __shfl_sync(0xffffffff, i, q);
                const bool t = rv > b || (rv == b && ri < bj);
                b = t ? rv : b;
                bj = t ? ri : bj;
            }
            if (lane == 0) {
                s_tok = bj;
                if (blockIdx.x == 0) p.drafts[r + 1] = bj;
            }
        }
        __syncthreads();
    }
    if (p.cx && blockIdx.x == 0 && tid == 0) {
        const int L = s_stop1 ? 1 : s_L;
        const int nc = s_stop1 ? 1 : min(s_L + 1, p.blk);
        for (int i = 0; i < nc; ++i) p.conf[i] = s_conf[i];
        cudaGraphSetConditional((cudaGraphConditionalHandle)p.hcond, (unsigned)(L - 1));
        p.conf[32] = __int_as_float(L);
        Meta* mw = (Meta*)meta;
        mw->m = L + 1;
        for (int i = 0; i <= L; ++i) mw->tokens[i] = p.drafts[i];
    }
    if (tid == 0) {
        if (atomicAdd(&p.bar[1], 1u) == (unsigned)G - 1) {
            p.bar[0] = 0;
            p.bar[1] = 0;
            for (int i = 0; i < 8; ++i) kmax[i] = 0ull;
            *p.seq = seq0 + rounds;
            __threadfence();
        }
    }
}
