#include "common.cuh"
#include "sample.cuh"

struct Meta {
    int pos;
    int m;
    int tokens[16];
};

DEV float bf(bf16 v) { return __bfloat162float(v); }
DEV bf16 tobf(float v) { return __float2bfloat16(v); }

template <int NT>
DEV float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < NT / 32; ++i) r += sh[i];
    return r;
}

extern "C" __global__ void embed_hc(const Meta* meta, const bf16* table, bf16* h, int dim, int hc,
                                    const int* token_map, int* cmp_hist, const int* toks) {
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.x;
    if (m >= meta->m) return;
    const int tok = toks ? toks[m] : meta->tokens[m];
    const uint4* src = (const uint4*)(table + (size_t)tok * dim);
    for (int i = threadIdx.x; i < hc * dim / 8; i += blockDim.x) {
        ((uint4*)(h + (size_t)m * hc * dim))[i] = src[i % (dim / 8)];
    }
    if (threadIdx.x == 0 && cmp_hist) cmp_hist[meta->pos + m] = token_map[tok];
}

extern "C" __global__ void hc_mean(const Meta* meta, const bf16* h, bf16* out, int dim, int ldo) {
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.y;
    if (m >= meta->m) return;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float s = 0.f;
    for (int c = 0; c < 4; ++c) s += bf(h[((size_t)m * 4 + c) * dim + d]);
    out[(size_t)m * ldo + d] = tobf(s * 0.25f);
}

#include "rope.cuh"

extern "C" __global__ void __launch_bounds__(64) rope_quant(const Meta* meta, const RopeParams p) { rope_body((const int*)meta, p, blockIdx.x, threadIdx.x, 0); }
extern "C" __global__ void __launch_bounds__(256) rope_quant4(const Meta* meta, const RopeParams p) {
    rope_body((const int*)meta, p, blockIdx.x * 4 + (threadIdx.x >> 6), threadIdx.x & 63, threadIdx.x >> 6);
}
extern "C" __global__ void __launch_bounds__(64) rope_quant2(const Meta* meta, const RopeParams p1, const RopeParams p2, int n1) {
    if ((int)blockIdx.x < n1) rope_body((const int*)meta, p1, blockIdx.x, threadIdx.x, 0);
    else rope_body((const int*)meta, p2, blockIdx.x - n1, threadIdx.x, 0);
}

extern "C" __global__ void scale_bf16(const Meta* meta, bf16* x, int n_per_token, float alpha) {
    pdl_wait();
    pdl_launch();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= meta->m * n_per_token) return;
    x[i] = tobf(bf(x[i]) * alpha);
}

struct EngramHash {
    const long long* mult;
    const long long* primes;
    const long long* offsets;
    int max_ngram, heads, pad_id;
};

extern "C" __global__ void engram_gather(const Meta* meta, const int* cmp_hist, const EngramHash hp,
                                         const uint8_t* table, const uint8_t* scales, long long row0,
                                         long long rows, float* out, int head_dim) {
    pdl_wait();
    pdl_launch();
    const int col = blockIdx.x, m = blockIdx.y;
    if (m >= meta->m) return;
    const int pos = meta->pos + m;
    const int n = col / hp.heads + 1;
    const int h = col % hp.heads;
    long long rolling = 0;
    bool blocked = false;
    for (int s = 0; s <= n; ++s) {
        int t;
        blocked = blocked || pos < s;
        t = blocked ? hp.pad_id : cmp_hist[pos - s];
        blocked = blocked || t == -1;
        if (blocked) t = hp.pad_id;
        const long long prod = (long long)t * hp.mult[s];
        rolling = s == 0 ? prod : (rolling ^ prod);
    }
    const long long id = rolling % hp.primes[(n - 1) * hp.heads + h] + hp.offsets[(n - 1) * hp.heads + h];
    float* o = out + ((size_t)m * gridDim.x + col) * head_dim;
    const long long local = id - row0;
    const int e = threadIdx.x * 4;
    if (local < 0 || local >= rows) {
        for (int i = 0; i < 4; ++i) o[e + i] = 0.f;
        return;
    }
    const uint8_t* r = table + local * head_dim;
    const float sc = e8m0_to_f32(scales[local * (head_dim / 32) + e / 32]);
    const uint32_t q = *(const uint32_t*)(r + e);
    uint32_t h0 = e4m3x2_to_f16x2(q & 0xffff), h1 = e4m3x2_to_f16x2(q >> 16);
    o[e + 0] = bf(tobf(__half2float(__ushort_as_half((unsigned short)(h0 & 0xffff))) * sc));
    o[e + 1] = bf(tobf(__half2float(__ushort_as_half((unsigned short)(h0 >> 16))) * sc));
    o[e + 2] = bf(tobf(__half2float(__ushort_as_half((unsigned short)(h1 & 0xffff))) * sc));
    o[e + 3] = bf(tobf(__half2float(__ushort_as_half((unsigned short)(h1 >> 16))) * sc));
}

struct EgPeers {
    const uint8_t* table[8];
    const uint8_t* scales[8];
    const float* part[8];
    const unsigned* flag[8];
    const unsigned* epoch;
    long long shard;
    int tp, nloc;
};
extern "C" __global__ void engram_gather_peer(const Meta* meta, const int* cmp_hist, const EngramHash hp,
                                              const __grid_constant__ EgPeers q, bf16* out, int head_dim) {
    pdl_wait();
    pdl_launch();
    const int col = blockIdx.x, m = blockIdx.y;
    if (m >= meta->m) return;
    const int pos = meta->pos + m;
    const int n = col / hp.heads + 1;
    const int h = col % hp.heads;
    long long rolling = 0;
    bool blocked = false;
    for (int s = 0; s <= n; ++s) {
        blocked = blocked || pos < s;
        int t = blocked ? hp.pad_id : cmp_hist[pos - s];
        blocked = blocked || t == -1;
        if (blocked) t = hp.pad_id;
        const long long prod = (long long)t * hp.mult[s];
        rolling = s == 0 ? prod : (rolling ^ prod);
    }
    const long long id = rolling % hp.primes[(n - 1) * hp.heads + h] + hp.offsets[(n - 1) * hp.heads + h];
    const int owner = (int)(id / q.shard);
    const long long local = id - owner * q.shard;
    bf16* o = out + ((size_t)m * gridDim.x + col) * head_dim;
    const int e = threadIdx.x * 4;
    const uint8_t* r = q.table[owner] + local * head_dim;
    const float sc = e8m0_to_f32(q.scales[owner][local * (head_dim / 32) + e / 32]);
    const uint32_t v = *(const uint32_t*)(r + e);
    const uint32_t h0 = e4m3x2_to_f16x2(v & 0xffff), h1 = e4m3x2_to_f16x2(v >> 16);
    o[e + 0] = tobf(__half2float(__ushort_as_half((unsigned short)(h0 & 0xffff))) * sc);
    o[e + 1] = tobf(__half2float(__ushort_as_half((unsigned short)(h0 >> 16))) * sc);
    o[e + 2] = tobf(__half2float(__ushort_as_half((unsigned short)(h1 & 0xffff))) * sc);
    o[e + 3] = tobf(__half2float(__ushort_as_half((unsigned short)(h1 >> 16))) * sc);
}

extern "C" __global__ void engram_signal(unsigned* epoch, unsigned* flag) {
    pdl_wait();
    pdl_launch();
    const unsigned e = *epoch + 1;
    *epoch = e;
    __threadfence_system();
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(flag), "r"(e) : "memory");
}

extern "C" __global__ void __launch_bounds__(256) engram_apply_peer(const Meta* meta, const int* cmp_hist, bf16* h, const __grid_constant__ EgPeers q,
                                                                   const float* qk, int dim, float eps) {
    __shared__ float sh[8];
    const int c = blockIdx.x, m = blockIdx.y;
    if (m >= meta->m) { pdl_wait(); pdl_launch(); return; }
    if (threadIdx.x < q.tp) {
        const unsigned e = *(const volatile unsigned*)q.epoch;
        unsigned f;
        do { asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(f) : "l"(q.flag[threadIdx.x]) : "memory"); } while ((int)(f - e) < 0);
    }
    __syncthreads();
    if (cmp_hist[meta->pos + m] == -1) { pdl_wait(); pdl_launch(); return; }
    __shared__ __align__(16) float skv[2][5120];
    {
        constexpr int U = 10;
        const int n4 = dim / 4;
        float4 v[U];
        int jj[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = threadIdx.x + u * 256;
            jj[u] = i;
            if (i < 2 * n4) {
                const int row = i / n4, j = (row ? 4 * dim : c * dim) + 4 * (i % n4);
                const int r = j / q.nloc;
                v[u] = __ldcv((const float4*)(q.part[r] + (size_t)m * q.nloc + (j - r * q.nloc)));
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
            if (jj[u] < 2 * n4) *(float4*)&skv[jj[u] / n4][4 * (jj[u] % n4)] = v[u];
        for (int i = threadIdx.x + U * 256; i < 2 * n4; i += 256) {
            const int row = i / n4, j = (row ? 4 * dim : c * dim) + 4 * (i % n4);
            const int r = j / q.nloc;
            *(float4*)&skv[row][4 * (i % n4)] = __ldcv((const float4*)(q.part[r] + (size_t)m * q.nloc + (j - r * q.nloc)));
        }
    }
    __syncthreads();
    pdl_wait();
    pdl_launch();
    auto kv = [&](int row, int d) -> float { return bf(tobf(skv[row][d])); };
    bf16* hr = h + ((size_t)m * 4 + c) * dim;
    const float* w = qk + (size_t)c * dim;
    constexpr int DPT = 20;
    float av[DPT], wv[DPT];
#pragma unroll
    for (int k = 0; k < DPT; ++k) {
        const int d = threadIdx.x + k * 256;
        av[k] = d < dim ? bf(hr[d]) : 0.f;
        wv[k] = d < dim ? w[d] : 0.f;
    }
    float hh = 0.f, kk = 0.f, dot = 0.f;
#pragma unroll
    for (int k = 0; k < DPT; ++k) {
        const int d = threadIdx.x + k * 256;
        if (d >= dim) break;
        const float a = av[k], kf = kv(0, d);
        hh += a * a;
        kk += kf * kf;
        dot += a * wv[k] * kf;
    }
    hh = block_sum<256>(hh, sh);
    kk = block_sum<256>(kk, sh);
    dot = block_sum<256>(dot, sh);
    const float rstd = rsqrtf(hh / dim + eps) * rsqrtf(kk / dim + eps);
    const float dv = dot * rstd * rsqrtf((float)dim);
    const float sg = copysignf(sqrtf(fmaxf(fabsf(dv), 1e-6f)), dv);
    const float gate = 1.f / (1.f + expf(-sg));
#pragma unroll
    for (int k = 0; k < DPT; ++k) {
        const int d = threadIdx.x + k * 256;
        if (d >= dim) break;
        hr[d] = tobf(av[k] + gate * kv(1, d));
    }
}

extern "C" __global__ void __launch_bounds__(256) engram_apply(const Meta* meta, const int* cmp_hist, bf16* h, const bf16* kv,
                                                              const float* qk, int dim, float eps) {
    pdl_wait();
    pdl_launch();
    __shared__ float sh[8];
    const int c = blockIdx.x, m = blockIdx.y;
    if (m >= meta->m || cmp_hist[meta->pos + m] == -1) return;
    bf16* hr = h + ((size_t)m * 4 + c) * dim;
    const bf16* key = kv + (size_t)m * 5 * dim + (size_t)c * dim;
    const bf16* val = kv + (size_t)m * 5 * dim + (size_t)4 * dim;
    const float* w = qk + (size_t)c * dim;
    float hh = 0.f, kk = 0.f, dot = 0.f;
    for (int d = threadIdx.x; d < dim; d += 256) {
        const float a = bf(hr[d]), k = bf(key[d]);
        hh += a * a;
        kk += k * k;
        dot += a * w[d] * k;
    }
    hh = block_sum<256>(hh, sh);
    kk = block_sum<256>(kk, sh);
    dot = block_sum<256>(dot, sh);
    const float rstd = rsqrtf(hh / dim + eps) * rsqrtf(kk / dim + eps);
    const float dv = dot * rstd * rsqrtf((float)dim);
    const float sg = copysignf(sqrtf(fmaxf(fabsf(dv), 1e-6f)), dv);
    const float gate = 1.f / (1.f + expf(-sg));
    for (int d = threadIdx.x; d < dim; d += 256) hr[d] = tobf(bf(hr[d]) + gate * bf(val[d]));
}

#define MOE_MAX_SLOTS 97
struct MoePlan {
    int n_slots;
    int expert[MOE_MAX_SLOTS];
    int cnt[MOE_MAX_SLOTS];
    int tok[MOE_MAX_SLOTS][16];
    int pair[MOE_MAX_SLOTS][16];
    float w[MOE_MAX_SLOTS][16];
};

#include "route.cuh"

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
extern "C" __global__ void __launch_bounds__(512, 2) markov_rounds(const Meta* meta, const __grid_constant__ MkParams p) {
    extern __shared__ __align__(16) uint8_t mk_w[];
    __shared__ __align__(16) uint16_t sx[MK_K];
    __shared__ float red[MK_QUADS][2][64];
    __shared__ float sv[16];
    __shared__ int si[16];
    __shared__ int s_tok;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = gridDim.x;
    const int q0 = blockIdx.x * p.qpc;
    const int nq = min(p.qpc, p.nquads - q0);
    const bool one_batch = nq <= MK_QUADS;
    auto load_w = [&](int b0, int nb) {
        for (int i = tid; i < nb * MK_STEPS * 128; i += blockDim.x)
            ((uint4*)mk_w)[i] = ((const uint4*)(p.w + (size_t)(q0 + b0) * MK_STEPS * 2048))[i];
    };
    if (one_batch) load_w(0, nq);
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
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        float lm = -INFINITY, ls = 0.f;
        for (int b0 = 0; b0 < nq; b0 += MK_QUADS) {
            const int nb = min(MK_QUADS, nq - b0);
            if (!one_batch) {
                __syncthreads();
                load_w(b0, nb);
                __syncthreads();
            }
            if (lq < nb) {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                const uint8_t* wq = mk_w + (size_t)lq * MK_STEPS * 2048;
#pragma unroll
                for (int j = ph; j < MK_STEPS; j += 2) {
                    const uint4 v = *(const uint4*)(wq + j * 2048 + rg * 512 + lane * 16);
                    const uint32_t a[4] = {v.x, v.y, v.z, v.w};
                    const uint32_t b0_ = g == 0 ? *(const uint32_t*)(sx + j * 16 + 2 * t) : 0u;
                    const uint32_t b1_ = g == 0 ? *(const uint32_t*)(sx + j * 16 + 8 + 2 * t) : 0u;
                    mma_bf16(acc, a, b0_, b1_);
                }
                if (t == 0) {
                    red[lq][ph][rg * 16 + g] = acc[0];
                    red[lq][ph][rg * 16 + g + 8] = acc[2];
                }
            }
            __syncthreads();
            const int Rb = (q0 + b0) * 64 + tid;
            const bool liveb = tid < nb * 64 && Rb < p.vloc;
            float v = -INFINITY;
            if (tid < 128 && liveb) {
                const float mk = red[tid >> 6][0][tid & 63] + red[tid >> 6][1][tid & 63];
                float l;
                if (one_batch) {
                    l = lg[0];
#pragma unroll
                    for (int k = 1; k < 8; ++k) if (k == r) l = lg[k];
                } else {
                    l = p.logits[(size_t)r * p.vpad + Rb];
                }
                v = l + mk;
            }
            float key = v;
            if (sampling && warp < 4) {
                if (liveb) p.vq[(size_t)r * p.vpad + Rb] = v;
                mk_lse(lm, ls, v * inv_t, v == -INFINITY ? 0.f : 1.f);
                const float gk = v * inv_t + rng_gumbel(rng_bits(sp, RNG_DRAFT, (unsigned)(Rb + p.offset), (unsigned)r));
                key = v == -INFINITY ? v : gk;
            }
            if (tid < 128 && liveb && mk_better(key, Rb + p.offset, bv, bi)) { bv = key; bi = Rb + p.offset; }
        }
        if (sampling && warp < 4) {
            __syncwarp();
            for (int o = 16; o > 0; o >>= 1) {
                const float om = __shfl_xor_sync(0xffffffff, lm, o), os = __shfl_xor_sync(0xffffffff, ls, o);
                mk_lse(lm, ls, om, os);
            }
            if (lane == 0) p.qstat[(size_t)r * G * 4 + blockIdx.x * 4 + warp] = make_float2(lm, ls);
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
            p.cand[par * G + blockIdx.x] = make_float2(sv[0], __int_as_float(si[0]));
            __threadfence();
            atomicAdd(&p.bar[0], 1u);
            const unsigned target = (unsigned)G * (r + 1);
            unsigned v;
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p.bar) : "memory");
            } while (v < target);
        }
        __syncthreads();
        if (r == 0) pdl_launch();

        bv = -INFINITY;
        bi = 0x7fffffff;
        for (int c = tid; c < G; c += blockDim.x) {
            const float2 cv = __ldcg(p.cand + par * G + c);
            if (mk_better(cv.x, __float_as_int(cv.y), bv, bi)) { bv = cv.x; bi = __float_as_int(cv.y); }
        }
        __syncwarp();
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, o);
            const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            const bool t = mk_better(ov, oi, bv, bi);
            bv = t ? ov : bv;
            bi = t ? oi : bi;
        }
        if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            bv = sv[0];
            bi = si[0];
            for (int w = 1; w < (int)(blockDim.x >> 5); ++w)
                if (mk_better(sv[w], si[w], bv, bi)) { bv = sv[w]; bi = si[w]; }
            if (blockIdx.x == 0 && lane < p.tp)
                asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p.recv[lane] + (par * 8 + p.rank) * 4), "r"(__float_as_uint(bv)), "r"(sq),
                             "r"((unsigned)bi), "r"(sq) : "memory");
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
            *p.seq = seq0 + rounds;
            __threadfence();
        }
    }
}

extern "C" __global__ void spec_draft_meta(Meta* meta, const int* state, int block, int noise) {
    pdl_launch();
    pdl_wait();
    if (threadIdx.x == 0) {
        meta->m = block;
        meta->tokens[0] = state[0];
        for (int i = 1; i < block; ++i) meta->tokens[i] = noise;
    }
}

extern "C" __global__ void spec_verify_meta(Meta* meta, const int* drafts, int n) {
    pdl_launch();
    pdl_wait();
    if (threadIdx.x == 0) {
        meta->m = n;
        for (int i = 0; i < n; ++i) meta->tokens[i] = drafts[i];
    }
}

extern "C" __global__ void spec_accept(Meta* meta, const int* drafts, const int* next, int block, int* out, int* count,
                                       int cap, int* state, int dblk, int noise, const int* verdict, int4* snap) {
    pdl_wait();
    pdl_launch();
    if (threadIdx.x != 0) return;
    const int c = verdict[0];
    const int tok = verdict[1];
    int n = *count;
    for (int i = 1; i <= c; ++i) out[(n++) % cap] = drafts[i];
    out[(n++) % cap] = tok;
    *count = n;
    state[0] = tok;
    state[1] += 1;
    state[2] += c;
    state[3] += c + 1;
    if (snap) snap[(state[1] - 1) & 1] = make_int4(state[0], state[1], state[2], state[3]);
    meta->pos += c + 1;
    if (dblk > 0) {
        meta->m = dblk;
        meta->tokens[0] = tok;
        for (int i = 1; i < dblk; ++i) meta->tokens[i] = noise;
    }
}

extern "C" __global__ void __launch_bounds__(256) moe_route_tokens(int M, const float* scores, const float* bias, int n_exp, int topk, float route_scale,
                                                                   int* route_ids, float* route_w) {
    pdl_wait();
    pdl_launch();
    __shared__ int sel[8][8];
    __shared__ float selw[8][8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, m = blockIdx.x * 8 + warp;
    if (m < M) route_token(m, lane, scores, bias, n_exp, topk, route_scale, sel[warp], selw[warp], route_ids, route_w);
}

extern "C" __global__ void rows_i2f(const int* src, float* dst, int m, int n, int r0, int r1) {
    pdl_wait();
    pdl_launch();
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)m * n) return;
    const int row = (int)(i / n);
    dst[i] = (row >= r0 && row < r1) ? (float)src[i] : 0.f;
}
extern "C" __global__ void rows_f2i(const float* src, int* dst, long long count) {
    pdl_wait();
    pdl_launch();
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < count) dst[i] = (int)src[i];
}
