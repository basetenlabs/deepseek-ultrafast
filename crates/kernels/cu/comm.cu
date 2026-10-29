#include "common.cuh"
#include "hcepi.cuh"

struct Meta {
    int pos;
    int m;
    int tokens[16];
};

#define MAX_TP 8
#define SENT 0x80000000u

DEV uint32_t comm_epoch_next(uint32_t state) { return state >= 0xFFFFFFFEu ? 0u : state + 1u; }
DEV int comm_stage(uint32_t e) { return (int)(e % 3u); }
DEV int comm_prev_stage(int stage) { return stage == 0 ? 2 : stage - 1; }
enum { CM_SUM = 0, CM_GATHER = 1, CM_SUM_HCPOST = 2 };
#define CM_SRC16 0x80
#define CM_RS16 0x40

struct CommParams {
    float* recv[MAX_TP];
    uint32_t* state;
    int tp, rank, cap;
    const float* src;
    int ld_src;
    int n_per_token;
    int mode;
    int round_bf16;
    void* out;
    int ld_out;
    int out_f32;
    const bf16* res;
    const float* post;
    const float* comb;
    int nsum;
    const float* src2;
    int epi;
    HcEpi e;
    HcMix mx;
    int mx_on;
};

DEV void st_remote_v4(float* p, float4 v) {
    asm volatile("st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
DEV float4 ld_poll_v4(const float* p) {
    float4 v;
    asm volatile("ld.relaxed.sys.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
    return v;
}
DEV bool arrived(float4 v) {
    return __float_as_uint(v.x) != SENT && __float_as_uint(v.y) != SENT && __float_as_uint(v.z) != SENT &&
           __float_as_uint(v.w) != SENT;
}
DEV float canon(float x) { return __float_as_uint(x) == SENT ? 0.f : x; }
DEV uint32_t pk_bf2(float a, float b) {
    uint32_t lo = __bfloat16_as_ushort(__float2bfloat16(a)), hi = __bfloat16_as_ushort(__float2bfloat16(b));
    if (lo == 0x8000u) lo = 0;
    if (hi == 0x8000u) hi = 0;
    return lo | (hi << 16);
}
DEV bool arr_bf2(uint32_t w) { return (w & 0xffffu) != 0x8000u && (w >> 16) != 0x8000u; }

struct HcIn {
    float cb[4][4], po[4], rr[4][4];
};
template <typename P>
DEV void hc_in_load(const P& p, int m, int c, int dim, HcIn& in, const float* post = nullptr, const float* comb = nullptr) {
    if (!comb) comb = p.comb + m * 16;
    if (!post) post = p.post + m * 4;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float4 v = *(const float4*)(comb + h * 4);
        in.cb[h][0] = v.x; in.cb[h][1] = v.y; in.cb[h][2] = v.z; in.cb[h][3] = v.w;
    }
    {
        const float4 v = *(const float4*)post;
        in.po[0] = v.x; in.po[1] = v.y; in.po[2] = v.z; in.po[3] = v.w;
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const uint2 v = *(const uint2*)(p.res + ((size_t)m * 4 + h) * dim + c);
        in.rr[h][0] = __uint_as_float(v.x << 16); in.rr[h][1] = __uint_as_float(v.x & 0xffff0000u);
        in.rr[h][2] = __uint_as_float(v.y << 16); in.rr[h][3] = __uint_as_float(v.y & 0xffff0000u);
    }
}

template <typename P>
DEV void sum_out(const P& p, int m, int c, int npt, const float acc[4], bool dots, const float* fns, uint64_t* fbar, const HcIn* pin = nullptr, const float* pre = nullptr) {
    if ((p.mode & 0x3f) == CM_SUM) {
        for (int k = 0; k < 4; ++k) {
            const size_t o = (size_t)m * p.ld_out + c + k;
            if (p.out_f32) ((float*)p.out)[o] = acc[k];
            else ((bf16*)p.out)[o] = __float2bfloat16(acc[k]);
        }
        return;
    }
    const int dim = npt;
    float hn[4][4];
    HcIn lin;
    if (!pin) hc_in_load(p, m, c, dim, lin);
    const HcIn& in = pin ? *pin : lin;
    const auto& cb = in.cb;
    const auto& po = in.po;
    const auto& rr = in.rr;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x = __bfloat162float(__float2bfloat16(acc[k]));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s2 = 0.f;
#pragma unroll
            for (int h = 0; h < 4; ++h) s2 += cb[h][j] * rr[h][k];
            hn[j][k] = __bfloat162float(__float2bfloat16(po[j] * x + s2));
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint2 o;
        o.x = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][0])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][1])) << 16);
        o.y = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][2])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][3])) << 16);
        *(uint2*)((bf16*)p.out + ((size_t)m * 4 + j) * dim + c) = o;
    }
    if (p.epi) {
        if (dots) hc_fn_wait(fbar);
        float a[NP];
        HcEpi e = p.e;
        if (pre && e.pre) e.pre = pre - m * HC;
        hc_epi_thread(e, m, c, hn, dots ? fns : nullptr, a);
        hc_epi_reduce(e, m, c / SLICE, dim / SLICE, a);
    }
}

extern "C" __global__ void __launch_bounds__(64) comm_push(const Meta* meta, const __grid_constant__ CommParams p) {
    __shared__ uint32_t s_epoch, s_l0, s_l1;
    extern __shared__ __align__(16) float fns[];
    __shared__ __align__(8) uint64_t fbar;
    const bool dots = p.epi && p.e.fn;
    if (dots) hc_fn_issue(p.e.fn, p.e.dim, (blockIdx.x * 256) % p.e.dim, fns, &fbar);
    __shared__ __align__(16) float s_pre[4], s_post[4], s_comb[16];
    __shared__ float s_mx[NP];
    const bool mxo = p.mode == CM_SUM_HCPOST && p.mx_on;
    if (mxo) {
        const int mt = (blockIdx.x * blockDim.x * 4) / p.n_per_token;
        if (threadIdx.x < 32 && mt < meta->m) hc_mix_token(p.mx, mt, threadIdx.x, s_mx, s_pre, s_post, s_comb);
    }
    pdl_wait();
    pdl_launch();
    const int b = blockIdx.x;
    const int M = meta->m;
    const int npt = p.n_per_token;
    const int n4 = M * npt / 4;
    const int i4 = b * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) {
        const uint32_t* st = p.state + b * 4;
        s_epoch = comm_epoch_next(st[0]);
        s_l0 = st[1];
        s_l1 = st[2];
    }
    __syncthreads();
    const uint32_t e = s_epoch;
    const int stage = comm_stage(e), prev = comm_prev_stage(stage);
    const size_t slot = (size_t)p.tp * p.cap;
    if (i4 < n4) {
        const int i = i4 * 4, m = i / npt, c = i % npt;
        float4 v;
        if (p.nsum > 1 || p.src2) {
            v = make_float4(0.f, 0.f, 0.f, 0.f);
            const int ns = p.nsum > 1 ? p.nsum : 1;
            for (int j = 0; j < ns; ++j) {
                const float4 a = *(const float4*)(p.src + (size_t)(m * ns + j) * p.ld_src + c);
                v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
            }
            if (p.src2) {
                const float4 a = *(const float4*)(p.src2 + (size_t)m * p.ld_src + c);
                v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
            }
        } else {
            v = *(const float4*)(p.src + (size_t)m * p.ld_src + c);
        }
        if (p.round_bf16 && p.mode != CM_GATHER) {
            const uint32_t w0 = pk_bf2(v.x, v.y), w1 = pk_bf2(v.z, v.w);
            const size_t wi = (size_t)(i / 256) * 256 + (i % 256) / 2;
            for (int q = 0; q < p.tp; ++q)
                asm volatile("st.relaxed.sys.global.v2.u32 [%0], {%1,%2};" ::"l"((uint32_t*)(p.recv[q] + stage * slot + (size_t)p.rank * p.cap) + wi), "r"(w0), "r"(w1) : "memory");
        } else {
            v = make_float4(canon(v.x), canon(v.y), canon(v.z), canon(v.w));
            for (int q = 0; q < p.tp; ++q) st_remote_v4(p.recv[q] + stage * slot + (size_t)p.rank * p.cap + i, v);
        }
    }
    if (mxo) {
        const int mt = (b * blockDim.x * 4) / npt;
        if (mt < M && (b * blockDim.x * 4) % npt == 0 && threadIdx.x < 24) {
            const int t = threadIdx.x;
            if (t < 16) p.mx.comb[mt * 16 + t] = s_comb[t];
            else if (t < 20) p.mx.pre[mt * 4 + t - 16] = s_pre[t - 16];
            else p.mx.post[mt * 4 + t - 20] = s_post[t - 20];
        }
    }
    HcIn hin;
    const bool hcp = p.mode == CM_SUM_HCPOST && i4 < n4;
    if (hcp) hc_in_load(p, (i4 * 4) / npt, (i4 * 4) % npt, npt, hin, mxo ? s_post : nullptr, mxo ? s_comb : nullptr);
    float* rb = p.recv[p.rank] + stage * slot;
    if (i4 < n4) {
        const int i = i4 * 4, m = i / npt, c = i % npt;
        float4 parts[MAX_TP];
        if (p.round_bf16 && p.mode != CM_GATHER) {
            const size_t wi = (size_t)(i / 256) * 256 + (i % 256) / 2;
            uint2 w[MAX_TP];
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                if (s < p.tp) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(w[s].x), "=r"(w[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
            unsigned miss = 0;
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                if (s < p.tp && !(arr_bf2(w[s].x) && arr_bf2(w[s].y))) miss |= 1u << s;
            while (miss) {
#pragma unroll
                for (int s = 0; s < MAX_TP; ++s)
                    if (miss & (1u << s)) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(w[s].x), "=r"(w[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
#pragma unroll
                for (int s = 0; s < MAX_TP; ++s)
                    if ((miss & (1u << s)) && arr_bf2(w[s].x) && arr_bf2(w[s].y)) miss &= ~(1u << s);
            }
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                parts[s] = make_float4(__uint_as_float(w[s].x << 16), __uint_as_float(w[s].x & 0xffff0000u), __uint_as_float(w[s].y << 16), __uint_as_float(w[s].y & 0xffff0000u));
        } else {
#pragma unroll
        for (int s = 0; s < MAX_TP; ++s)
            if (s < p.tp) parts[s] = ld_poll_v4(rb + (size_t)s * p.cap + i);
        unsigned miss = 0;
#pragma unroll
        for (int s = 0; s < MAX_TP; ++s)
            if (s < p.tp && !arrived(parts[s])) miss |= 1u << s;
        while (miss) {
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                if (miss & (1u << s)) parts[s] = ld_poll_v4(rb + (size_t)s * p.cap + i);
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                if ((miss & (1u << s)) && arrived(parts[s])) miss &= ~(1u << s);
        }
        }
        if (p.mode == CM_GATHER) {
            for (int s = 0; s < p.tp; ++s) {
                const float vv[4] = {parts[s].x, parts[s].y, parts[s].z, parts[s].w};
                for (int k = 0; k < 4; ++k) {
                    const size_t o = (size_t)m * p.ld_out + s * npt + c + k;
                    if (p.out_f32) ((float*)p.out)[o] = vv[k];
                    else ((bf16*)p.out)[o] = __float2bfloat16(vv[k]);
                }
            }
        } else {
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (int s = 0; s < p.tp; ++s) {
                float vv[4] = {parts[s].x, parts[s].y, parts[s].z, parts[s].w};
                for (int k = 0; k < 4; ++k) {
                    if (p.round_bf16) vv[k] = __bfloat162float(__float2bfloat16(vv[k]));
                    acc[k] += vv[k];
                }
            }
            sum_out(p, m, c, npt, acc, dots, fns, &fbar, hcp ? &hin : nullptr, mxo ? s_pre : nullptr);
        }
    }
    {
        float* pb = p.recv[p.rank] + prev * slot;
        for (int j = s_l0 + threadIdx.x; j < (int)s_l1; j += blockDim.x)
            for (int s = 0; s < p.tp; ++s) *(float4*)(pb + (size_t)s * p.cap + (size_t)j * 4) = make_float4(-0.f, -0.f, -0.f, -0.f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t* st = p.state + b * 4;
        st[0] = e;
        const int lo = b * blockDim.x, hi = min(n4, lo + (int)blockDim.x);
        st[1] = lo;
        st[2] = hi > lo ? hi : lo;
    }
}

extern "C" __global__ void comm_init(float* recv, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        recv[i] = -0.f;
}

#define SENT2 0x80008000u
struct Comm2Params {
    float* rs[MAX_TP];
    uint32_t* ag[MAX_TP];
    uint32_t* state;
    int tp, rank, cap;
    const float* src;
    int ld_src;
    int n_per_token;
    int mode;
    int round_bf16;
    void* out;
    int ld_out;
    int out_f32;
    const bf16* res;
    const float* post;
    const float* comb;
    int nsum;
    const float* src2;
    int epi;
    HcEpi e;
};
DEV uint2 ld_poll_v2(const uint32_t* p) {
    uint2 v;
    asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
    return v;
}
DEV bool arrived2(uint32_t w) { return (w & 0xffffu) != 0x8000u && (w >> 16) != 0x8000u; }
DEV uint32_t pack_bf2(float a, float b) {
    uint32_t lo = __bfloat16_as_ushort(__float2bfloat16(a)), hi = __bfloat16_as_ushort(__float2bfloat16(b));
    if (lo == 0x8000u) lo = 0;
    if (hi == 0x8000u) hi = 0;
    return lo | (hi << 16);
}
DEV float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
DEV float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

template <int TP, int C2_K = 16>
DEV void comm_push2_body(const Meta* meta, const Comm2Params& p) {
    constexpr int NVMAX = 8;
    __shared__ uint32_t s_epoch[NVMAX], s_l0[NVMAX], s_l1[NVMAX], s_own[NVMAX];
    extern __shared__ __align__(16) float fns[];
    __shared__ __align__(8) uint64_t fbar;
    const bool dots = p.epi && p.e.fn;
    pdl_wait();
    pdl_launch();
    const int npt = p.n_per_token;
    const int n = meta->m * npt;
    const int nb = (n + C2_K * 256 - 1) / (C2_K * 256);
    const int G = gridDim.x, nv = (nb - (int)blockIdx.x + G - 1) / G;
    if (nv <= 0) return;
    if (threadIdx.x < nv) {
        const int v = blockIdx.x + threadIdx.x * G;
        const uint32_t* st = p.state + v * 4;
        s_epoch[threadIdx.x] = comm_epoch_next(st[0]);
        s_l0[threadIdx.x] = st[1];
        s_l1[threadIdx.x] = st[2];
        s_own[threadIdx.x] = st[3];
    }
    __syncthreads();
    const size_t cap2 = (size_t)p.cap / 2;
    for (int j = 0; j < nv; ++j) {
        const int v = blockIdx.x + j * G, base = v * C2_K * 256, owner = (int)((long long)v * TP / nb);
        const int stage = comm_stage(s_epoch[j]);
        for (int sl = 0; sl < C2_K; ++sl) {
            const int i = base + sl * 256 + threadIdx.x * 4;
            if (i >= n) break;
            const int m = i / npt, c = i % npt;
            float4 val;
            auto row4 = [&](const float* base, size_t row) -> float4 {
                if (p.mode & CM_SRC16) {
                    const uint2 w = *(const uint2*)((const bf16*)base + row * p.ld_src + c);
                    return make_float4(bf_lo(w.x), bf_hi(w.x), bf_lo(w.y), bf_hi(w.y));
                }
                return *(const float4*)(base + row * p.ld_src + c);
            };
            if (p.nsum > 1 || p.src2) {
                val = make_float4(0.f, 0.f, 0.f, 0.f);
                const int ns = p.nsum > 1 ? p.nsum : 1;
                for (int q = 0; q < ns; ++q) {
                    const float4 a = row4(p.src, (size_t)(m * ns + q));
                    val.x += a.x; val.y += a.y; val.z += a.z; val.w += a.w;
                }
                if (p.src2) {
                    const float4 a = row4(p.src2, (size_t)m);
                    val.x += a.x; val.y += a.y; val.z += a.z; val.w += a.w;
                }
            } else {
                val = row4(p.src, (size_t)m);
            }
            if (p.round_bf16) {
                val.x = __bfloat162float(__float2bfloat16(val.x)); val.y = __bfloat162float(__float2bfloat16(val.y));
                val.z = __bfloat162float(__float2bfloat16(val.z)); val.w = __bfloat162float(__float2bfloat16(val.w));
            }
            if (p.mode & CM_RS16) {
                const uint2 r2 = make_uint2(pack_bf2(val.x, val.y), pack_bf2(val.z, val.w));
                const uint32_t ox = __shfl_xor_sync(0xffffffff, r2.x, 1), oy = __shfl_xor_sync(0xffffffff, r2.y, 1);
                if (!(threadIdx.x & 1))
                    asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"((uint32_t*)p.rs[owner] + ((size_t)stage * TP + p.rank) * p.cap + i / 2),
                                 "r"(r2.x), "r"(r2.y), "r"(ox), "r"(oy) : "memory");
                continue;
            }
            val = make_float4(canon(val.x), canon(val.y), canon(val.z), canon(val.w));
            st_remote_v4(p.rs[owner] + ((size_t)stage * TP + p.rank) * p.cap + i, val);
        }
    }
    constexpr int C2_B = 2;
    const int xf = p.mode >> 8;
    if (xf == 2) return;
    for (int j = 0; j < nv; ++j) {
        const int v = blockIdx.x + j * G, base = v * C2_K * 256, owner = (int)((long long)v * TP / nb);
        if (p.rank != owner) continue;
        const int stage = comm_stage(s_epoch[j]);
        for (int s0 = 0; s0 < C2_K; s0 += C2_B) {
            float4 u[C2_B][TP];
            bool ok = true;
            if (p.mode & CM_RS16) {
                uint2 w[C2_B][TP];
#pragma unroll
                for (int q = 0; q < C2_B; ++q) {
                    const int i = base + (s0 + q) * 256 + threadIdx.x * 4;
                    const bool in = s0 + q < C2_K && i < n;
#pragma unroll
                    for (int r = 0; r < TP; ++r) w[q][r] = in ? ld_poll_v2((const uint32_t*)p.rs[p.rank] + ((size_t)stage * TP + r) * p.cap + i / 2) : make_uint2(0, 0);
                }
                do {
                    ok = true;
#pragma unroll
                    for (int q = 0; q < C2_B; ++q) {
                        const int i = base + (s0 + q) * 256 + threadIdx.x * 4;
                        const bool in = s0 + q < C2_K && i < n;
#pragma unroll
                        for (int r = 0; r < TP; ++r)
                            if (in && ((w[q][r].x >> 16) == 0x8000u || (w[q][r].y >> 16) == 0x8000u)) {
                                w[q][r] = ld_poll_v2((const uint32_t*)p.rs[p.rank] + ((size_t)stage * TP + r) * p.cap + i / 2);
                                ok = false;
                            }
                    }
                } while (!ok);
#pragma unroll
                for (int q = 0; q < C2_B; ++q)
#pragma unroll
                    for (int r = 0; r < TP; ++r) u[q][r] = make_float4(bf_lo(w[q][r].x), bf_hi(w[q][r].x), bf_lo(w[q][r].y), bf_hi(w[q][r].y));
            } else {
#pragma unroll
            for (int q = 0; q < C2_B; ++q) {
                const int i = base + (s0 + q) * 256 + threadIdx.x * 4;
                const bool in = s0 + q < C2_K && i < n;
#pragma unroll
                for (int r = 0; r < TP; ++r) u[q][r] = in ? ld_poll_v4(p.rs[p.rank] + ((size_t)stage * TP + r) * p.cap + i) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
            }
            do {
                if (p.mode & CM_RS16) break;
                ok = true;
#pragma unroll
                for (int q = 0; q < C2_B; ++q) {
                    const int i = base + (s0 + q) * 256 + threadIdx.x * 4;
                    const bool in = s0 + q < C2_K && i < n;
#pragma unroll
                    for (int r = 0; r < TP; ++r)
                        if (in && !arrived(u[q][r])) {
                            u[q][r] = ld_poll_v4(p.rs[p.rank] + ((size_t)stage * TP + r) * p.cap + i);
                            ok = false;
                        }
                }
            } while (!ok);
#pragma unroll
            for (int q = 0; q < C2_B; ++q) {
                const int i = base + (s0 + q) * 256 + threadIdx.x * 4;
                if (s0 + q >= C2_K || i >= n) break;
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int r = 0; r < TP; ++r) { acc[0] += u[q][r].x; acc[1] += u[q][r].y; acc[2] += u[q][r].z; acc[3] += u[q][r].w; }
                const uint2 rr = make_uint2(pack_bf2(acc[0], acc[1]), pack_bf2(acc[2], acc[3]));
                const uint32_t ox = __shfl_xor_sync(0xffffffff, rr.x, 1), oy = __shfl_xor_sync(0xffffffff, rr.y, 1);
                if (!(threadIdx.x & 1))
#pragma unroll
                    for (int d = 0; d < TP; ++d)
                        asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p.ag[d] + (size_t)stage * cap2 + i / 2), "r"(rr.x), "r"(rr.y), "r"(ox), "r"(oy) : "memory");
            }
        }
    }
    if (xf == 1) return;
    for (int j = 0; j < nv; ++j) {
        const int v = blockIdx.x + j * G, base = v * C2_K * 256;
        const int stage = comm_stage(s_epoch[j]);
        for (int s0 = 0; s0 < C2_K; s0 += C2_B) {
            if (base + s0 * 256 >= n) break;
            uint2 u[C2_B];
#pragma unroll
            for (int q = 0; q < C2_B; ++q) {
                const int i = base + (s0 + q) * 256 + threadIdx.x * 4;
                u[q] = s0 + q < C2_K && i < n ? ld_poll_v2(p.ag[p.rank] + (size_t)stage * cap2 + i / 2) : make_uint2(0, 0);
            }
            bool ok;
            do {
                ok = true;
#pragma unroll
                for (int q = 0; q < C2_B; ++q) {
                    const int i = base + (s0 + q) * 256 + threadIdx.x * 4;
                    if (s0 + q < C2_K && i < n && (!arrived2(u[q].x) || !arrived2(u[q].y))) {
                        u[q] = ld_poll_v2(p.ag[p.rank] + (size_t)stage * cap2 + i / 2);
                        ok = false;
                    }
                }
            } while (!ok);
#pragma unroll
            for (int q = 0; q < C2_B; ++q) {
                const int i0 = base + (s0 + q) * 256;
                if (s0 + q >= C2_K || i0 >= n) break;
                const int i = i0 + threadIdx.x * 4;
                const int m = i / npt, c = i % npt;
                if (dots) {
                    __syncthreads();
                    hc_fn_issue(p.e.fn, p.e.dim, c - c % 256, fns, &fbar);
                }
                const float acc[4] = {bf_lo(u[q].x), bf_hi(u[q].x), bf_lo(u[q].y), bf_hi(u[q].y)};
                sum_out(p, m, c, npt, acc, dots, fns, &fbar);
            }
        }
    }
    for (int j = 0; j < nv; ++j) {
        const int prev = comm_prev_stage(comm_stage(s_epoch[j]));
        uint32_t* ab = p.ag[p.rank] + (size_t)prev * cap2;
        for (int q = s_l0[j] + threadIdx.x; q < (int)s_l1[j]; q += blockDim.x) ab[q] = SENT2;
        if (s_own[j])
            for (int r = 0; r < TP; ++r) {
                float* rb = p.rs[p.rank] + ((size_t)prev * TP + r) * p.cap;
                if (p.mode & CM_RS16)
                    for (int q = s_l0[j] + threadIdx.x; q < (int)s_l1[j]; q += blockDim.x) ((uint32_t*)rb)[q] = SENT2;
                else
                    for (int q = s_l0[j] + threadIdx.x; q < (int)s_l1[j]; q += blockDim.x) *(float2*)(rb + 2 * q) = make_float2(-0.f, -0.f);
            }
    }
    __syncthreads();
    if (threadIdx.x < nv) {
        const int jj = threadIdx.x, v = blockIdx.x + jj * G, base = v * C2_K * 256;
        uint32_t* st = p.state + v * 4;
        st[0] = s_epoch[jj];
        const int lo = base / 2, hi = min(n, base + C2_K * 256) / 2;
        st[1] = lo;
        st[2] = hi > lo ? hi : lo;
        st[3] = p.rank == (int)((long long)v * TP / nb);
    }
}

#define C2_ENTRY(TP) \
    extern "C" __global__ void __launch_bounds__(64, COMM_MIN_CTAS_PER_SM) comm_push2_tp##TP(const Meta* meta, const __grid_constant__ Comm2Params p) { comm_push2_body<TP>(meta, p); }
C2_ENTRY(2)
C2_ENTRY(4)
C2_ENTRY(8)

#define C2_K1_ENTRY(TP) \
    extern "C" __global__ void __launch_bounds__(64, COMM_MIN_CTAS_PER_SM) comm_push2_k1_tp##TP(const Meta* meta, const __grid_constant__ Comm2Params p) { comm_push2_body<TP, 1>(meta, p); }
C2_K1_ENTRY(2)
C2_K1_ENTRY(4)

extern "C" __global__ void comm_init2(uint32_t* buf, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) buf[i] = SENT2;
}

struct NvlsParams {
    CommParams c;
    const unsigned* mixf;
    const int* sg_perm;
    const float* sg_w;
    const int* sg_rec;
    const __nv_bfloat16* sg_gamma;
    uint8_t* sg_img;
    int sg_early;
    int _sgpad;
};
DEV float4 nv_contrib(const CommParams& p, int m, int c) {
    if (!(p.nsum > 1 || p.src2)) return *(const float4*)(p.src + (size_t)m * p.ld_src + c);
    const int ns = p.nsum > 1 ? p.nsum : 1;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (ns <= 8) {
        float4 a[8], a2 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (j < ns) a[j] = *(const float4*)(p.src + (size_t)(m * ns + j) * p.ld_src + c);
        if (p.src2) a2 = *(const float4*)(p.src2 + (size_t)m * p.ld_src + c);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (j < ns) { v.x += a[j].x; v.y += a[j].y; v.z += a[j].z; v.w += a[j].w; }
        if (p.src2) { v.x += a2.x; v.y += a2.y; v.z += a2.z; v.w += a2.w; }
        return v;
    }
    for (int j = 0; j < ns; ++j) {
        const float4 a = *(const float4*)(p.src + (size_t)(m * ns + j) * p.ld_src + c);
        v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
    }
    if (p.src2) {
        const float4 a = *(const float4*)(p.src2 + (size_t)m * p.ld_src + c);
        v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
    }
    return v;
}

DEV void comm_nvls_body(const Meta* meta, const NvlsParams& q) {
    const CommParams& p = q.c;
    __shared__ uint32_t s_epoch, s_l0, s_l1;
    extern __shared__ __align__(16) float fns[];
    __shared__ __align__(8) uint64_t fbar;
    const bool dots = p.epi && p.e.fn;
    if (dots) hc_fn_issue(p.e.fn, p.e.dim, (blockIdx.x * 256) % p.e.dim, fns, &fbar);
    __shared__ __align__(16) float s_pre[4], s_post[4], s_comb[16];
    __shared__ float s_mx[NP];
    const bool mxo = p.mode == CM_SUM_HCPOST && p.mx_on;
    if (mxo) {
        const int mt = (blockIdx.x * blockDim.x * 4) / p.n_per_token;
        if (threadIdx.x < 32 && mt < meta->m) hc_mix_token(p.mx, mt, threadIdx.x, s_mx, s_pre, s_post, s_comb);
    }
    pdl_wait();
    pdl_launch();
    const int b = blockIdx.x;
    const int npt = p.n_per_token;
    const int i4 = b * blockDim.x + threadIdx.x;
    const bool pre = npt % 256 == 0;
    float4 vpre = make_float4(0.f, 0.f, 0.f, 0.f);
    if (pre) vpre = nv_contrib(p, (i4 * 4) / npt, (i4 * 4) % npt);
    if (threadIdx.x == 0) {
        const uint32_t* st = p.state + b * 4;
        s_epoch = comm_epoch_next(st[0]);
        s_l0 = st[1];
        s_l1 = st[2];
    }
    const int M = meta->m;
    const int n4 = M * npt / 4;
    __syncthreads();
    const uint32_t e = s_epoch;
    const int stage = comm_stage(e), prev = comm_prev_stage(stage);
    const size_t slot = (size_t)p.tp * p.cap;
    float* const mine = p.recv[p.rank];
    const bool x16 = p.round_bf16 && p.mode != CM_GATHER;
    if (i4 < n4) {
        const int i = i4 * 4, m = i / npt, c = i % npt;
        float4 v = pre ? vpre : nv_contrib(p, m, c);
        if (x16) {
            const uint32_t w0 = pk_bf2(v.x, v.y), w1 = pk_bf2(v.z, v.w);
            const size_t wi = (size_t)(i / 256) * 256 + (i % 256) / 2;
            for (int r = 0; r < p.tp; ++r)
                asm volatile("st.relaxed.sys.global.v2.u32 [%0], {%1,%2};" ::"l"((uint32_t*)(p.recv[r] + stage * slot + (size_t)p.rank * p.cap) + wi), "r"(w0), "r"(w1) : "memory");
        } else {
            v = make_float4(canon(v.x), canon(v.y), canon(v.z), canon(v.w));
            for (int r = 0; r < p.tp; ++r) st_remote_v4(p.recv[r] + stage * slot + (size_t)p.rank * p.cap + i, v);
        }
    }
    float* pb = mine + prev * slot;
    for (int j = s_l0 + threadIdx.x; j < (int)s_l1; j += blockDim.x)
        for (int s = 0; s < p.tp; ++s) *(float4*)(pb + (size_t)s * p.cap + (size_t)j * 4) = make_float4(-0.f, -0.f, -0.f, -0.f);
    if (threadIdx.x == 0) {
        uint32_t* st = p.state + b * 4;
        st[0] = e;
        const int lo = b * blockDim.x, hi = min(n4, lo + (int)blockDim.x);
        st[1] = lo;
        st[2] = hi > lo ? hi : lo;
    }
    if (mxo) {
        const int mt = (b * blockDim.x * 4) / npt;
        if (mt < M && (b * blockDim.x * 4) % npt == 0 && threadIdx.x < 24) {
            const int t = threadIdx.x;
            if (t < 16) p.mx.comb[mt * 16 + t] = s_comb[t];
            else if (t < 20) p.mx.pre[mt * 4 + t - 16] = s_pre[t - 16];
            else p.mx.post[mt * 4 + t - 20] = s_post[t - 20];
        }
    }
    HcIn hin;
    const bool hcp = p.mode == CM_SUM_HCPOST && i4 < n4;
    if (hcp) hc_in_load(p, (i4 * 4) / npt, (i4 * 4) % npt, npt, hin, mxo ? s_post : nullptr, mxo ? s_comb : nullptr);
    if (i4 < n4) {
        const int i = i4 * 4, m = i / npt, c = i % npt;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const float* rb = mine + stage * slot;
        float4 parts[MAX_TP];
        if (x16) {
            const size_t wi = (size_t)(i / 256) * 256 + (i % 256) / 2;
            uint2 w[MAX_TP];
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                if (s < p.tp) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(w[s].x), "=r"(w[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
            unsigned miss = 0;
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                if (s < p.tp && !(arr_bf2(w[s].x) && arr_bf2(w[s].y))) miss |= 1u << s;
            while (miss) {
#pragma unroll
                for (int s = 0; s < MAX_TP; ++s)
                    if (miss & (1u << s)) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(w[s].x), "=r"(w[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
#pragma unroll
                for (int s = 0; s < MAX_TP; ++s)
                    if ((miss & (1u << s)) && arr_bf2(w[s].x) && arr_bf2(w[s].y)) miss &= ~(1u << s);
            }
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                parts[s] = make_float4(__uint_as_float(w[s].x << 16), __uint_as_float(w[s].x & 0xffff0000u), __uint_as_float(w[s].y << 16), __uint_as_float(w[s].y & 0xffff0000u));
        } else {
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                if (s < p.tp) parts[s] = ld_poll_v4(rb + (size_t)s * p.cap + i);
            unsigned miss = 0;
#pragma unroll
            for (int s = 0; s < MAX_TP; ++s)
                if (s < p.tp && !arrived(parts[s])) miss |= 1u << s;
            while (miss) {
#pragma unroll
                for (int s = 0; s < MAX_TP; ++s)
                    if (miss & (1u << s)) parts[s] = ld_poll_v4(rb + (size_t)s * p.cap + i);
#pragma unroll
                for (int s = 0; s < MAX_TP; ++s)
                    if ((miss & (1u << s)) && arrived(parts[s])) miss &= ~(1u << s);
            }
        }
        for (int s = 0; s < p.tp; ++s) {
            float vv[4] = {parts[s].x, parts[s].y, parts[s].z, parts[s].w};
            for (int k = 0; k < 4; ++k) {
                if (p.round_bf16) vv[k] = __bfloat162float(__float2bfloat16(vv[k]));
                acc[k] += vv[k];
            }
        }
        sum_out(p, m, c, npt, acc, dots, fns, &fbar, hcp ? &hin : nullptr, mxo ? s_pre : nullptr);
    }
}
extern "C" __global__ void __launch_bounds__(64) comm_nvls_ucpush(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_nvls_body(meta, q); }

extern "C" __global__ void comm_fill_u32(uint32_t* buf, size_t n, uint32_t v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) buf[i] = v;
}
