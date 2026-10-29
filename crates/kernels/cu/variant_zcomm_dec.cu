#include "comm.cu"

template <bool RB, int NS, bool S2, int TPC = 4>
DEV void comm_dec_body(const Meta* meta, const NvlsParams& q) {
    const CommParams& p = q.c;
    constexpr bool X16 = RB;
    __shared__ uint32_t s_epoch, s_l0, s_l1;
    __shared__ float red[2][2];
    const int tid = threadIdx.x, b = blockIdx.x, npt = p.n_per_token;
    const int i4 = b * blockDim.x + tid, i = i4 * 4, m = i / npt, c = i % npt;
    pdl_wait();
    pdl_launch();
    const bool s2 = NS > 0 ? S2 : p.src2 != nullptr;
    const int ns = NS > 0 ? NS : (p.nsum > 1 ? p.nsum : 1);
    const bool one = NS > 0 ? (NS == 1 && !S2) : !(p.nsum > 1 || p.src2);
    constexpr int NR = NS > 0 ? NS : 8;
    float4 a[NR], a2 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int j = 0; j < NR; ++j)
        if (j < ns) a[j] = *(const float4*)(p.src + (size_t)(m * ns + j) * p.ld_src + c);
    if (s2) a2 = *(const float4*)(p.src2 + (size_t)m * p.ld_src + c);
    if (tid == 0) {
        const uint4 st = *(const uint4*)(p.state + b * 4);
        s_epoch = comm_epoch_next(st.x);
        s_l0 = st.y;
        s_l1 = st.z;
    }
    const int M = meta->m;
    const int n4 = M * npt / 4;
    __syncthreads();
    const uint32_t e = s_epoch;
    const int stage = comm_stage(e), prev = comm_prev_stage(stage);
    const size_t slot = (size_t)TPC * p.cap;
    float* const mine = p.recv[p.rank];
    const bool act = i4 < n4;
    float4 v;
    if (one) {
        v = a[0];
    } else {
        v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int j = 0; j < NR; ++j)
            if (j < ns) { v.x += a[j].x; v.y += a[j].y; v.z += a[j].z; v.w += a[j].w; }
        if (s2) { v.x += a2.x; v.y += a2.y; v.z += a2.z; v.w += a2.w; }
    }
    const size_t wi = (size_t)(i / 256) * 256 + (i % 256) / 2;
    if (act) {
        if (X16) {
            const uint32_t w0 = pk_bf2(v.x, v.y), w1 = pk_bf2(v.z, v.w);
#pragma unroll
            for (int r = 0; r < TPC; ++r)
                asm volatile("st.relaxed.sys.global.v2.u32 [%0], {%1,%2};" ::"l"((uint32_t*)(p.recv[r] + stage * slot + (size_t)p.rank * p.cap) + wi), "r"(w0), "r"(w1) : "memory");
        } else {
            v = make_float4(canon(v.x), canon(v.y), canon(v.z), canon(v.w));
#pragma unroll
            for (int r = 0; r < TPC; ++r) st_remote_v4(p.recv[r] + stage * slot + (size_t)p.rank * p.cap + i, v);
        }
    }
    {
        float* pb = mine + prev * slot;
        for (int j = s_l0 + tid; j < (int)s_l1; j += blockDim.x)
#pragma unroll
            for (int s = 0; s < TPC; ++s) *(float4*)(pb + (size_t)s * p.cap + (size_t)j * 4) = make_float4(-0.f, -0.f, -0.f, -0.f);
        if (tid == 0) {
            const int lo = b * blockDim.x, hi = min(n4, lo + (int)blockDim.x);
            *(uint4*)(p.state + b * 4) = make_uint4(e, (uint32_t)lo, (uint32_t)(hi > lo ? hi : lo), 0u);
        }
    }
    if (act) {
        HcIn hin;
        hc_in_load(p, m, c, npt, hin);
        float pr[4];
        const bool hp = p.e.pre != nullptr;
        if (hp) {
            const float4 t = *(const float4*)(p.e.pre + m * HC);
            pr[0] = t.x; pr[1] = t.y; pr[2] = t.z; pr[3] = t.w;
        }
        const float* rb = mine + stage * slot;
        float4 parts[TPC];
        if (X16) {
            uint2 w[TPC];
#pragma unroll
            for (int s = 0; s < TPC; ++s) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(w[s].x), "=r"(w[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
            unsigned miss = 0;
#pragma unroll
            for (int s = 0; s < TPC; ++s) if (!(arr_bf2(w[s].x) && arr_bf2(w[s].y))) miss |= 1u << s;
            while (miss) {
#pragma unroll
                for (int s = 0; s < TPC; ++s)
                    if (miss & (1u << s)) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(w[s].x), "=r"(w[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
#pragma unroll
                for (int s = 0; s < TPC; ++s) if ((miss & (1u << s)) && arr_bf2(w[s].x) && arr_bf2(w[s].y)) miss &= ~(1u << s);
            }
#pragma unroll
            for (int s = 0; s < TPC; ++s) parts[s] = make_float4(__uint_as_float(w[s].x << 16), __uint_as_float(w[s].x & 0xffff0000u), __uint_as_float(w[s].y << 16), __uint_as_float(w[s].y & 0xffff0000u));
        } else {
#pragma unroll
            for (int s = 0; s < TPC; ++s) parts[s] = ld_poll_v4(rb + (size_t)s * p.cap + i);
            unsigned miss = 0;
#pragma unroll
            for (int s = 0; s < TPC; ++s) if (!arrived(parts[s])) miss |= 1u << s;
            while (miss) {
#pragma unroll
                for (int s = 0; s < TPC; ++s) if (miss & (1u << s)) parts[s] = ld_poll_v4(rb + (size_t)s * p.cap + i);
#pragma unroll
                for (int s = 0; s < TPC; ++s) if ((miss & (1u << s)) && arrived(parts[s])) miss &= ~(1u << s);
            }
        }
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < TPC; ++s) {
            float vv[4] = {parts[s].x, parts[s].y, parts[s].z, parts[s].w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (RB) vv[k] = __bfloat162float(__float2bfloat16(vv[k]));
                acc[k] += vv[k];
            }
        }
        const int dim = npt;
        float hn[4][4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float x = __bfloat162float(__float2bfloat16(acc[k]));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s2v = 0.f;
#pragma unroll
                for (int h = 0; h < 4; ++h) s2v += hin.cb[h][j] * hin.rr[h][k];
                hn[j][k] = __bfloat162float(__float2bfloat16(hin.po[j] * x + s2v));
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint2 o;
            o.x = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][0])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][1])) << 16);
            o.y = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][2])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][3])) << 16);
            *(uint2*)((bf16*)p.out + ((size_t)m * 4 + j) * dim + c) = o;
        }
        float xr[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float y = 0.f;
            if (hp) {
#pragma unroll
                for (int h = 0; h < HC; ++h) y += pr[h] * hn[h][k];
            } else {
                y = hn[0][k];
            }
            xr[k] = __bfloat162float(__float2bfloat16(y));
        }
        *(__nv_bfloat162*)(p.e.x_raw + (size_t)m * p.e.dim + c) = __floats2bfloat162_rn(xr[0], xr[1]);
        *(__nv_bfloat162*)(p.e.x_raw + (size_t)m * p.e.dim + c + 2) = __floats2bfloat162_rn(xr[2], xr[3]);
        float sh = 0.f, sx = 0.f;
#pragma unroll
        for (int h = 0; h < HC; ++h)
#pragma unroll
            for (int k = 0; k < 4; ++k) sh += hn[h][k] * hn[h][k];
#pragma unroll
        for (int k = 0; k < 4; ++k) sx += xr[k] * xr[k];
        __syncwarp();
        sh = warp_sum(sh);
        sx = warp_sum(sx);
        if ((tid & 31) == 0) { red[tid >> 5][0] = sh; red[tid >> 5][1] = sx; }
        __syncthreads();
        if (tid < 2) p.e.part[((size_t)m * (dim / SLICE) + c / SLICE) * NP + MIX + tid] = red[0][tid] + red[1][tid];
    }
}

#ifndef COMM_DEC_NO_ENTRIES
extern "C" __global__ void __launch_bounds__(64) comm_dec_x16(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_body<true, 0, false>(meta, q); }
extern "C" __global__ void __launch_bounds__(64) comm_dec_f32(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_body<false, 0, false>(meta, q); }
#define COMM_DEC_ENTRY(NS)                                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(64) comm_dec_attn_##NS(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_body<true, NS, false>(meta, q); } \
    extern "C" __global__ void __launch_bounds__(64) comm_dec_ffn_##NS(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_body<false, NS, true>(meta, q); }
COMM_DEC_ENTRY(1)
COMM_DEC_ENTRY(2)
COMM_DEC_ENTRY(3)
COMM_DEC_ENTRY(4)
COMM_DEC_ENTRY(5)
COMM_DEC_ENTRY(6)
COMM_DEC_ENTRY(7)
COMM_DEC_ENTRY(8)
#endif
