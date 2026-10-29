#include "hcpipe.cu"

namespace hcs {
template <bool PRE>
DEV void epi_thread(const HcEpi& e, int m, int d, const float (&h4)[HC][4], const float* fns, float (&acc)[NP]) {
    float pre[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) pre[c] = PRE ? e.pre[m * HC + c] : (c == 0 ? 1.f : 0.f);
    float xr[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float y = 0.f;
        if (PRE) {
#pragma unroll
            for (int c = 0; c < HC; ++c) y += pre[c] * h4[c][k];
        } else {
            y = h4[0][k];
        }
        xr[k] = __bfloat162float(__float2bfloat16(y));
    }
    *(__nv_bfloat162*)(e.x_raw + (size_t)m * e.dim + d) = __floats2bfloat162_rn(xr[0], xr[1]);
    *(__nv_bfloat162*)(e.x_raw + (size_t)m * e.dim + d + 2) = __floats2bfloat162_rn(xr[2], xr[3]);
    float sh = 0.f, sx = 0.f;
#pragma unroll
    for (int c = 0; c < HC; ++c)
#pragma unroll
        for (int k = 0; k < 4; ++k) sh += h4[c][k] * h4[c][k];
#pragma unroll
    for (int k = 0; k < 4; ++k) sx += xr[k] * xr[k];
    acc[MIX] = sh;
    acc[MIX + 1] = sx;
    const int dl = d % SLICE;
#pragma unroll
    for (int j = 0; j < MIX; ++j) {
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const float4 f = *(const float4*)(fns + ((size_t)j * HC + c) * SLICE + dl);
            s += h4[c][0] * f.x + h4[c][1] * f.y + h4[c][2] * f.z + h4[c][3] * f.w;
        }
        acc[j] = s;
    }
}

DEV void epi_reduce(const HcEpi& e, int m, int slice, int nslices, const float (&acc)[NP]) {
    __shared__ float red[2][NP];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncwarp();
#pragma unroll
    for (int j = 0; j < NP; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) red[w][j] = v;
    }
    __syncthreads();
    if (threadIdx.x < NP) e.part[((size_t)m * nslices + slice) * NP + threadIdx.x] = red[0][threadIdx.x] + red[1][threadIdx.x];
}

template <bool PRE>
DEV void prep(const Meta* meta, const bf16* h, const HcEpi& e, float* fns, uint64_t* bar) {
    const int slice = blockIdx.x, m = blockIdx.y, nslices = gridDim.x;
    hc_fn_issue_w(e.fn, e.dim, slice * SLICE, fns, bar);
    pdl_wait();
    pdl_launch();
    if (m >= meta->m) return;
    const int d = slice * SLICE + threadIdx.x * 4;
    float h4[HC][4];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const uint2 v = *(const uint2*)(h + ((size_t)m * HC + c) * e.dim + d);
        const bf16* b = (const bf16*)&v;
#pragma unroll
        for (int k = 0; k < 4; ++k) h4[c][k] = __bfloat162float(b[k]);
    }
    __syncthreads();
    hc_fn_wait(bar);
    float acc[NP];
    epi_thread<PRE>(e, m, d, h4, fns, acc);
    epi_reduce(e, m, slice, nslices, acc);
}
}

extern "C" __global__ void __launch_bounds__(64) hc_prep_pd(const Meta* meta, const bf16* h, const __grid_constant__ HcEpi e) {
    extern __shared__ __align__(16) float fns[];
    __shared__ __align__(8) uint64_t bar;
    hcs::prep<true>(meta, h, e, fns, &bar);
}

extern "C" __global__ void __launch_bounds__(64) hc_prep_d(const Meta* meta, const bf16* h, const __grid_constant__ HcEpi e) {
    extern __shared__ __align__(16) float fns[];
    __shared__ __align__(8) uint64_t bar;
    hcs::prep<false>(meta, h, e, fns, &bar);
}

extern "C" __global__ void __launch_bounds__(64) embed_prep(const Meta* meta, const bf16* table, bf16* h, const int* token_map, int* cmp_hist,
                                                           const __grid_constant__ HcEpi e) {
    extern __shared__ __align__(16) float fns[];
    __shared__ __align__(8) uint64_t bar;
    const int slice = blockIdx.x, m = blockIdx.y, nslices = gridDim.x;
    hc_fn_issue_w(e.fn, e.dim, slice * SLICE, fns, &bar);
    pdl_wait();
    pdl_launch();
    if (m >= meta->m) return;
    const int tok = meta->tokens[m];
    const int d = slice * SLICE + threadIdx.x * 4;
    const uint2 v = *(const uint2*)(table + (size_t)tok * e.dim + d);
#pragma unroll
    for (int c = 0; c < HC; ++c) *(uint2*)(h + ((size_t)m * HC + c) * e.dim + d) = v;
    if (slice == 0 && threadIdx.x == 0 && cmp_hist) cmp_hist[meta->pos + m] = token_map[tok];
    const bf16* b = (const bf16*)&v;
    float h4[HC][4];
#pragma unroll
    for (int c = 0; c < HC; ++c)
#pragma unroll
        for (int k = 0; k < 4; ++k) h4[c][k] = __bfloat162float(b[k]);
    __syncthreads();
    hc_fn_wait(&bar);
    float acc[NP];
    hcs::epi_thread<false>(e, m, d, h4, fns, acc);
    hcs::epi_reduce(e, m, slice, nslices, acc);
}
