#include "common.cuh"

#ifndef HC_MIX16
#define HC_MIX16 0
#endif

struct Meta {
    int pos;
    int m;
    int tokens[16];
};

#include "hcepi.cuh"

extern "C" __global__ void __launch_bounds__(64) hc_prep(const Meta* meta, const bf16* h, const __grid_constant__ HcEpi e) {
    extern __shared__ __align__(16) float fns[];
    __shared__ __align__(8) uint64_t bar;
    const int slice = blockIdx.x, m = blockIdx.y, nslices = gridDim.x;
    if (e.fn) hc_fn_issue(e.fn, e.dim, slice * SLICE, fns, &bar);
    pdl_wait();
    pdl_launch();
    if (m >= meta->m) return;
    __syncthreads();
    if (e.fn) hc_fn_wait(&bar);
    const int d = slice * SLICE + threadIdx.x * 4;
    float h4[HC][4];
    for (int c = 0; c < HC; ++c) {
        const uint2 v = *(const uint2*)(h + ((size_t)m * HC + c) * e.dim + d);
        const bf16* b = (const bf16*)&v;
        for (int k = 0; k < 4; ++k) h4[c][k] = __bfloat162float(b[k]);
    }
    float acc[NP];
    hc_epi_thread(e, m, d, h4, e.fn ? fns : nullptr, acc);
    hc_epi_reduce(e, m, slice, nslices, acc);
}

extern "C" __global__ void __launch_bounds__(512) hc_mix(const Meta* meta, const __grid_constant__ HcMix p) {
    pdl_wait();
    pdl_launch();
    const int lane = threadIdx.x & 31, wl = threadIdx.x >> 5, m = blockIdx.x * 16 + wl;
    if (m >= meta->m) return;
    __shared__ float mx_s[16][MIX + 1];
#if HC_MIX16
    hc_mix_token16(p, m, lane, mx_s[wl], p.pre + m * HC, p.post + m * HC, p.comb + m * HC * HC);
#else
    hc_mix_token(p, m, lane, mx_s[wl], p.pre + m * HC, p.post + m * HC, p.comb + m * HC * HC);
#endif
}

extern "C" __global__ void __launch_bounds__(256) hc_side_dots(const Meta* meta, const bf16* h, const float* fn, float* part, int dim) {
    extern __shared__ __align__(16) float fns[];
    __shared__ __align__(8) uint64_t bar;
    __shared__ float red[4][MIX][65];
    const int slice = blockIdx.x, nslices = gridDim.x;
    const int nj = MIX / gridDim.z, jz = (int)blockIdx.z * nj;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa_(&bar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa_(&bar)), "r"((uint32_t)(nj * HC * SLICE * 4)) : "memory");
        for (int j = jz; j < jz + nj; ++j)
            for (int c = 0; c < HC; ++c)
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa_(fns + ((size_t)(j - jz) * HC + c) * SLICE)),
                             "l"(fn + (size_t)j * HC * dim + (size_t)c * dim + slice * SLICE), "r"((uint32_t)(SLICE * 4)), "r"(sa_(&bar))
                             : "memory");
    }
    pdl_wait();
    pdl_launch();
    __syncthreads();
    hc_fn_wait(&bar);
    const int M = min(meta->m, (int)(blockIdx.y + 1) * 64);
    const int grp = threadIdx.x >> 6, t = threadIdx.x & 63;
    const int d = slice * SLICE + t * 4, dl = t * 4;
    for (int m0 = blockIdx.y * 64; m0 < M; m0 += 4) {
        const int m = m0 + grp;
        if (m < M) {
            float h4[HC][4];
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const uint2 v = *(const uint2*)(h + ((size_t)m * HC + c) * dim + d);
                const bf16* b = (const bf16*)&v;
#pragma unroll
                for (int k = 0; k < 4; ++k) h4[c][k] = __bfloat162float(b[k]);
            }
#pragma unroll
            for (int j = 0; j < MIX; ++j) {
                if (j < jz || j >= jz + nj) continue;
                float s = 0.f;
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const float4 f = *(const float4*)(fns + ((size_t)(j - jz) * HC + c) * SLICE + dl);
                    s += h4[c][0] * f.x + h4[c][1] * f.y + h4[c][2] * f.z + h4[c][3] * f.w;
                }
                red[grp][j][t] = s;
            }
        }
        __syncthreads();
        if (t >= jz && t < jz + nj && m < M) {
            float s = 0.f;
            for (int q = 0; q < 64; ++q) s += red[grp][t][q];
            part[((size_t)m * nslices + slice) * NP + t] = s;
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(128) hc_side_dots_p(const Meta* meta, const bf16* h, const float* fn, float* part, int dim, int tile) {
    extern __shared__ __align__(16) float fns[];
    __shared__ __align__(8) uint64_t bar;
    const int slice = blockIdx.x, nslices = gridDim.x;
    hc_fn_issue(fn, dim, slice * SLICE, fns, &bar);
    pdl_wait();
    pdl_launch();
    __syncthreads();
    hc_fn_wait(&bar);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, q = lane & 7, g = lane >> 3;
    const int M = min(meta->m, (int)(blockIdx.y + 1) * tile);
    constexpr int NI = HC * (SLICE / 32);
    for (int m0 = blockIdx.y * tile; m0 < M; m0 += 64) {
        const int mb = m0 + warp * 16 + g * 4;
        if (mb >= M) continue;
        const bf16* hb[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) hb[t] = h + (size_t)min(mb + t, M - 1) * HC * dim + slice * SLICE + 4 * q;
        auto load = [&](int it, uint2* v) {
            const int c = it >> 3, i = it & 7;
#pragma unroll
            for (int t = 0; t < 4; ++t) v[t] = *(const uint2*)(hb[t] + (size_t)c * dim + 32 * i);
        };
        float acc[4][MIX];
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int j = 0; j < MIX; ++j) acc[t][j] = 0.f;
        uint2 cur[4], nxt[4];
        load(0, cur);
#pragma unroll 1
        for (int it = 0; it < NI; ++it) {
            if (it + 1 < NI) load(it + 1, nxt);
            const int c = it >> 3, dl = 32 * (it & 7) + 4 * q;
            float hv[4][4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                hv[t][0] = __uint_as_float(cur[t].x << 16);
                hv[t][1] = __uint_as_float(cur[t].x & 0xffff0000u);
                hv[t][2] = __uint_as_float(cur[t].y << 16);
                hv[t][3] = __uint_as_float(cur[t].y & 0xffff0000u);
            }
#pragma unroll
            for (int j = 0; j < MIX; ++j) {
                const float4 f = *(const float4*)(fns + ((size_t)j * HC + c) * SLICE + dl);
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    acc[t][j] = fmaf(hv[t][3], f.w, fmaf(hv[t][2], f.z, fmaf(hv[t][1], f.y, fmaf(hv[t][0], f.x, acc[t][j]))));
            }
#pragma unroll
            for (int t = 0; t < 4; ++t) cur[t] = nxt[t];
        }
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int j = 0; j < MIX; ++j) {
                acc[t][j] += __shfl_xor_sync(0xffffffff, acc[t][j], 1);
                acc[t][j] += __shfl_xor_sync(0xffffffff, acc[t][j], 2);
                acc[t][j] += __shfl_xor_sync(0xffffffff, acc[t][j], 4);
            }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int m = mb + t;
            if (m < M && (q & 3) == t) {
#pragma unroll
                for (int j = 0; j < MIX; j += 2) {
                    if ((q >> 2) == 0) part[((size_t)m * nslices + slice) * NP + j] = acc[t][j];
                    else part[((size_t)m * nslices + slice) * NP + j + 1] = acc[t][j + 1];
                }
            }
        }
    }
}

constexpr int FROW = HC * SLICE + 8;
extern "C" __global__ void hc_fn_split(const float* fn, int dim, bf16* out) {
    const int nsl = dim / SLICE;
    const long long n = (long long)nsl * MIX * HC * SLICE;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int k = (int)(i % (HC * SLICE)), j = (int)((i / (HC * SLICE)) % MIX), s = (int)(i / (HC * SLICE * MIX));
        const int c = k / SLICE, d = k % SLICE;
        const float f = fn[(size_t)j * HC * dim + (size_t)c * dim + s * SLICE + d];
        const bf16 hi = __float2bfloat16(f);
        const bf16 lo = __float2bfloat16(f - __bfloat162float(hi));
        bf16* o = out + (size_t)s * 2 * MIX * FROW;
        o[(size_t)j * FROW + k] = hi;
        o[(size_t)(MIX + j) * FROW + k] = lo;
    }
}

extern "C" __global__ void __launch_bounds__(256) hc_side_dots_tc(const Meta* meta, const bf16* h, const bf16* fnb, float* part, int dim, int tile) {
    extern __shared__ __align__(128) uint8_t smem_dyn[];
    bf16* fs = (bf16*)smem_dyn;
    __shared__ __align__(8) uint64_t bar;
    const int slice = blockIdx.x, nslices = gridDim.x;
    constexpr uint32_t FB = 2 * MIX * FROW * 2;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa_(&bar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa_(&bar)), "r"(FB) : "memory");
        for (uint32_t o = 0; o < FB; o += 32768)
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa_(smem_dyn + o)),
                         "l"((const uint8_t*)fnb + (size_t)slice * FB + o), "r"(min(32768u, FB - o)), "r"(sa_(&bar))
                         : "memory");
    }
    pdl_wait();
    pdl_launch();
    __syncthreads();
    hc_fn_wait(&bar);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int M = min(meta->m, (int)(blockIdx.y + 1) * tile);
    for (int m0 = blockIdx.y * tile + warp * 16; m0 < M; m0 += 8 * 16) {
        const int ra = min(m0 + g, M - 1), rb = min(m0 + g + 8, M - 1);
        float acc[3][4];
#pragma unroll
        for (int nt = 0; nt < 3; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll 2
        for (int q = 0; q < HC * SLICE / 32; ++q) {
            const int c = q >> 3, d = (q & 7) * 32 + 8 * t;
            const uint4 A0 = *(const uint4*)(h + ((size_t)ra * HC + c) * dim + slice * SLICE + d);
            const uint4 A1 = *(const uint4*)(h + ((size_t)rb * HC + c) * dim + slice * SLICE + d);
            const uint32_t a0[4] = {A0.x, A1.x, A0.y, A1.y}, a1[4] = {A0.z, A1.z, A0.w, A1.w};
#pragma unroll
            for (int nt = 0; nt < 3; ++nt)
#pragma unroll
                for (int hl = 0; hl < 2; ++hl) {
                    const uint4 B = *(const uint4*)(fs + ((size_t)(hl * MIX + nt * 8 + g)) * FROW + q * 32 + 8 * t);
                    mma_bf16(acc[nt], a0, B.x, B.y);
                    mma_bf16(acc[nt], a1, B.z, B.w);
                }
        }
#pragma unroll
        for (int nt = 0; nt < 3; ++nt) {
            const int j = nt * 8 + 2 * t;
            if (m0 + g < M) *(float2*)(part + ((size_t)(m0 + g) * nslices + slice) * NP + j) = make_float2(acc[nt][0], acc[nt][1]);
            if (m0 + g + 8 < M) *(float2*)(part + ((size_t)(m0 + g + 8) * nslices + slice) * NP + j) = make_float2(acc[nt][2], acc[nt][3]);
        }
    }
}
