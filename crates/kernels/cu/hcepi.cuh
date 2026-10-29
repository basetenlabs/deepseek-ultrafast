#pragma once
#include "common.cuh"

constexpr int HC = 4, MIX = 24, NP = 26;
constexpr int SLICE = 256;

struct HcEpi {
    const float* fn;
    const float* pre;
    bf16* x_raw;
    float* part;
    int dim;
};

DEV void hc_epi_thread(const HcEpi& e, int m, int d, const float h4[HC][4], const float* fns, float acc[NP]) {
    float pre[HC];
    for (int c = 0; c < HC; ++c) pre[c] = e.pre ? e.pre[m * HC + c] : (c == 0 ? 1.f : 0.f);
    float xr[4];
    for (int k = 0; k < 4; ++k) {
        float y = 0.f;
        if (e.pre) {
            for (int c = 0; c < HC; ++c) y += pre[c] * h4[c][k];
        } else {
            y = h4[0][k];
        }
        xr[k] = __bfloat162float(__float2bfloat16(y));
    }
    *(__nv_bfloat162*)(e.x_raw + (size_t)m * e.dim + d) = __floats2bfloat162_rn(xr[0], xr[1]);
    *(__nv_bfloat162*)(e.x_raw + (size_t)m * e.dim + d + 2) = __floats2bfloat162_rn(xr[2], xr[3]);
    float sh = 0.f, sx = 0.f;
    for (int c = 0; c < HC; ++c)
        for (int k = 0; k < 4; ++k) sh += h4[c][k] * h4[c][k];
    for (int k = 0; k < 4; ++k) sx += xr[k] * xr[k];
    acc[MIX] = sh;
    acc[MIX + 1] = sx;
    if (fns) {
        const int dl = d % SLICE;
        for (int j = 0; j < MIX; ++j) {
            float s = 0.f;
            for (int c = 0; c < HC; ++c) {
                const float4 f = *(const float4*)(fns + ((size_t)j * HC + c) * SLICE + dl);
                s += h4[c][0] * f.x + h4[c][1] * f.y + h4[c][2] * f.z + h4[c][3] * f.w;
            }
            acc[j] = s;
        }
    } else {
        for (int j = 0; j < MIX; ++j) acc[j] = 0.f;
    }
}

DEV void hc_epi_reduce(const HcEpi& e, int m, int slice, int nslices, float acc[NP]) {
    __shared__ float red[2][NP];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int j0 = e.fn ? 0 : MIX;
    __syncwarp();
    for (int j = j0; j < NP; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) red[w][j] = v;
    }
    __syncthreads();
    if (threadIdx.x >= j0 && threadIdx.x < NP) e.part[((size_t)m * nslices + slice) * NP + threadIdx.x] = red[0][threadIdx.x] + red[1][threadIdx.x];
    __syncthreads();
}

DEV uint32_t sa_(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void hc_fn_issue(const float* fn, int dim, int d0, float* fns, uint64_t* bar) {
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa_(bar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa_(bar)), "r"((uint32_t)(MIX * HC * SLICE * 4)) : "memory");
        for (int j = 0; j < MIX; ++j)
            for (int c = 0; c < HC; ++c)
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa_(fns + ((size_t)j * HC + c) * SLICE)),
                             "l"(fn + (size_t)j * HC * dim + (size_t)c * dim + d0), "r"((uint32_t)(SLICE * 4)), "r"(sa_(bar))
                             : "memory");
    }
}
DEV void hc_fn_issue_w(const float* fn, int dim, int d0, float* fns, uint64_t* bar) {
    if (threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa_(bar)));
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa_(bar)), "r"((uint32_t)(MIX * HC * SLICE * 4)) : "memory");
        }
        __syncwarp();
        for (int q = threadIdx.x; q < MIX * HC; q += 32) {
            const int j = q / HC, c = q % HC;
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa_(fns + ((size_t)j * HC + c) * SLICE)),
                         "l"(fn + (size_t)j * HC * dim + (size_t)c * dim + d0), "r"((uint32_t)(SLICE * 4)), "r"(sa_(bar))
                         : "memory");
        }
    }
}
DEV void hc_fn_wait(uint64_t* bar) {
    asm volatile("{ .reg .pred P; W_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W_%=; }" ::"r"(sa_(bar)) : "memory");
}

DEV float frcp(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }

struct HcMix {
    const float* part;
    int nslices;
    const float* base;
    const float* scale;
    float* pre;
    float* post;
    float* comb;
    int dim, iters;
    float norm_eps, hc_eps;
};

#if HC_MIX16
template <bool LDCG = false>
DEV void hc_mix_token16(const HcMix& p, int m, int lane, float* mx, float* pre, float* post, float* comb) {
    if (lane <= MIX) {
        constexpr int SMAX = 32;
        float v[SMAX];
#pragma unroll
        for (int sl = 0; sl < SMAX; ++sl) {
            const float* a = p.part + ((size_t)m * p.nslices + sl) * NP + lane;
            v[sl] = sl < p.nslices ? (LDCG ? __ldcg(a) : *a) : 0.f;
        }
        float s = 0.f;
#pragma unroll
        for (int sl = 0; sl < SMAX; ++sl)
            if (sl < p.nslices) s += v[sl];
        for (int sl = SMAX; sl < p.nslices; ++sl) {
            const float* a = p.part + ((size_t)m * p.nslices + sl) * NP + lane;
            s += LDCG ? __ldcg(a) : *a;
        }
        mx[lane] = s;
    }
    __syncwarp();
    const float rs = rsqrtf(mx[MIX] / (HC * p.dim) + p.norm_eps);
    if (lane < 16) {
        constexpr unsigned M16 = 0xffffu;
        const int j = lane >> 2, k = lane & 3, r0 = lane & ~3;
        float c = mx[2 * HC + j * HC + k] * rs * p.scale[2] + p.base[2 * HC + j * HC + k];
        const float a0 = __shfl_sync(M16, c, r0), a1 = __shfl_sync(M16, c, r0 + 1), a2 = __shfl_sync(M16, c, r0 + 2), a3 = __shfl_sync(M16, c, r0 + 3);
        const float mxr = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
        const float e = expf(c - mxr);
        const float e0 = __shfl_sync(M16, e, r0), e1 = __shfl_sync(M16, e, r0 + 1), e2 = __shfl_sync(M16, e, r0 + 2), e3 = __shfl_sync(M16, e, r0 + 3);
        float s = 0.f;
        s += e0;
        s += e1;
        s += e2;
        s += e3;
        c = e / s + p.hc_eps;
        auto colnorm = [&]() {
            const float c0 = __shfl_sync(M16, c, k), c1 = __shfl_sync(M16, c, 4 + k), c2 = __shfl_sync(M16, c, 8 + k), c3 = __shfl_sync(M16, c, 12 + k);
            const float r = frcp(((c0 + c1) + c2) + c3 + p.hc_eps);
            c *= r;
        };
        colnorm();
        for (int it = 0; it < p.iters - 1; ++it) {
            const float c0 = __shfl_sync(M16, c, r0), c1 = __shfl_sync(M16, c, r0 + 1), c2 = __shfl_sync(M16, c, r0 + 2), c3 = __shfl_sync(M16, c, r0 + 3);
            const float r = frcp(((c0 + c1) + c2) + c3 + p.hc_eps);
            c *= r;
            colnorm();
        }
        comb[j * HC + k] = c;
    } else if (lane < 16 + HC) {
        const int j = lane - 16;
        pre[j] = 1.f / (1.f + expf(-(mx[j] * rs * p.scale[0] + p.base[j]))) + p.hc_eps;
    } else if (lane < 16 + 2 * HC) {
        const int j = lane - 16 - HC;
        post[j] = 2.f / (1.f + expf(-(mx[HC + j] * rs * p.scale[1] + p.base[HC + j])));
    }
}
#endif

template <bool LDCG = false>
DEV void hc_mix_token(const HcMix& p, int m, int lane, float* mx, float* pre, float* post, float* comb) {
    if (lane <= MIX) {
        constexpr int SMAX = 32;
        float v[SMAX];
#pragma unroll
        for (int sl = 0; sl < SMAX; ++sl) {
            const float* a = p.part + ((size_t)m * p.nslices + sl) * NP + lane;
            v[sl] = sl < p.nslices ? (LDCG ? __ldcg(a) : *a) : 0.f;
        }
        float s = 0.f;
#pragma unroll
        for (int sl = 0; sl < SMAX; ++sl)
            if (sl < p.nslices) s += v[sl];
        for (int sl = SMAX; sl < p.nslices; ++sl) {
            const float* a = p.part + ((size_t)m * p.nslices + sl) * NP + lane;
            s += LDCG ? __ldcg(a) : *a;
        }
        mx[lane] = s;
    }
    __syncwarp();
    const float rs = rsqrtf(mx[MIX] / (HC * p.dim) + p.norm_eps);
    if (lane == 0) {
        float c[HC][HC];
        for (int j = 0; j < HC; ++j)
            for (int k = 0; k < HC; ++k) c[j][k] = mx[2 * HC + j * HC + k] * rs * p.scale[2] + p.base[2 * HC + j * HC + k];
        for (int j = 0; j < HC; ++j) {
            const float mxr = fmaxf(fmaxf(c[j][0], c[j][1]), fmaxf(c[j][2], c[j][3]));
            float e[HC], s = 0.f;
            for (int k = 0; k < HC; ++k) { e[k] = expf(c[j][k] - mxr); s += e[k]; }
            for (int k = 0; k < HC; ++k) c[j][k] = e[k] / s + p.hc_eps;
        }
        auto colnorm = [&]() {
            for (int k = 0; k < HC; ++k) {
                const float r = frcp(((c[0][k] + c[1][k]) + c[2][k]) + c[3][k] + p.hc_eps);
                for (int j = 0; j < HC; ++j) c[j][k] *= r;
            }
        };
        colnorm();
        for (int it = 0; it < p.iters - 1; ++it) {
            for (int j = 0; j < HC; ++j) {
                const float r = frcp(((c[j][0] + c[j][1]) + c[j][2]) + c[j][3] + p.hc_eps);
                for (int k = 0; k < HC; ++k) c[j][k] *= r;
            }
            colnorm();
        }
        for (int j = 0; j < HC; ++j)
            for (int k = 0; k < HC; ++k) comb[j * HC + k] = c[j][k];
    } else if (lane <= HC) {
        const int j = lane - 1;
        pre[j] = 1.f / (1.f + expf(-(mx[j] * rs * p.scale[0] + p.base[j]))) + p.hc_eps;
    } else if (lane <= 2 * HC) {
        const int j = lane - 1 - HC;
        post[j] = 2.f / (1.f + expf(-(mx[HC + j] * rs * p.scale[1] + p.base[HC + j])));
    }
}
