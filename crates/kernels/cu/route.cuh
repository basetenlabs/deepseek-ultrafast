#pragma once
#include "common.cuh"

DEV void route_token(int m, int lane, const float* scores, const float* bias, int n_exp, int topk, float route_scale,
                     int* sel, float* selw, int* route_ids, float* route_w, int nsum = 1) {
    __syncwarp();
    constexpr int PER = 12;
    float sc[PER], bs[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int e = lane + 32 * i;
        if (nsum > 1) {
            float v = 0.f;
            for (int r = 0; r < nsum; ++r) v += e < n_exp ? scores[((size_t)m * nsum + r) * n_exp + e] : 0.f;
            sc[i] = e < n_exp ? sqrt_softplus(v) : 0.f;
        } else {
            sc[i] = e < n_exp ? scores[(size_t)m * n_exp + e] : 0.f;
        }
        bs[i] = e < n_exp ? bias[e] : 0.f;
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) bs[i] = lane + 32 * i < n_exp ? sc[i] + bs[i] : -INFINITY;
    float wsum = 0.f;
    for (int k = 0; k < topk; ++k) {
        float v[PER];
        int ix[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) { v[i] = bs[i]; ix[i] = i; }
#pragma unroll
        for (int w = 1; w < PER; w *= 2)
#pragma unroll
            for (int i = 0; i + w < PER; i += 2 * w)
                if (v[i + w] > v[i]) { v[i] = v[i + w]; ix[i] = ix[i + w]; }
        const unsigned u = __float_as_uint(v[0]);
        const unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        const unsigned kmax = __reduce_max_sync(0xffffffffu, key);
        const int mine = lane + 32 * ix[0];
        int bi = (int)__reduce_min_sync(0xffffffffu, key == kmax ? (unsigned)mine : 0x7fffffffu);
        float w = 0.f;
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (lane + 32 * i == bi) { w = sc[i]; bs[i] = -INFINITY; }
        w = __shfl_sync(0xffffffffu, w, bi & 31);
        if (lane == 0) { sel[k] = bi; selw[k] = w; }
        wsum += w;
    }
    __syncwarp();
    if (lane < topk) {
        const float w = topk > 1 ? selw[lane] / (wsum + 1e-20f) : selw[lane];
        selw[lane] = w * route_scale;
        if (route_ids) {
            route_ids[(size_t)m * topk + lane] = sel[lane];
            route_w[(size_t)m * topk + lane] = selw[lane];
        }
    }
    __syncwarp();
}
