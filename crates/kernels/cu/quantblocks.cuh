#pragma once
#include "common.cuh"

#ifndef QB8_U
#define QB8_U 2
#endif
template <class Load8>
DEV void quant_blocks8(int warp, int nwarps, int lane, int MT, int n, int nkb, uint16_t* xs, int LDS, float* sa, Load8 load8, float dqmul = 1.f) {
    constexpr int U = QB8_U;
    const int total = n * nkb, bl = lane >> 2, c = lane & 3, nb = nwarps * 8;
    const float rnkb = 1.0f / (float)nkb;
    for (int b0 = warp * 8 + bl; b0 - bl < total; b0 += nb * U) {
        float v[U][8];
        int mm[U], kk[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int b = b0 + u * nb;
            mm[u] = __float2int_rd(((float)b + 0.5f) * rnkb);
            kk[u] = b - mm[u] * nkb;
            if (b < total) load8(mm[u], kk[u], c, v[u]);
            else {
#pragma unroll
                for (int e = 0; e < 8; ++e) v[u][e] = 0.f;
            }
        }
        float a[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            a[u] = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) a[u] = fmaxf(a[u], fabsf(v[u][e]));
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1)
#pragma unroll
            for (int u = 0; u < U; ++u) a[u] = fmaxf(a[u], __shfl_xor_sync(0xffffffff, a[u], o));
        __syncwarp();
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int b = b0 + u * nb;
            if (b < total) {
                const float sc = round_scale_pow2(fmaxf(a[u], 1e-4f), 1.0f / 448.0f);
                const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
                uint16_t* dst = xs + mm[u] * LDS + kk[u] * 32 + 2 * (c & 1) + 4 * (c >> 1);
#pragma unroll
                for (int i0 = 0; i0 < 4; ++i0) {
                    const int i = (i0 + bl) & 3;
                    float x0 = v[u][0], x1 = v[u][1];
#pragma unroll
                    for (int q = 1; q < 4; ++q)
                        if (q == i) { x0 = v[u][2 * q]; x1 = v[u][2 * q + 1]; }
                    *(uint32_t*)(dst + 8 * i) = f16x2_dequant(e4m3x2_to_f16x2(f32x2_to_e4m3x2(x0 * inv, x1 * inv)), sc * dqmul);
                }
                if (c == 0) sa[kk[u] * MT + mm[u]] = sc;
            }
        }
    }
}
