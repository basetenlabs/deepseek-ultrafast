#pragma once
#include "common.cuh"

DEV float rp_bf(bf16 v) { return __bfloat162float(v); }
DEV bf16 rp_tobf(float v) { return __float2bfloat16(v); }

enum { Q_NONE = 0, Q_FP8 = 1, Q_FP4_E8M0 = 2, Q_FP4_E4M3_16 = 3 };

DEV float fp4_round(float v) {
    uint16_t b;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %1; cvt.u16.u8 %0, t; }" : "=h"(b) : "f"(v));
    uint32_t h = e2m1x2_to_f16x2(b & 0xff);
    return __half2float(__ushort_as_half((unsigned short)(h & 0xffff)));
}

struct RopeParams {
    bf16* x;
    int ldx, n;
    int rows_per_token;
    int rope_off;
    const float* cos;
    const float* sin;
    int inverse;
    int quant;
    int ratio;
    bf16* dst;
    int pos_offset;
    const bf16* norm_w;
    float norm_eps;
    int dst_by_row;
    uint8_t* q4;
    uint8_t* sf4;
};

DEV void rope_body(const int* meta, const RopeParams& p, int row_, int lt, int grp) {
    pdl_wait();
    pdl_launch();
    const int row = row_;
    const int m = row / p.rows_per_token, head = row % p.rows_per_token;
    if (m >= meta[1]) return;
    int pos = meta[0] + m + p.pos_offset;
    int cidx = pos;
    if (p.ratio > 0) {
        if ((pos + 1) % p.ratio != 0) return;
        cidx = pos / p.ratio;
        pos = cidx * p.ratio;
    }
    bf16* xr = p.x + (size_t)row * p.ldx;
    const int e0 = lt * 8;
    float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (e0 < p.n) {
        uint4 q = *(const uint4*)(xr + e0);
        const bf16* hb = (const bf16*)&q;
        for (int i = 0; i < 8; ++i) v[i] = rp_bf(hb[i]);
    }
    if (p.norm_w) {
        __shared__ float nsh[4][2];
        float ss = 0.f;
        for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
        ss = warp_sum(ss);
        if ((lt & 31) == 0) nsh[grp][lt >> 5] = ss;
        switch (grp) {
            case 0: asm volatile("bar.sync 1, 64;" ::: "memory"); break;
            case 1: asm volatile("bar.sync 2, 64;" ::: "memory"); break;
            case 2: asm volatile("bar.sync 3, 64;" ::: "memory"); break;
            default: asm volatile("bar.sync 4, 64;" ::: "memory"); break;
        }
        const float r = rsqrtf((nsh[grp][0] + nsh[grp][1]) / p.n + p.norm_eps);
        if (e0 < p.n)
            for (int i = 0; i < 8; ++i) v[i] = rp_bf(rp_tobf(rp_bf(p.norm_w[e0 + i]) * (v[i] * r)));
    }
    if (e0 < p.n) {
        if (e0 >= p.rope_off) {
            for (int i = 0; i < 8; i += 2) {
                const int f = (e0 - p.rope_off + i) / 2;
                const float c = p.cos[(size_t)pos * 32 + f], s = p.sin[(size_t)pos * 32 + f];
                const float a = v[i], b = v[i + 1];
                if (p.inverse) { v[i] = __fmaf_rn(a, c, __fmul_rn(b, s)); v[i + 1] = __fmaf_rn(b, c, -__fmul_rn(a, s)); }
                else { v[i] = __fmaf_rn(a, c, -__fmul_rn(b, s)); v[i + 1] = __fmaf_rn(a, s, __fmul_rn(b, c)); }
            }
            for (int i = 0; i < 8; ++i) v[i] = rp_bf(rp_tobf(v[i]));
        }
    }
    if (p.quant != Q_NONE) {
        const int G = p.quant == Q_FP4_E4M3_16 ? 16 : 32;
        float amax = 0.f;
        for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
        for (int o = 1; o < G / 8; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        float s;
        if (p.quant == Q_FP8) {
            amax = fmaxf(amax, 1e-4f);
            s = round_scale_pow2(amax, 1.0f / 448.0f);
            for (int i = 0; i < 8; i += 2) {
                uint32_t h = e4m3x2_to_f16x2(f32x2_to_e4m3x2(v[i] / s, v[i + 1] / s));
                v[i] = __half2float(__ushort_as_half((unsigned short)(h & 0xffff))) * s;
                v[i + 1] = __half2float(__ushort_as_half((unsigned short)(h >> 16))) * s;
            }
        } else {
            if (p.quant == Q_FP4_E8M0) {
                amax = fmaxf(amax, 6.f * 1.1754943508222875e-38f);
                s = round_scale_pow2(amax, 1.0f / 6.0f);
            } else {
                amax = fmaxf(amax, 6.f / 512.f);
                uint16_t e = f32x2_to_e4m3x2(amax / 6.f, 0.f);
                uint32_t h = e4m3x2_to_f16x2(e);
                s = __half2float(__ushort_as_half((unsigned short)(h & 0xffff)));
            }
            if (p.q4 && p.quant == Q_FP4_E8M0 && e0 < p.n) {
                const size_t r4 = p.dst ? (size_t)(p.dst_by_row ? m : cidx) * p.rows_per_token + head : (size_t)row;
                uint32_t pk = 0;
#pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    uint16_t b;
                    const float a0 = fminf(fmaxf(__fdiv_rn(v[i], s), -6.f), 6.f), a1 = fminf(fmaxf(__fdiv_rn(v[i + 1], s), -6.f), 6.f);
                    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }" : "=h"(b) : "f"(a1), "f"(a0));
                    pk |= (uint32_t)(b & 0xff) << (4 * i);
                }
                *(uint32_t*)(p.q4 + r4 * 64 + e0 / 2) = pk;
                if ((e0 & 31) == 0)
                    p.sf4[(r4 / 128) * 512 + (r4 % 32) * 16 + ((r4 % 128) / 32) * 4 + e0 / 32] = (uint8_t)((__float_as_uint(s) >> 23) & 0xff);
            }
            for (int i = 0; i < 8; ++i) v[i] = fp4_round(fminf(fmaxf(__fdiv_rn(v[i], s), -6.f), 6.f)) * s;
        }
    }
    if (e0 < p.n) {
        uint4 q;
        bf16* hb = (bf16*)&q;
        for (int i = 0; i < 8; ++i) hb[i] = rp_tobf(v[i]);
        *(uint4*)(xr + e0) = q;
        if (p.dst) *(uint4*)(p.dst + ((size_t)(p.dst_by_row ? m : cidx) * p.rows_per_token + head) * p.n + e0) = q;
    }
}
