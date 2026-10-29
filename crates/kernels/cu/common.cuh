#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

#define DEV __device__ __forceinline__

#define COMM_MIN_CTAS_PER_SM 18
#define GEMV_MIN_CTAS_PER_SM(PH) 2

DEV void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEV void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

DEV float sqrt_softplus(float x) {
    if constexpr (1 == 1) {
        const float e = __expf(x);
        const float l = e < 0.0078125f ? e * (1.f - e * (0.5f - e * 0.33333334f)) : __logf(1.f + e);
        return sqrtf(x > 20.f ? x : l);
    }
    return sqrtf(x > 20.f ? x : log1pf(expf(x)));
}

DEV float norm_part_sum(const float* part, int row, int n) {
    constexpr int MAXS = 32;
    float v[MAXS];
    const float* b = part + (size_t)row * n * 26 + 25;
#pragma unroll
    for (int i = 0; i < MAXS; ++i) v[i] = i < n ? __ldcg(b + (size_t)i * 26) : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < MAXS; ++i)
        if (i < n) ss += v[i];
    return ss;
}

DEV uint32_t e4m3x2_to_f16x2(uint16_t v) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
}
DEV uint32_t e4m3x2_to_bf16x2(uint16_t v) {
    uint32_t h = e4m3x2_to_f16x2(v), r;
    float lo = __half2float(__ushort_as_half((unsigned short)(h & 0xffff)));
    float hi = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DEV uint32_t e2m1x2_to_f16x2(uint32_t byte) {
    uint32_t r;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "r"(byte));
    return r;
}
DEV void cvt4_e2m1(uint32_t b, uint32_t* a) {
    asm("{ .reg .b8 b0, b1, b2, b3;\n"
        "  mov.b32 {b0, b1, b2, b3}, %4;\n"
        "  cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "  cvt.rn.f16x2.e2m1x2 %1, b2;\n"
        "  cvt.rn.f16x2.e2m1x2 %2, b1;\n"
        "  cvt.rn.f16x2.e2m1x2 %3, b3; }"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(b));
}
DEV uint8_t* align1k(uint8_t* p) { return p + ((1024u - ((uint32_t)__cvta_generic_to_shared(p) & 1023u)) & 1023u); }
DEV constexpr int kpos32(int k) { return 8 * ((k & 7) >> 1) + (k & 1) + 2 * ((k >> 3) & 1) + 4 * (k >> 4); }
DEV uint16_t f32x2_to_e4m3x2(float lo, float hi) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}
DEV float e8m0_to_f32(uint8_t e) { return __uint_as_float(((uint32_t)e) << 23); }
DEV uint32_t f16x2_dequant(uint32_t h, float sc) {
    const float2 f = __half22float2(*(const __half2*)&h);
    const __half2 r = __floats2half2_rn(f.x * sc, f.y * sc);
    return *(const uint32_t*)&r;
}

DEV float round_scale_pow2(float amax, float qmax_inv) {
    uint32_t bits = __float_as_uint(amax * qmax_inv);
    int e = (int)((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffff) != 0 ? 1 : 0);
    return __uint_as_float((uint32_t)(e + 127) << 23);
}

DEV float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
DEV float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

DEV void mma_f16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEV void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
