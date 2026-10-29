#include "common.cuh"

template<int G>
DEV void offload_pack(const bf16* src, uint8_t* data, uint8_t* scales, uint32_t* valid, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float x = i < n ? __bfloat162float(src[i]) : 0.f;
    float amax = fabsf(x);
    bool ok = isfinite(x) && (G != 16 || amax <= 2688.f);
    for (int d = 1; d < G; d <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d, G));
    float scale;
    uint8_t scale_byte;
    if constexpr (G == 16) {
        amax = fmaxf(amax, 6.f / 512.f);
        const uint16_t e = f32x2_to_e4m3x2(__fdiv_rn(amax, 6.f), 0.f);
        scale_byte = e & 0xff;
        const uint32_t h = e4m3x2_to_f16x2(e);
        scale = __half2float(__ushort_as_half((unsigned short)(h & 0xffff)));
        if (scale_byte < 126 && 6.f * scale < amax) {
            ++scale_byte;
            const uint32_t next = e4m3x2_to_f16x2(scale_byte);
            scale = __half2float(__ushort_as_half((unsigned short)(next & 0xffff)));
        }
    } else {
        amax = fmaxf(amax, 6.f * 1.1754943508222875e-38f);
        scale = round_scale_pow2(amax, 1.f / 6.f);
        scale_byte = (__float_as_uint(scale) >> 23) & 0xff;
    }
    const float q = fminf(fmaxf(__fdiv_rn(x, scale), -6.f), 6.f);
    uint16_t pair;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %1; cvt.u16.u8 %0, t; }" : "=h"(pair) : "f"(q));
    const uint32_t code = (pair & 7) | ((__float_as_uint(x) >> 28) & 8);
    const uint32_t other = __shfl_xor_sync(0xffffffff, code, 1, G);
    const uint32_t h = e2m1x2_to_f16x2(code & 7);
    const float restored = __half2float(__ushort_as_half((unsigned short)(h & 0xffff))) * scale;
    ok = ok && isfinite(restored) && restored <= __uint_as_float(0x7f7f0000);
    if (i < n) {
        if (!ok) atomicAnd(valid, 0u);
        if ((i & 1) == 0) data[i / 2] = code | (other << 4);
        if (i % G == 0) scales[i / G] = scale_byte;
    }
}

template<int G>
DEV void offload_unpack(const uint8_t* data, const uint8_t* scales, bf16* dst, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t code = (data[i / 2] >> (4 * (i & 1))) & 15;
    float scale;
    if constexpr (G == 16) {
        const uint32_t h = e4m3x2_to_f16x2(scales[i / G]);
        scale = __half2float(__ushort_as_half((unsigned short)(h & 0xffff)));
    } else {
        scale = __uint_as_float((uint32_t)scales[i / G] << 23);
    }
    const uint32_t h = e2m1x2_to_f16x2(code & 7);
    const float magnitude = __half2float(__ushort_as_half((unsigned short)(h & 0xffff))) * scale;
    ((uint16_t*)dst)[i] = __bfloat16_as_ushort(__float2bfloat16_rn(magnitude)) | ((code & 8) << 12);
}

extern "C" __global__ void kv_fp4_pack16(const bf16* src, uint8_t* data, uint8_t* scales, uint32_t* valid, size_t n) {
    offload_pack<16>(src, data, scales, valid, n);
}
extern "C" __global__ void kv_fp4_pack32(const bf16* src, uint8_t* data, uint8_t* scales, uint32_t* valid, size_t n) {
    offload_pack<32>(src, data, scales, valid, n);
}
extern "C" __global__ void kv_fp4_unpack16(const uint8_t* data, const uint8_t* scales, bf16* dst, size_t n) {
    offload_unpack<16>(data, scales, dst, n);
}
extern "C" __global__ void kv_fp4_unpack32(const uint8_t* data, const uint8_t* scales, bf16* dst, size_t n) {
    offload_unpack<32>(data, scales, dst, n);
}
