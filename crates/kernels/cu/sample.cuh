#pragma once
#include "common.cuh"

struct SampParams {
    float temp;
    float top_p;
    unsigned long long seed;
    unsigned long long step;
    unsigned pad;
};
static_assert(sizeof(SampParams) == 32);

enum { RNG_DRAFT = 1, RNG_ACCEPT = 2, RNG_FINAL = 3 };

DEV uint4 philox4x32_10(uint4 c, uint2 k) {
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        const unsigned hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        const unsigned hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k.x += 0x9E3779B9u;
        k.y += 0xBB67AE85u;
    }
    return c;
}
DEV unsigned rng_bits(const SampParams& sp, unsigned purpose, unsigned a, unsigned b) {
    const uint4 r = philox4x32_10(make_uint4(a, (b & 0xffffffu) | (purpose << 24), (unsigned)sp.step, (unsigned)(sp.step >> 32)),
                                  make_uint2((unsigned)sp.seed, (unsigned)(sp.seed >> 32)));
    return r.x;
}
DEV float rng_u01(unsigned x) { return ((float)(x >> 9) + 0.5f) * (1.0f / 8388608.0f); }
DEV float rng_gumbel(unsigned x) { return -logf(-logf(rng_u01(x))); }
