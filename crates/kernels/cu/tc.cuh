#pragma once
#include "common.cuh"

DEV uint32_t s32a(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mb_init_t(uint64_t* b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(s32a(b)), "r"(c)); }
DEV void mb_expect_t(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(s32a(b)), "r"(bytes) : "memory");
}
DEV void mb_wait_t(uint64_t* b, uint32_t par) {
    asm volatile("{ .reg .pred p; W_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W_%=; }" ::"r"(s32a(b)), "r"(par) : "memory");
}
DEV void tma2d(void* dst, const void* tmap, int c0, int c1, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(s32a(dst)), "l"(tmap),
                 "r"(c0), "r"(c1), "r"(s32a(bar))
                 : "memory");
}
DEV void tma3d(void* dst, const void* tmap, int c0, int c1, int c2, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(s32a(dst)),
                 "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(s32a(bar))
                 : "memory");
}
DEV void bulk1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(s32a(dst)), "l"(src), "r"(bytes),
                 "r"(s32a(bar))
                 : "memory");
}
constexpr uint64_t L2_EVICT_FIRST = 0x12F0000000000000ull;
DEV void tma2d_ef(void* dst, const void* tmap, int c0, int c1, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(s32a(dst)),
                 "l"(tmap), "r"(c0), "r"(c1), "r"(s32a(bar)), "l"(L2_EVICT_FIRST)
                 : "memory");
}
DEV void tma3d_ef(void* dst, const void* tmap, int c0, int c1, int c2, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(s32a(dst)),
                 "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(s32a(bar)), "l"(L2_EVICT_FIRST)
                 : "memory");
}
DEV void bulk1d_ef(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(s32a(dst)), "l"(src),
                 "r"(bytes), "r"(s32a(bar)), "l"(L2_EVICT_FIRST)
                 : "memory");
}
DEV void tmalloc(uint32_t* dst, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(s32a(dst)), "r"(ncols) : "memory");
}
DEV void tmdealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
DEV void tcbefore() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEV void tcafter() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
DEV void tccommit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(s32a(bar)) : "memory");
}
DEV void tccpsf(uint32_t taddr, uint64_t desc) { asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(desc) : "memory"); }
DEV void tcmma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; }" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb)
        : "memory");
}
DEV bool elect_one() {
    uint32_t pred;
    asm volatile("{ .reg .pred e; elect.sync _|e, 0xffffffff; selp.u32 %0, 1, 0, e; }" : "=r"(pred));
    return pred != 0;
}
DEV uint32_t uni(uint32_t v) { return __shfl_sync(0xffffffffu, v, 0); }
DEV uint64_t uni64(uint64_t v) { return ((uint64_t)uni((uint32_t)(v >> 32)) << 32) | uni((uint32_t)v); }
DEV void tcld32(uint32_t taddr, float* v) {
    uint32_t r[32];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
          "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]),
          "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
          "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = __uint_as_float(r[i]);
}
DEV uint64_t sdesc_t(const void* p, uint32_t lbo, uint32_t sbo, uint32_t layout) {
    uint64_t d = (uint64_t)((s32a(p) & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)(layout & 7) << 61;
    return d;
}
DEV uint32_t idesc_e4m3(int n, int kc) { return ((uint32_t)(n >> 3) << 17) | (1u << 23) | ((128u >> 4) << 24) | ((uint32_t)kc << 29) | ((uint32_t)kc << 4); }
DEV uint32_t idesc_e2m1_e4m3(int n, int kc) { return (5u << 7) | idesc_e4m3(n, kc); }
DEV void tcfence_proxy() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

DEV void mb_wait_sleep(uint64_t* b, uint32_t par, int ns) {
    uint32_t done;
    for (;;) {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(s32a(b)), "r"(par) : "memory");
        if (done) break;
        __nanosleep(ns);
    }
}
DEV void tccpsf_128x256(uint32_t taddr, uint64_t desc) { asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" ::"r"(taddr), "l"(desc) : "memory"); }
DEV uint32_t idesc_bf16(int n) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(n >> 3) << 17) | ((128u >> 4) << 24); }
DEV void tcmma_f16(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(acc)
        : "memory");
}
