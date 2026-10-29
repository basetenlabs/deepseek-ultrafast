#include "common.cuh"

struct Meta {
    int pos;
    int m;
    int tokens[16];
};

struct AttnFParams {
    const bf16* q;
    const bf16* win;
    const bf16* comp;
    const int* topk;
    int ktop;
    float scale;
    const float* sink;
    bf16* out;
    const float* cos;
    const float* sin;
    int R;
    int fixed_window;
    int nh;
    int pre;
    int win_floor;
};

namespace sds {
constexpr int kDim = 512;
constexpr int kSwa = 128;
constexpr int kChunk = 80;
constexpr int kSplits = 8;
constexpr int kSlice = kDim / kSplits;
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;
constexpr int kRowB = kDim * 2;
constexpr int kPStride = kChunk + 8;
constexpr int kRowsPerWarp = kChunk / kWarps;
constexpr int kQkParts = 3;
constexpr int kDimsPerWarp = kDim / kWarps;
constexpr float kLog2e = 1.4426950408889634f;
static_assert(kChunk % kWarps == 0 && (kChunk / 16) * kQkParts <= kWarps, "warp layout");
static_assert(kSlice % kDimsPerWarp == 0, "a warp's head dims belong to one peer");

template <int H>
struct alignas(128) Smem {
    alignas(128) uint8_t kv[kChunk * kRowB];
    alignas(128) uint8_t q[H * kRowB];
    alignas(16) bf16 pt[H * kPStride];
    float s[kQkParts][H][kChunk];
    int masked[kChunk];
    float2 ml[H];
    alignas(16) float stage[kWarps][H * kDimsPerWarp];
    alignas(16) float recv[kSplits][H][kSlice];
    alignas(8) float2 recv_ml[kSplits][H];
    alignas(8) uint64_t recv_bar;
};

DEV uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEV int swz(int row, int chunk) { return row * kRowB + ((chunk ^ (row & 7)) << 4); }

DEV void ldmatrix_x4(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEV void ldmatrix_x4_trans(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEV void ldmatrix_x2(uint32_t addr, uint32_t (&r)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
DEV void mma_bf16_(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEV uint4 ldg_nc_16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
DEV uint4 ldg_16(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
DEV uint32_t map_peer(uint32_t local, int rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(local), "r"(rank));
    return r;
}
DEV void st_async_v4(uint32_t remote, float4 v, uint32_t remote_bar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(remote), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w),
                 "r"(remote_bar)
                 : "memory");
}
DEV void st_async_v2(uint32_t remote, float2 v, uint32_t remote_bar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n" ::"r"(remote), "f"(v.x), "f"(v.y), "r"(remote_bar) : "memory");
}
DEV void mbar_wait_parity0(uint32_t bar) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
            " selp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(done)
            : "r"(bar)
            : "memory");
    }
}
DEV void cluster_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
DEV void cluster_wait_acquire() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

DEV const bf16* col_row(const AttnFParams& p, int m, int pos0, int c, bool before_wait) {
    const int pos = pos0 + m, wend = p.fixed_window ? pos0 - 1 : pos;
    if (c < kSwa) {
        const int ps = wend - (kSwa - 1) + c;
        if (ps < p.win_floor) return nullptr;
        if (before_wait && !((p.pre & 1) && ps < pos0)) return (const bf16*)1;
        return p.win + (size_t)ps * kDim;
    }
    if (c >= kSwa + p.ktop) return nullptr;
    if (before_wait && !(p.pre & 2)) return (const bf16*)1;
    const int j = p.topk[m * p.ktop + (c - kSwa)];
    return j >= 0 ? p.comp + (size_t)j * kDim : nullptr;
}
}

template <int H>
DEV void attn_sds_body(const Meta* meta, const AttnFParams& p) {
    using namespace sds;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    {
        uint32_t dyn;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
        if (dyn < sizeof(Smem<H>) + 128) __trap();
    }
    auto& sm = *reinterpret_cast<Smem<H>*>(((uintptr_t)smem_raw + 127) & ~(uintptr_t)127);
    const uint32_t bar = smem_u32(&sm.recv_bar);
    if (threadIdx.x == 0) {
        constexpr uint32_t kBytes = kSplits * (sizeof(float) * H * kSlice + sizeof(float2) * H);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(bar));
        asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(kBytes));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    cluster_arrive_release();

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int split = blockIdx.x, row = blockIdx.y;
    const int c0 = split * kChunk;
    const int pos0 = meta->pos;
    const bool live = row < meta->m;

    const bf16* my_src = (const bf16*)1;
    if (live && lane < kRowsPerWarp) my_src = col_row(p, row, pos0, c0 + warp + kWarps * lane, true);
    uint32_t defer = 0;
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
        const bf16* src = (const bf16*)__shfl_sync(0xffffffffu, (unsigned long long)my_src, i);
        const int t = warp + kWarps * i;
        if (src == (const bf16*)1) {
            defer |= 1u << i;
            continue;
        }
        uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
        if (src) {
            a = ldg_nc_16((const uint8_t*)src + lane * 32);
            b = ldg_nc_16((const uint8_t*)src + lane * 32 + 16);
        }
        *reinterpret_cast<uint4*>(sm.kv + swz(t, 2 * lane)) = a;
        *reinterpret_cast<uint4*>(sm.kv + swz(t, 2 * lane + 1)) = b;
        if (lane == 0) sm.masked[t] = src == nullptr;
    }

    pdl_wait();
    pdl_launch();
    if (!live) {
        cluster_wait_acquire();
        return;
    }

    if (defer) {
        const bf16* src_l = nullptr;
        if (lane < kRowsPerWarp && (defer >> lane & 1)) src_l = col_row(p, row, pos0, c0 + warp + kWarps * lane, false);
#pragma unroll
        for (int i = 0; i < kRowsPerWarp; ++i) {
            if (!(defer >> i & 1)) continue;
            const bf16* src = (const bf16*)__shfl_sync(0xffffffffu, (unsigned long long)src_l, i);
            const int t = warp + kWarps * i;
            uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
            if (src) {
                a = ldg_16((const uint8_t*)src + lane * 32);
                b = ldg_16((const uint8_t*)src + lane * 32 + 16);
            }
            *reinterpret_cast<uint4*>(sm.kv + swz(t, 2 * lane)) = a;
            *reinterpret_cast<uint4*>(sm.kv + swz(t, 2 * lane + 1)) = b;
            if (lane == 0) sm.masked[t] = src == nullptr;
        }
    }

    for (int i = tid; i < H * (kRowB / 16); i += kThreads) {
        const int h = i / (kRowB / 16), ch = i % (kRowB / 16);
        *reinterpret_cast<uint4*>(sm.q + swz(h, ch)) = ldg_16((const uint8_t*)(p.q + ((size_t)row * p.nh + h) * kDim) + ch * 16);
    }
    __syncthreads();

    if (warp < (kChunk / 16) * kQkParts) {
        const int mt = warp % (kChunk / 16);
        const int part = warp / (kChunk / 16);
        constexpr int kSteps = kDim / 16;
        const int k_lo = part * kSteps / kQkParts;
        const int k_hi = (part + 1) * kSteps / kQkParts;
        float acc[H / 8][4] = {};
        const int a_row = mt * 16 + (lane % 16);
        const int a_chunk = lane / 16;
        const int b_chunk = (lane / 8) % 2;
        for (int k = k_lo; k < k_hi; ++k) {
            uint32_t a[4];
            ldmatrix_x4(smem_u32(sm.kv + swz(a_row, 2 * k + a_chunk)), a);
#pragma unroll
            for (int n = 0; n < H / 8; ++n) {
                uint32_t b[2];
                ldmatrix_x2(smem_u32(sm.q + swz(n * 8 + lane % 8, 2 * k + b_chunk)), b);
                mma_bf16_(acc[n], a, b);
            }
        }
        const int g = lane / 4, tig = lane % 4;
#pragma unroll
        for (int n = 0; n < H / 8; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int t = mt * 16 + g + (i / 2) * 8;
                const int h = n * 8 + 2 * tig + (i % 2);
                sm.s[part][h][t] = acc[n][i];
            }
    }
    __syncthreads();

    const float scale_log2 = p.scale * kLog2e;
    for (int h = warp; h < H; h += kWarps) {
        float v[3];
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int t = lane + 32 * i;
            v[i] = -INFINITY;
            if (t < kChunk && !sm.masked[t]) {
                float acc = 0.f;
#pragma unroll
                for (int part = 0; part < kQkParts; ++part) acc += sm.s[part][h][t];
                v[i] = acc * scale_log2;
            }
            mx = fmaxf(mx, v[i]);
        }
#pragma unroll
        for (int o = 16; o > 0; o /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int t = lane + 32 * i;
            const float e = mx == -INFINITY ? 0.f : exp2f(v[i] - mx);
            sum += e;
            if (t < kChunk) sm.pt[h * kPStride + t] = __float2bfloat16(e);
        }
#pragma unroll
        for (int o = 16; o > 0; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) sm.ml[h] = make_float2(mx, sum);
    }
    cluster_wait_acquire();
    __syncthreads();
    if (tid < H * kSplits) {
        const int h = tid % H, peer = tid / H;
        st_async_v2(map_peer(smem_u32(&sm.recv_ml[split][h]), peer), sm.ml[h], map_peer(bar, peer));
    }

    {
        constexpr int kTiles = kDimsPerWarp / 16;
        float acc[kTiles][H / 8][4] = {};
#pragma unroll
        for (int k = 0; k < kChunk / 16; ++k) {
            uint32_t b[H / 8][2];
#pragma unroll
            for (int n = 0; n < H / 8; ++n) {
                const int b_row = n * 8 + (lane % 8);
                ldmatrix_x2(smem_u32(&sm.pt[b_row * kPStride + k * 16 + ((lane / 8) % 2) * 8]), b[n]);
            }
#pragma unroll
            for (int mt = 0; mt < kTiles; ++mt) {
                const int d0 = warp * kDimsPerWarp + mt * 16;
                const int t = k * 16 + (lane % 8) + (lane / 16) * 8;
                const int chunk = d0 / 8 + (lane / 8) % 2;
                uint32_t a[4];
                ldmatrix_x4_trans(smem_u32(sm.kv + swz(t, chunk)), a);
#pragma unroll
                for (int n = 0; n < H / 8; ++n) mma_bf16_(acc[mt][n], a, b[n]);
            }
        }
        float* stage = sm.stage[warp];
        const int g = lane / 4, tig = lane % 4;
#pragma unroll
        for (int mt = 0; mt < kTiles; ++mt)
#pragma unroll
            for (int n = 0; n < H / 8; ++n)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int d = mt * 16 + g + (i / 2) * 8;
                    const int h = n * 8 + 2 * tig + (i % 2);
                    stage[h * kDimsPerWarp + d] = acc[mt][n][i];
                }
        __syncwarp();
        const int owner = warp * kDimsPerWarp / kSlice;
        const int d_base = warp * kDimsPerWarp - owner * kSlice;
        const uint32_t peer_bar = map_peer(bar, owner);
        for (int v = lane; v < H * kDimsPerWarp / 4; v += 32) {
            const int h = v / (kDimsPerWarp / 4);
            const int d = (v % (kDimsPerWarp / 4)) * 4;
            st_async_v4(map_peer(smem_u32(&sm.recv[split][h][d_base + d]), owner), *reinterpret_cast<const float4*>(&stage[h * kDimsPerWarp + d]), peer_bar);
        }
    }
    mbar_wait_parity0(bar);

    for (int task = tid; task < H * kSlice / 4; task += kThreads) {
        const int h = task / (kSlice / 4);
        const int dl = (task % (kSlice / 4)) * 4, d = split * kSlice + dl;
        const float sink = p.sink[h] * kLog2e;
        float2 ml[kSplits];
        float mx = sink;
#pragma unroll
        for (int i = 0; i < kSplits; ++i) {
            ml[i] = sm.recv_ml[i][h];
            mx = fmaxf(mx, ml[i].x);
        }
        float denom = exp2f(sink - mx);
        float o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < kSplits; ++i) {
            const float c = ml[i].x == -INFINITY ? 0.f : exp2f(ml[i].x - mx);
            denom += c * ml[i].y;
            const float4 v = *reinterpret_cast<const float4*>(&sm.recv[i][h][dl]);
            o[0] += c * v.x;
            o[1] += c * v.y;
            o[2] += c * v.z;
            o[3] += c * v.w;
        }
        const float inv = 1.f / denom;
        float a[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) a[k] = __bfloat162float(__float2bfloat16(o[k] * inv));
        if (d >= kDim - 64) {
            const size_t ro = (size_t)(pos0 + row) * 32 + (d - (kDim - 64)) / 2;
            const float rc0 = p.cos[ro], rs0 = p.sin[ro], rc1 = p.cos[ro + 1], rs1 = p.sin[ro + 1];
            const float x0 = a[0] * rc0 + a[1] * rs0, x1 = a[1] * rc0 - a[0] * rs0;
            const float x2 = a[2] * rc1 + a[3] * rs1, x3 = a[3] * rc1 - a[2] * rs1;
            a[0] = x0;
            a[1] = x1;
            a[2] = x2;
            a[3] = x3;
        }
        const __nv_bfloat162 lo = __floats2bfloat162_rn(a[0], a[1]), hi = __floats2bfloat162_rn(a[2], a[3]);
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&lo);
        packed.y = *reinterpret_cast<const uint32_t*>(&hi);
        *reinterpret_cast<uint2*>(p.out + ((size_t)row * p.nh + h) * kDim + d) = packed;
    }
}

extern "C" __global__ void __launch_bounds__(sds::kThreads, 1) attn_sds_8(const Meta* meta, const __grid_constant__ AttnFParams p) { attn_sds_body<8>(meta, p); }
