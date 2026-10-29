#define GEMV_XB 1
#include "gemv.cu"

__device__ __forceinline__ void gemv_fp8b_persist_body(const GemvParams& p) {
    constexpr int PH = 4, CONSUMERS = 4 * PH, TK = 32, STEP = 2048, MT = 8;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int M = p.M, K = p.K, nquads = p.N / 64;
    const int nsteps = K / TK, SS = p.stage_steps, nst = (nsteps + SS - 1) / SS;
    const int nqc = (nquads + (int)gridDim.x - 1) / (int)gridDim.x;
    uint8_t* ring = smem;
    uint64_t* full = (uint64_t*)(ring + (size_t)STAGES * SS * STEP);
    uint64_t* empty = full + STAGES;
    const int LDS = K + 8;
    uint16_t* xs = (uint16_t*)(empty + STAGES);
    float* red = (float*)(xs + MT * LDS);
    bf16* pws = (bf16*)(red + PH * MT * 64);
    uint8_t* wsc = (uint8_t*)(pws + K);
    __shared__ __align__(8) uint64_t xbar;
    __shared__ __align__(8) uint64_t gbar;
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CONSUMERS); }
        mbar_init(&xbar, 1);
        mbar_init(&gbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == CONSUMERS) {
        if (lane == 0) {
            int g = 0;
            for (int q = blockIdx.x; q < nquads; q += gridDim.x) {
                const uint8_t* wsrc = p.w + (size_t)q * nsteps * STEP;
                for (int st = 0; st < nst; ++st, ++g) {
                    const int s = g % STAGES;
                    if (g >= STAGES) mbar_wait(&empty[s], ((g / STAGES) - 1) & 1);
                    const uint32_t bytes = (uint32_t)min(SS, nsteps - st * SS) * STEP;
                    mbar_expect_tx(&full[s], bytes);
                    tma_load_1d(ring + (size_t)s * SS * STEP, wsrc + (size_t)st * SS * STEP, bytes, &full[s]);
                }
            }
        }
        pdl_wait();
        pdl_launch();
        return;
    }
    if (tid == 0 && p.prenorm_w) {
        mbar_expect_tx(&gbar, (uint32_t)K * 2);
        tma_load_1d_en(pws, p.prenorm_w, K * 2, &gbar);
    }
    for (int i = tid; i < nqc * 2 * nsteps; i += CONSUMERS * 32) {
        const int qi = i / (2 * nsteps), r = i % (2 * nsteps), q = blockIdx.x + qi * gridDim.x;
        wsc[i] = q < nquads ? __ldg(p.ws + (size_t)(q * 2 + r / nsteps) * nsteps + r % nsteps) : 0;
    }
    pdl_wait();
    pdl_launch();
    if (tid == 0) {
        mbar_expect_tx(&xbar, (uint32_t)M * K * 2);
        for (int m = 0; m < M; ++m) tma_load_1d_en(xs + m * LDS, p.x + (size_t)m * p.ldx, K * 2, &xbar);
    }
    float* rs = red;
    if (p.prenorm_w && p.norm_part) {
        for (int m = tid; m < MT; m += CONSUMERS * 32) {
            float ss = 0.f;
            if (m < M) ss = norm_part_sum(p.norm_part, m, p.norm_nslices);
            rs[m] = rsqrtf(ss / p.K + p.prenorm_eps);
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    }
    mbar_wait(&xbar, 0);
    if (p.prenorm_w) mbar_wait(&gbar, 0);
    for (int i = tid; i < MT * (K / 8); i += CONSUMERS * 32) {
        const int m = i / (K / 8), c = (i % (K / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (m < M) {
            v = *(const uint4*)(xs + m * LDS + c);
            if (p.prenorm_w) {
                bf16* h = (bf16*)&v;
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    h[e] = __float2bfloat16(__bfloat162float(__float2bfloat16(__bfloat162float(pws[c + e]) * (__bfloat162float(h[e]) * rs[m]))));
            }
        }
        *(uint4*)(xs + m * LDS + c) = v;
    }
    asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    const int g8 = lane >> 2, t = lane & 3;
    const int rg = warp % 4, ph = warp / 4;
    int g = 0, qi = 0;
    for (int q = blockIdx.x; q < nquads; q += gridDim.x, ++qi) {
        const uint8_t* qs = wsc + (size_t)(qi * 2 + rg / 2) * nsteps;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int st = 0; st < nst; ++st, ++g) {
            const int s = g % STAGES;
            mbar_wait(&full[s], (g / STAGES) & 1);
            const int steps = min(SS, nsteps - st * SS);
            const uint8_t* base = ring + (size_t)s * SS * STEP;
            for (int j = ph; j < steps; j += PH) {
                const uint4 v = *(const uint4*)(base + (size_t)j * STEP + rg * 512 + lane * 16);
                const int ks = st * SS + j, kl = ks * TK;
                const float sw = e8m0_to_f32(qs[ks]);
                const uint32_t a0[4] = {e4m3x2_to_bf16x2(v.x & 0xffff), e4m3x2_to_bf16x2(v.z & 0xffff),
                                        e4m3x2_to_bf16x2(v.x >> 16), e4m3x2_to_bf16x2(v.z >> 16)};
                const uint32_t a1[4] = {e4m3x2_to_bf16x2(v.y & 0xffff), e4m3x2_to_bf16x2(v.w & 0xffff),
                                        e4m3x2_to_bf16x2(v.y >> 16), e4m3x2_to_bf16x2(v.w >> 16)};
                const uint16_t* xb = xs + g8 * LDS + kl;
                float c[4] = {0.f, 0.f, 0.f, 0.f};
                mma_bf16(c, a0, *(const uint32_t*)(xb + 2 * t), *(const uint32_t*)(xb + 8 + 2 * t));
                mma_bf16(c, a1, *(const uint32_t*)(xb + 16 + 2 * t), *(const uint32_t*)(xb + 24 + 2 * t));
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[e] = fmaf(c[e], sw, acc[e]);
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty[s]);
        }
        float* rp = red + (size_t)ph * MT * 64;
        {
            const int m0 = 2 * t, r = rg * 16 + g8;
            rp[m0 * 64 + r] = acc[0];
            rp[(m0 + 1) * 64 + r] = acc[1];
            rp[m0 * 64 + r + 8] = acc[2];
            rp[(m0 + 1) * 64 + r + 8] = acc[3];
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        for (int i = tid; i < MT * 64; i += CONSUMERS * 32) {
            float v = red[i];
#pragma unroll
            for (int k = 1; k < PH; ++k) v += red[(size_t)k * MT * 64 + i];
            red[i] = v;
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
        for (int i = tid; i < M * 64; i += CONSUMERS * 32) {
            const int m = i / 64, r = q * 64 + i % 64;
            ((float*)p.out)[(size_t)m * p.ldo + r] = red[i] * p.alpha;
        }
        if (p.sp && warp < M) {
            __syncwarp();
            const int m = warp;
            const float inv_t = 1.f / p.sp->temp;
            float v[2];
            int id[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = q * 64 + lane + 32 * h;
                v[h] = r < p.sp_vloc ? red[m * 64 + lane + 32 * h] * p.alpha * inv_t : -INFINITY;
                id[h] = r + p.sp_off;
            }
            const float v0 = v[0], v1 = v[1];
            float* rec = p.spart + ((size_t)m * nquads + q) * 20;
            float mx = -INFINITY;
#pragma unroll 1
            for (int k = 0; k < 9; ++k) {
                const int hb = v[1] > v[0] ? 1 : 0;
                const float lv = hb ? v[1] : v[0];
                const int li = hb ? id[1] : id[0];
                const unsigned u = __float_as_uint(lv);
                const unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                const unsigned kmax = __reduce_max_sync(0xffffffffu, key);
                const int bi = (int)__reduce_min_sync(0xffffffffu, key == kmax ? (unsigned)li : 0x7fffffffu);
                const float bv = __uint_as_float((kmax & 0x80000000u) ? (kmax & 0x7fffffffu) : ~kmax);
                if (k == 0) mx = bv;
                if (lane == 0) {
                    if (k < 8) { rec[3 + k] = bv; rec[11 + k] = __int_as_float(bi); } else rec[2] = bv;
                }
                v[0] = id[0] == bi ? -INFINITY : v[0];
                v[1] = id[1] == bi ? -INFINITY : v[1];
            }
            const float se = warp_sum((v0 == -INFINITY ? 0.f : expf(v0 - mx)) + (v1 == -INFINITY ? 0.f : expf(v1 - mx)));
            if (lane == 0) { rec[0] = mx; rec[1] = se; }
        }
        asm volatile("bar.sync 1, %0;" ::"r"(CONSUMERS * 32));
    }
}
extern "C" __global__ void __launch_bounds__(17 * 32) gemv_fp8b_persist_p4(const __grid_constant__ GemvParams p) { gemv_fp8b_persist_body(p); }
