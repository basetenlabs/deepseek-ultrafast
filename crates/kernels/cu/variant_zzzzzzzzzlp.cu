#define COMM_DEC_NO_ENTRIES 1
#include "variant_zcomm_dec.cu"

DEV float4 lp_ld(const float* p) {
    float4 v;
    asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
    return v;
}
DEV bool lp_miss(float4 v) { return !arrived(v); }
DEV uint32_t lp_ld_acq(const uint32_t* s) {
    uint32_t v;
    asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(sa_(s)) : "memory");
    return v;
}
DEV void lp_st_rel(uint32_t* s, uint32_t v) { asm volatile("st.release.cta.shared::cta.u32 [%0], %1;" ::"r"(sa_(s)), "r"(v) : "memory"); }

template <int NS, int G>
DEV void comm_lp_body(const Meta* meta, const NvlsParams& q) {
    static_assert(NS >= 1 && NS <= 8 && (G == 1 || G == 2 || G == 4), "comm_lp_body");
    constexpr int TPC = 4;
    const CommParams& p = q.c;
    __shared__ uint32_t s_waited;
    __shared__ uint32_t s_flag[G];
    __shared__ float s_red[G][2];
    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
    if (tid == 0) s_waited = 0u;
    if (tid < G) s_flag[tid] = 0u;
    __syncthreads();
    const int dw = warp - 1, g = dw >> 1, w = dw & 1, t = w * 32 + lane;
    const int vb = (int)blockIdx.x * G + g;
    const int npt = p.n_per_token;
    uint32_t e = 0u;
    int n4 = 0;
    if (warp == 0) {
        pdl_wait();
        __syncwarp();
        if (lane == 0) lp_st_rel(&s_waited, 1u);
        pdl_launch();
    } else {
        const int i4 = vb * 64 + t, i = i4 * 4, m = i / npt, c = i % npt;
        const int M = meta->m;
        n4 = M * npt / 4;
        const bool act = i4 < n4;
        const uint4 st = *(const uint4*)(p.state + vb * 4);
        e = comm_epoch_next(st.x);
        const int stage = comm_stage(e), prev = comm_prev_stage(stage);
        const size_t slot = (size_t)TPC * p.cap;
        float* const mine = p.recv[p.rank];
        float* const r0 = (float*)p.src + (size_t)(m * NS) * p.ld_src + c;
        float* const r2 = (float*)p.src2 + (size_t)m * p.ld_src + c;
        float4 a[NS], a2;
        if (act) {
#pragma unroll
            for (int j = 0; j < NS; ++j) a[j] = lp_ld(r0 + (size_t)j * p.ld_src);
            a2 = lp_ld(r2);
            unsigned miss = 0;
#pragma unroll
            for (int j = 0; j < NS; ++j) if (lp_miss(a[j])) miss |= 1u << j;
            if (lp_miss(a2)) miss |= 1u << NS;
            while (__any_sync(0xffffffffu, miss != 0) && !__any_sync(0xffffffffu, lp_ld_acq(&s_waited) != 0)) {
#pragma unroll
                for (int j = 0; j < NS; ++j) if (miss & (1u << j)) a[j] = lp_ld(r0 + (size_t)j * p.ld_src);
                if (miss & (1u << NS)) a2 = lp_ld(r2);
                miss = 0;
#pragma unroll
                for (int j = 0; j < NS; ++j) if (lp_miss(a[j])) miss |= 1u << j;
                if (lp_miss(a2)) miss |= 1u << NS;
            }
            if (__any_sync(0xffffffffu, miss != 0)) {
                pdl_wait();
#pragma unroll
                for (int j = 0; j < NS; ++j) if (miss & (1u << j)) a[j] = lp_ld(r0 + (size_t)j * p.ld_src);
                if (miss & (1u << NS)) a2 = lp_ld(r2);
            }
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int j = 0; j < NS; ++j) { v.x += a[j].x; v.y += a[j].y; v.z += a[j].z; v.w += a[j].w; }
            v.x += a2.x; v.y += a2.y; v.z += a2.z; v.w += a2.w;
            v = make_float4(canon(v.x), canon(v.y), canon(v.z), canon(v.w));
#pragma unroll
            for (int r = 0; r < TPC; ++r) st_remote_v4(p.recv[r] + stage * slot + (size_t)p.rank * p.cap + i, v);
        }
        {
            float* pb = mine + prev * slot;
            for (int j = (int)st.y + t; j < (int)st.z; j += 64)
#pragma unroll
                for (int s = 0; s < TPC; ++s) *(float4*)(pb + (size_t)s * p.cap + (size_t)j * 4) = make_float4(-0.f, -0.f, -0.f, -0.f);
        }
        if (q.mixf) {
            if (act) {
                for (;;) {
                    uint32_t f;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(q.mixf + m) : "memory");
                    if (__all_sync(0xffffffffu, f != 0u)) break;
                }
            }
        } else {
            pdl_wait();
        }
        if (act) {
            HcIn hin;
            hc_in_load(p, m, c, npt, hin);
            float pr[4];
            const bool hp = p.e.pre != nullptr;
            if (hp) {
                const float4 tq = *(const float4*)(p.e.pre + m * HC);
                pr[0] = tq.x; pr[1] = tq.y; pr[2] = tq.z; pr[3] = tq.w;
            }
            float s2p[4][4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float s2v = 0.f;
#pragma unroll
                    for (int h = 0; h < 4; ++h) s2v += hin.cb[h][j] * hin.rr[h][k];
                    s2p[j][k] = s2v;
                }
            const float* rb = mine + stage * slot;
            float4 parts[TPC];
#pragma unroll
            for (int s = 0; s < TPC; ++s) parts[s] = ld_poll_v4(rb + (size_t)s * p.cap + i);
            unsigned miss = 0;
#pragma unroll
            for (int s = 0; s < TPC; ++s) if (!arrived(parts[s])) miss |= 1u << s;
            while (miss) {
#pragma unroll
                for (int s = 0; s < TPC; ++s) if (miss & (1u << s)) parts[s] = ld_poll_v4(rb + (size_t)s * p.cap + i);
#pragma unroll
                for (int s = 0; s < TPC; ++s) if ((miss & (1u << s)) && arrived(parts[s])) miss &= ~(1u << s);
            }
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int s = 0; s < TPC; ++s) {
                const float vv[4] = {parts[s].x, parts[s].y, parts[s].z, parts[s].w};
#pragma unroll
                for (int k = 0; k < 4; ++k) acc[k] += vv[k];
            }
            const int dim = npt;
            float hn[4][4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float x = __bfloat162float(__float2bfloat16(acc[k]));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float s2v;
                    s2v = s2p[j][k];
                    hn[j][k] = __bfloat162float(__float2bfloat16(hin.po[j] * x + s2v));
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint2 o;
                o.x = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][0])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][1])) << 16);
                o.y = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][2])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][3])) << 16);
                *(uint2*)((bf16*)p.out + ((size_t)m * 4 + j) * dim + c) = o;
            }
            float xr[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float y = 0.f;
                if (hp) {
#pragma unroll
                    for (int h = 0; h < HC; ++h) y += pr[h] * hn[h][k];
                } else {
                    y = hn[0][k];
                }
                xr[k] = __bfloat162float(__float2bfloat16(y));
            }
            *(__nv_bfloat162*)(p.e.x_raw + (size_t)m * p.e.dim + c) = __floats2bfloat162_rn(xr[0], xr[1]);
            *(__nv_bfloat162*)(p.e.x_raw + (size_t)m * p.e.dim + c + 2) = __floats2bfloat162_rn(xr[2], xr[3]);
            float sh = 0.f, sx = 0.f;
#pragma unroll
            for (int h = 0; h < HC; ++h)
#pragma unroll
                for (int k = 0; k < 4; ++k) sh += hn[h][k] * hn[h][k];
#pragma unroll
            for (int k = 0; k < 4; ++k) sx += xr[k] * xr[k];
            __syncwarp();
            sh = warp_sum(sh);
            sx = warp_sum(sx);
            if (w == 1) {
                if (lane == 0) {
                    s_red[g][0] = sh;
                    s_red[g][1] = sx;
                    lp_st_rel(&s_flag[g], 1u);
                }
            } else if (lane < 2) {
                while (!lp_ld_acq(&s_flag[g])) {}
                p.e.part[((size_t)m * (dim / SLICE) + c / SLICE) * NP + MIX + lane] = (lane == 0 ? sh : sx) + s_red[g][lane];
            }
        }
        const float4 z = make_float4(-0.f, -0.f, -0.f, -0.f);
#pragma unroll
        for (int j = 0; j < NS; ++j) *(float4*)(r0 + (size_t)j * p.ld_src) = z;
        *(float4*)r2 = z;
    }
    __syncthreads();
    if (warp > 0 && t == 0) {
        const int lo = vb * 64, hi = min(n4, lo + 64);
        *(uint4*)(p.state + vb * 4) = make_uint4(e, (uint32_t)lo, (uint32_t)(hi > lo ? hi : lo), 0u);
    }
}

DEV uint2 lp_ld2(const void* p) {
    uint2 v;
    asm volatile("ld.relaxed.gpu.global.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
    return v;
}
DEV bool lp_miss_bf(uint2 v) {
    return (v.x & 0xffffu) == 0x8000u || (v.x >> 16) == 0x8000u || (v.y & 0xffffu) == 0x8000u || (v.y >> 16) == 0x8000u;
}
template <int TOPK, int G, int TPC = 4>
DEV void comm_lp_sgl_body(const Meta* meta, const NvlsParams& q) {
    static_assert(TOPK >= 1 && TOPK <= 8 && (G == 1 || G == 2 || G == 4) && (TPC == 4 || TPC == 8), "comm_lp_sgl_body");
    constexpr int NR = TOPK + 1;
    const CommParams& p = q.c;
    __shared__ uint32_t s_waited;
    __shared__ uint32_t s_flag[G];
    __shared__ float s_red[G][2];
    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
    if (tid == 0) s_waited = 0u;
    if (tid < G) s_flag[tid] = 0u;
    __syncthreads();
    const int dw = warp - 1, g = dw >> 1, w = dw & 1, t = w * 32 + lane;
    const int vb = (int)blockIdx.x * G + g;
    const int npt = p.n_per_token;
    uint32_t e = 0u;
    int n4 = 0;
    if (warp == 0) {
        if (q.sg_early) pdl_launch();
        pdl_wait();
        __syncwarp();
        if (lane == 0) lp_st_rel(&s_waited, 1u);
        if (!q.sg_early) pdl_launch();
    } else {
        const int i4 = vb * 64 + t, i = i4 * 4, m = i / npt, c = i % npt;
        const int M = meta->m;
        n4 = M * npt / 4;
        const bool act = i4 < n4;
        const uint4 st = *(const uint4*)(p.state + vb * 4);
        e = comm_epoch_next(st.x);
        const int stage = comm_stage(e), prev = comm_prev_stage(stage);
        const size_t slot = (size_t)TPC * p.cap;
        float* const mine = p.recv[p.rank];
        const __nv_bfloat16* rows_base = (const __nv_bfloat16*)p.src;
        const size_t wi = (size_t)(i / 256) * 256 + (i % 256) / 2;
        int rr[NR];
        float wk[TOPK];
        const __nv_bfloat16* shp = nullptr;
        if (act) {
#pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                rr[k] = __ldcg(q.sg_perm + m * TOPK + k);
                wk[k] = __ldcg(q.sg_w + m * TOPK + k);
            }
            rr[TOPK] = __ldcg(q.sg_rec + 1) * 8 + m;
            shp = rows_base + (size_t)rr[TOPK] * npt + c;
            auto rowp = [&](int k) { return k == TOPK ? shp : rows_base + (size_t)rr[k] * npt + c; };
            uint2 a[NR];
#pragma unroll
            for (int k = 0; k < NR; ++k) a[k] = rr[k] >= 0 ? lp_ld2(rowp(k)) : make_uint2(0u, 0u);
            unsigned miss = 0;
#pragma unroll
            for (int k = 0; k < NR; ++k) if (rr[k] >= 0 && lp_miss_bf(a[k])) miss |= 1u << k;
            while (__any_sync(0xffffffffu, miss != 0) && !__any_sync(0xffffffffu, lp_ld_acq(&s_waited) != 0)) {
#pragma unroll
                for (int k = 0; k < NR; ++k) if (miss & (1u << k)) a[k] = lp_ld2(rowp(k));
                unsigned still = 0;
#pragma unroll
                for (int k = 0; k < NR; ++k) if ((miss & (1u << k)) && lp_miss_bf(a[k])) still |= 1u << k;
                miss = still;
            }
            if (__any_sync(0xffffffffu, miss != 0)) {
                pdl_wait();
#pragma unroll
                for (int k = 0; k < NR; ++k) if (miss & (1u << k)) a[k] = lp_ld2(rowp(k));
            }
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                if (rr[k] < 0) continue;
                const float x[4] = {__uint_as_float(a[k].x << 16), __uint_as_float(a[k].x & 0xffff0000u), __uint_as_float(a[k].y << 16), __uint_as_float(a[k].y & 0xffff0000u)};
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[j] = fmaf(x[j], wk[k], acc[j]);
            }
            const float sh[4] = {__uint_as_float(a[TOPK].x << 16), __uint_as_float(a[TOPK].x & 0xffff0000u), __uint_as_float(a[TOPK].y << 16), __uint_as_float(a[TOPK].y & 0xffff0000u)};
            float vv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) vv[j] = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(acc[j])) + sh[j]));
            const uint32_t w0 = pk_bf2(vv[0], vv[1]), w1 = pk_bf2(vv[2], vv[3]);
#pragma unroll
            for (int r = 0; r < TPC; ++r)
                asm volatile("st.relaxed.sys.global.v2.u32 [%0], {%1,%2};" ::"l"((uint32_t*)(p.recv[r] + stage * slot + (size_t)p.rank * p.cap) + wi), "r"(w0), "r"(w1) : "memory");
        }
        {
            float* pb = mine + prev * slot;
            for (int j = (int)st.y + t; j < (int)st.z; j += 64)
#pragma unroll
                for (int s = 0; s < TPC; ++s) *(float4*)(pb + (size_t)s * p.cap + (size_t)j * 4) = make_float4(-0.f, -0.f, -0.f, -0.f);
        }
        pdl_wait();
        if (act) {
            HcIn hin;
            hc_in_load(p, m, c, npt, hin);
            float pr[4];
            const bool hp = p.e.pre != nullptr;
            if (hp) {
                const float4 tq = *(const float4*)(p.e.pre + m * HC);
                pr[0] = tq.x; pr[1] = tq.y; pr[2] = tq.z; pr[3] = tq.w;
            }
            float s2p[4][4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float s2v = 0.f;
#pragma unroll
                    for (int h = 0; h < 4; ++h) s2v += hin.cb[h][j] * hin.rr[h][k];
                    s2p[j][k] = s2v;
                }
            const float* rb = mine + stage * slot;
            float4 parts[TPC];
            {
                uint2 wv[TPC];
#pragma unroll
                for (int s = 0; s < TPC; ++s) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(wv[s].x), "=r"(wv[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
                unsigned rm = 0;
#pragma unroll
                for (int s = 0; s < TPC; ++s) if (!(arr_bf2(wv[s].x) && arr_bf2(wv[s].y))) rm |= 1u << s;
                while (rm) {
#pragma unroll
                    for (int s = 0; s < TPC; ++s)
                        if (rm & (1u << s)) asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(wv[s].x), "=r"(wv[s].y) : "l"((const uint32_t*)(rb + (size_t)s * p.cap) + wi) : "memory");
#pragma unroll
                    for (int s = 0; s < TPC; ++s) if ((rm & (1u << s)) && arr_bf2(wv[s].x) && arr_bf2(wv[s].y)) rm &= ~(1u << s);
                }
#pragma unroll
                for (int s = 0; s < TPC; ++s) parts[s] = make_float4(__uint_as_float(wv[s].x << 16), __uint_as_float(wv[s].x & 0xffff0000u), __uint_as_float(wv[s].y << 16), __uint_as_float(wv[s].y & 0xffff0000u));
            }
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int s = 0; s < TPC; ++s) {
                const float vv[4] = {parts[s].x, parts[s].y, parts[s].z, parts[s].w};
#pragma unroll
                for (int k = 0; k < 4; ++k) acc[k] += vv[k];
            }
            const int dim = npt;
            float hn[4][4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float x = __bfloat162float(__float2bfloat16(acc[k]));
#pragma unroll
                for (int j = 0; j < 4; ++j) hn[j][k] = __bfloat162float(__float2bfloat16(hin.po[j] * x + s2p[j][k]));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint2 o;
                o.x = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][0])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][1])) << 16);
                o.y = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][2])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hn[j][3])) << 16);
                *(uint2*)((bf16*)p.out + ((size_t)m * 4 + j) * dim + c) = o;
            }
            float xr[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float y = 0.f;
                if (hp) {
#pragma unroll
                    for (int h = 0; h < HC; ++h) y += pr[h] * hn[h][k];
                } else {
                    y = hn[0][k];
                }
                xr[k] = __bfloat162float(__float2bfloat16(y));
            }
            *(__nv_bfloat162*)(p.e.x_raw + (size_t)m * p.e.dim + c) = __floats2bfloat162_rn(xr[0], xr[1]);
            *(__nv_bfloat162*)(p.e.x_raw + (size_t)m * p.e.dim + c + 2) = __floats2bfloat162_rn(xr[2], xr[3]);
            if (q.sg_img) {
                const uint2 gw = *(const uint2*)(q.sg_gamma + c);
                const float g4[4] = {__uint_as_float(gw.x << 16), __uint_as_float(gw.x & 0xffff0000u), __uint_as_float(gw.y << 16), __uint_as_float(gw.y & 0xffff0000u)};
                float v[4], a = 0.f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    v[k] = g4[k] * xr[k];
                    a = fmaxf(a, fabsf(v[k]));
                }
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4));
                const float sc = round_scale_pow2(fmaxf(a, 1e-4f), 1.0f / 448.0f);
                const float inv = __uint_as_float(0x7f000000u - __float_as_uint(sc));
                const uint32_t w = (uint32_t)f32x2_to_e4m3x2(v[0] * inv, v[1] * inv) | ((uint32_t)f32x2_to_e4m3x2(v[2] * inv, v[3] * inv) << 16);
                const int tt = c >> 7, kk = c & 127;
                *(uint32_t*)(q.sg_img + (size_t)tt * 1024 + (kk >> 4) * 128 + m * 16 + (kk & 15)) = w;
                if ((c & 31) == 0) q.sg_img[(size_t)(npt / 128) * 1024 + (size_t)tt * 512 + m * 16 + (kk >> 5)] = (uint8_t)(__float_as_uint(sc) >> 23);
            }
            float sh = 0.f, sx = 0.f;
#pragma unroll
            for (int h = 0; h < HC; ++h)
#pragma unroll
                for (int k = 0; k < 4; ++k) sh += hn[h][k] * hn[h][k];
#pragma unroll
            for (int k = 0; k < 4; ++k) sx += xr[k] * xr[k];
            __syncwarp();
            sh = warp_sum(sh);
            sx = warp_sum(sx);
            if (w == 1) {
                if (lane == 0) {
                    s_red[g][0] = sh;
                    s_red[g][1] = sx;
                    lp_st_rel(&s_flag[g], 1u);
                }
            } else if (lane < 2) {
                while (!lp_ld_acq(&s_flag[g])) {}
                p.e.part[((size_t)m * (dim / SLICE) + c / SLICE) * NP + MIX + lane] = (lane == 0 ? sh : sx) + s_red[g][lane];
            }
#pragma unroll
            for (int k = 0; k < NR; ++k)
                if (rr[k] >= 0) *(uint2*)(k == TOPK ? (__nv_bfloat16*)shp : (__nv_bfloat16*)p.src + (size_t)rr[k] * npt + c) = make_uint2(0x80008000u, 0x80008000u);
        }
    }
    __syncthreads();
    if (warp > 0 && t == 0) {
        const int lo = vb * 64, hi = min(n4, lo + 64);
        *(uint4*)(p.state + vb * 4) = make_uint4(e, (uint32_t)lo, (uint32_t)(hi > lo ? hi : lo), 0u);
    }
}
extern "C" __global__ void __launch_bounds__(64 * 2 + 32) comm_lp_sgl_6_2(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_lp_sgl_body<6, 2>(meta, q); }
extern "C" __global__ void __launch_bounds__(64 * 2 + 32) comm_lp_sgl8_6_2(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_lp_sgl_body<6, 2, 8>(meta, q); }
extern "C" __global__ void __launch_bounds__(64 * 2 + 32) comm_lp_sgl8_3_2(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_lp_sgl_body<3, 2, 8>(meta, q); }
extern "C" __global__ void __launch_bounds__(64 * 2 + 32) comm_lp_ffn_6_2zp(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_lp_body<6, 2>(meta, q); }
extern "C" __global__ void __launch_bounds__(64 * 2 + 32) comm_lp_ffn_3_2zp(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_lp_body<3, 2>(meta, q); }
