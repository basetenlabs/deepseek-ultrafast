#define IDX_SPLIT 1
#include "tcgemm.cu"

DEV uint32_t ixm_fkey(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
DEV unsigned long long ixm_min64(unsigned long long v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long w = __shfl_xor_sync(0xffffffffu, v, o);
        v = w < v ? w : v;
    }
    return v;
}

extern "C" __global__ void __launch_bounds__(1024) idx_merge(const MetaT* meta, const float* scores, int ld, int ratio, const int* told, const uint32_t* okeys, int k, int* out) {
    pdl_wait();
    pdl_launch();
    const int m = blockIdx.x;
    if (m >= meta->m) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int pos = meta->pos;
    const int n_old = pos / ratio, n_m = (pos + m + 1) / ratio;
    const int nnew = n_m > n_old ? n_m - n_old : 0;
    int idx = -1;
    uint32_t key = 0;
    if (tid < k) {
        idx = told[(size_t)m * k + tid];
        key = okeys[(size_t)m * k + tid];
    } else if (tid < k + nnew) {
        idx = n_old + (tid - k);
        key = ixm_fkey(scores[(size_t)m * ld + idx]);
    }
    const bool valid = idx >= 0;
    const unsigned long long me = valid ? (((unsigned long long)key << 32) | (uint32_t)~(uint32_t)idx) : ~0ull;
    const int nv = __syncthreads_count(valid);
    const int D = nv - k;
    bool keep = valid;
    __shared__ unsigned long long wl[32][16];
    __shared__ unsigned long long s_thr;
    if (D > 0) {
        unsigned long long v = me;
        for (int r = 0; r < D; ++r) {
            const unsigned long long w = ixm_min64(v);
            if (lane == 0) wl[warp][r] = w;
            if (v == w) v = ~0ull;
        }
        __syncthreads();
        if (warp == 0) {
            int h = 0;
            unsigned long long t = 0;
            for (int r = 0; r < D; ++r) {
                const unsigned long long head = h < D ? wl[lane][h] : ~0ull;
                const unsigned long long w = ixm_min64(head);
                if (head == w && w != ~0ull) ++h;
                t = w;
            }
            if (lane == 0) s_thr = t;
        }
        __syncthreads();
        keep = valid && me > s_thr;
    }
    __shared__ int wcnt[32];
    const unsigned bal = __ballot_sync(0xffffffffu, keep);
    if (lane == 0) wcnt[warp] = __popc(bal);
    __syncthreads();
    int base = 0;
    for (int w = 0; w < warp; ++w) base += wcnt[w];
    if (keep) out[(size_t)m * k + base + __popc(bal & ((1u << lane) - 1))] = idx;
    const int kept = nv < k ? nv : k;
    for (int i = kept + tid; i < k; i += blockDim.x) out[(size_t)m * k + i] = -1;
}
