#include "common.cuh"
#include "route.cuh"

extern "C" __global__ void vision_norm(const bf16* x, const float* weight, bf16* y, int dim) {
    const int row = blockIdx.x;
    float ss = 0.f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) { float v = __bfloat162float(x[row * dim + j]); ss += v * v; }
    for (int d = 16; d; d >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, d);
    __shared__ float sums[8];
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = ss;
    __syncthreads();
    float sum = 0.f;
    for (int i = 0; i < 8; ++i) sum += sums[i];
    const float scale = rsqrtf(sum / dim + 1e-6f);
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        y[row * dim + j] = __float2bfloat16_rn((__bfloat162float(x[row * dim + j]) * scale) * weight[j]);
}

extern "C" __global__ void vision_linear_finish(const float* x, const bf16* bias, bf16* y, const bf16* residual, int count, int dim, int gelu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    bf16 v = __float2bfloat16_rn(x[i] + (bias ? __bfloat162float(bias[i % dim]) : 0.f));
    if (gelu) v = __float2bfloat16_rn(.5f * __bfloat162float(v) * (1.f + erff(__bfloat162float(v) * .7071067811865475f)));
    if (residual) v = __float2bfloat16_rn(__bfloat162float(v) + __bfloat162float(residual[i]));
    y[i] = v;
}
extern "C" __global__ void vision_swiglu(const bf16* x, bf16* y, int count, int inter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int base = (i / inter) * 2 * inter + i % inter;
    float g = __bfloat162float(x[base]);
    bf16 s = __float2bfloat16_rn(g / (1.f + expf(-g)));
    y[i] = __float2bfloat16_rn(__bfloat162float(s) * __bfloat162float(x[base + inter]));
}
extern "C" __global__ void vision_rope(bf16* qkv, const float* cs, int count, int heads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int j = i % 32, h = (i / 32) % heads, qk = (i / (32 * heads)) % 2, row = i / (64 * heads);
    int off = row * (3 * heads * 64) + qk * heads * 64 + h * 64 + j;
    float a = __bfloat162float(qkv[off]), b = __bfloat162float(qkv[off + 32]);
    float c = cs[row * 64 + j], s = cs[row * 64 + 32 + j];
    qkv[off] = __float2bfloat16_rn(__fmul_rn(a,c) - __fmul_rn(b,s));
    qkv[off + 32] = __float2bfloat16_rn(__fmul_rn(b,c) + __fmul_rn(a,s));
}

extern "C" __global__ void vision_unfold(const bf16* x, bf16* y, int h, int w, int dim) {
    int nh = (h + 2) / 3, nw = (w + 2) / 3, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nh * nw * dim * 9) return;
    int tile = i / (dim * 9), d = (i / 9) % dim, pos = i % 9;
    int yy = (tile / nw) * 3 + pos / 3, xx = (tile % nw) * 3 + pos % 3;
    y[i] = yy < h && xx < w ? x[(yy * w + xx) * dim + d] : __float2bfloat16(0);
}
extern "C" __global__ void vision_pack(const bf16* x, bf16* y, const bf16* start, const bf16* newline, const bf16* end, int h, int w, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, len = h * (w + 1) + 2;
    if (i >= len * dim) return;
    int row = i / dim, d = i % dim;
    const bf16* p = row == 0 ? start : row == len - 1 ? end : (row - 1) % (w + 1) == w ? newline
        : x + (((row - 1) / (w + 1)) * w + (row - 1) % (w + 1)) * dim;
    y[i] = p[d];
}
extern "C" __global__ void vision_embed(const bf16* rows, bf16* h, int* cmp_hist, int count, int dim, int hc) {
    pdl_wait(); pdl_launch();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * dim * hc) h[i] = rows[(i / (dim * hc)) * dim + i % dim];
    if (i < count && cmp_hist) cmp_hist[i] = -1;
}
extern "C" __global__ void vision_route(int m, const float* scores, const float* bias, int n_exp, int topk, float scale,
    int* ids, float* weights, const float* bias_vl, const int2* spans, int nspans, int pos) {
    pdl_wait(); pdl_launch();
    __shared__ int sel[8][8]; __shared__ float selw[8][8];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, row = blockIdx.x * 8 + warp;
    bool image = false;
    for (int i = 0; i < nspans; ++i) image |= pos + row >= spans[i].x && pos + row < spans[i].y;
    if (row < m) route_token(row, lane, scores, image ? bias_vl : bias, n_exp, topk, scale, sel[warp], selw[warp], ids, weights);
}
