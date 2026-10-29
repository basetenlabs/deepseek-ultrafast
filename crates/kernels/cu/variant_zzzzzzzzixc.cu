#include "variant_topk_cand.cu"

extern "C" __global__ void __launch_bounds__(1024, 1) topk_clc2_2_1024(const Meta* meta, const __grid_constant__ TopkParams t, const int* cand_ids, int ncand) {
    topk_clc_body<2, 1024>(meta, t, cand_ids, ncand);
}
