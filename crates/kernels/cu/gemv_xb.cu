#define GEMV_XB 1
#include "gemv.cu"

#define XB_ENTRY(name, PH)                                                                                      \
    extern "C" __global__ void __launch_bounds__((4 * PH + 1) * 32, GEMV_MIN_CTAS_PER_SM(PH)) name(const __grid_constant__ GemvParams p) { \
        gemv_body<FMT_BF16, 1, PH>(p);                                                                            \
    }
XB_ENTRY(gemv_bf16_xb_n1_p1, 1)
XB_ENTRY(gemv_bf16_xb_n1_p2, 2)
XB_ENTRY(gemv_bf16_xb_n1_p4, 4)
