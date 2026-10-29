#include "variant_zzzztg3.cu"

#define TG3Y_E(NAME, KIND, VAR, O) \
    extern "C" __global__ void __launch_bounds__(128, 1) NAME(const __grid_constant__ Tg2Params p) { tg3_body<KIND, VAR, TG3_UNI | TG3_CHAIN | (O)>(p); }
TG3Y_E(tc_gemv3yr_qb, 0, V_QB, TG3_REL)
#undef TG3Y_E
