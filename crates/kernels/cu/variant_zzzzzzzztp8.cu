#define COMM_DEC_NO_ENTRIES 1
#include "variant_zcomm_dec.cu"

extern "C" __global__ void __launch_bounds__(64) comm_dec8_x16(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_body<true, 0, false, 8>(meta, q); }
extern "C" __global__ void __launch_bounds__(64) comm_dec8_f32(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_body<false, 0, false, 8>(meta, q); }
#define COMM_DEC8_ENTRY(NS)                                                                                                                                         \
    extern "C" __global__ void __launch_bounds__(64) comm_dec8_attn_##NS(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_body<true, NS, false, 8>(meta, q); } \
    extern "C" __global__ void __launch_bounds__(64) comm_dec8_ffn_##NS(const Meta* meta, const __grid_constant__ NvlsParams q) { comm_dec_body<false, NS, true, 8>(meta, q); }
COMM_DEC8_ENTRY(1)
COMM_DEC8_ENTRY(2)
COMM_DEC8_ENTRY(3)
COMM_DEC8_ENTRY(4)
COMM_DEC8_ENTRY(5)
COMM_DEC8_ENTRY(6)
COMM_DEC8_ENTRY(7)
COMM_DEC8_ENTRY(8)

extern "C" __global__ void __launch_bounds__(64, COMM_MIN_CTAS_PER_SM) comm_push2_k1_tp8(const Meta* meta, const __grid_constant__ Comm2Params p) { comm_push2_body<8, 1>(meta, p); }
