#include "binary_op_macros.cuh"

struct PReLUOp {};

BINARY_OP(float, prelu_fwd_f32, prelu_bwd_lhs_f32, prelu_bwd_rhs_f32, PReLUOp,
    x < 0.0 ? x * y : x,
    x < 0.0 ? y : 1.0,
    x < 0.0 ? x : 0.0)

BINARY_OP(double, prelu_fwd_f64, prelu_bwd_lhs_f64, prelu_bwd_rhs_f64, PReLUOp,
    x < 0.0 ? x * y : x,
    x < 0.0 ? y : 1.0,
    x < 0.0 ? x : 0.0)
