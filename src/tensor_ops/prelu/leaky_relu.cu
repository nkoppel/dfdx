#include "unary_op_macros.cuh"

template<typename F>
struct LeakyReLUKernelOp {
    F alpha;
};

UNARY_OP(float, leaky_relu_fwd_f32, leaky_relu_bwd_f32, LeakyReLUKernelOp<float>,
    x < 0.0 ? x * op.alpha : x,
    x < 0.0 ? op.alpha : 1.0);

UNARY_OP(double, leaky_relu_fwd_f64, leaky_relu_bwd_f64, LeakyReLUKernelOp<double>,
    x < 0.0 ? x * op.alpha : x,
    x < 0.0 ? op.alpha : 1.0);
