use super::{PReLUKernelOp as Binary, LeakyReLUKernelOp as Scalar};
use crate::tensor_ops::cuda_kernels::{cuda_binary, cuda_unary};

unsafe impl cudarc::driver::DeviceRepr for Scalar<f32> {}
unsafe impl cudarc::driver::DeviceRepr for Scalar<f64> {}
unsafe impl cudarc::driver::DeviceRepr for Binary {}

const SCALAR_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/leaky_relu.ptx"));
const BINARY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/prelu.ptx"));

cuda_unary!(Scalar<f32>, f32, SCALAR_PTX, "leaky_relu_fwd_f32", "leaky_relu_bwd_f32");
cuda_unary!(Scalar<f64>, f64, SCALAR_PTX, "leaky_relu_fwd_f64", "leaky_relu_bwd_f64");
cuda_binary!(
    Binary,
    f32,
    BINARY_PTX,
    "prelu_fwd_f32",
    "prelu_bwd_lhs_f32",
    "prelu_bwd_rhs_f32"
);
cuda_binary!(
    Binary,
    f64,
    BINARY_PTX,
    "prelu_fwd_f64",
    "prelu_bwd_lhs_f64",
    "prelu_bwd_rhs_f64"
);
