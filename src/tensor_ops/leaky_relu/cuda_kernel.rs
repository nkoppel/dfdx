use super::LeakyReLUKernelOp as Scalar;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for Scalar<f32> {}
unsafe impl cudarc::driver::DeviceRepr for Scalar<f64> {}

const SCALAR_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/leaky_relu.ptx"));

cuda_unary!(
    Scalar<f32>,
    f32,
    SCALAR_PTX,
    "leaky_relu_fwd_f32",
    "leaky_relu_bwd_f32"
);
cuda_unary!(
    Scalar<f64>,
    f64,
    SCALAR_PTX,
    "leaky_relu_fwd_f64",
    "leaky_relu_bwd_f64"
);
