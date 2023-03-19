use super::PReLUKernelOp as Binary;
use crate::tensor_ops::cuda_kernels::cuda_binary;

unsafe impl cudarc::driver::DeviceRepr for Binary {}

const BINARY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/prelu.ptx"));

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
