mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::*;
use crate::{
    shapes::*,
    tensor::{HasErr, Merge, Tape, Tensor},
};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct PReLUKernelOp;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLUKernelOp<E> {
    alpha: E,
}

pub fn leaky_relu<S: Shape, E: Dtype, D: UnaryKernel<LeakyReLUKernelOp<E>, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    alpha: E,
) -> Tensor<S, E, D, T> {
    t.leaky_relu(alpha)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<LeakyReLUKernelOp<E>, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [relu]
    pub fn leaky_relu(self, alpha: E) -> Self {
        self.try_leaky_relu(alpha).unwrap()
    }
    /// See [relu]
    pub fn try_leaky_relu(self, alpha: E) -> Result<Self, D::Err> {
        try_unary_op(LeakyReLUKernelOp { alpha }, self)
    }
}

pub fn prelu<S: Shape, E: Dtype, D: BinaryKernel<PReLUKernelOp, E>, LTape, RTape>(
    t: Tensor<S, E, D, LTape>,
    alpha: Tensor<S, E, D, RTape>,
) -> Tensor<S, E, D, LTape>
where
    LTape: Tape<E, D> + Merge<RTape>,
    RTape: Tape<E, D>,
{
    t.prelu(alpha)
}

impl<S: Shape, E: Dtype, D: BinaryKernel<PReLUKernelOp, E>, LTape: Tape<E, D>>
    Tensor<S, E, D, LTape>
{
    pub fn prelu<RTape: Tape<E, D>>(self, alpha: Tensor<S, E, D, RTape>) -> Self
    where
        LTape: Merge<RTape>,
    {
        self.try_prelu(alpha).unwrap()
    }
    pub fn try_prelu<RTape>(self, prob: Tensor<S, E, D, RTape>) -> Result<Self, D::Err>
    where
        RTape: Tape<E, D>,
        LTape: Merge<RTape>,
    {
        try_binary_op(PReLUKernelOp, self, prob)
    }
}
#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_prelu_0d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor(-1.0);
        let alpha: Tensor<_, TestDtype, _> = dev.tensor(0.5);

        let r = x.leaky_trace().prelu(alpha.clone());
        assert_eq!(r.array(), -0.5);
        let g = r.backward();
        assert_eq!(g.get(&x).array(), 0.5);
        assert_eq!(g.get(&alpha).array(), -1.0);
    }

    #[test]
    fn test_prelu_1d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 1.0, 2.0]);
        let alpha: Tensor<_, TestDtype, _> = dev.tensor([0.25, 0.5, 0.0, 0.0]);

        let r = x.leaky_trace().prelu(alpha.clone());
        assert_eq!(r.array(), [-0.5, -0.5, 1.0, 2.0]);
        let g = r.sum().backward();
        assert_eq!(g.get(&x).array(), [0.25, 0.5, 1.0, 1.0]);
        assert_eq!(g.get(&alpha).array(), [-2.0, -1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_leaky_relu_0d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor(-1.0);
        let r = x.leaky_trace().leaky_relu(0.5);
        assert_eq!(r.array(), -0.5);
        let g = r.backward();
        assert_eq!(g.get(&x).array(), 0.5);
    }

    #[test]
    fn test_leaky_relu_1d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 1.0, 2.0]);
        let r = x.leaky_trace().leaky_relu(0.5);
        assert_eq!(r.array(), [-1.0, -0.5, 1.0, 2.0]);
        let g = r.sum().backward();
        assert_close(&g.get(&x).array(), &[0.5, 0.5, 1.0, 1.0]);
    }
}
