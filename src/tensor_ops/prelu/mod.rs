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

pub trait PReLU<Rhs>: HasErr {
    fn try_prelu(self, rhs: Rhs) -> Result<Self, Self::Err>;

    fn prelu(self, rhs: Rhs) -> Self {
        self.try_prelu(rhs).unwrap()
    }
}

impl<S: Shape, E: Dtype, D, LhsTape: Tape<E, D>, R> PReLU<Tensor<S, E, D, R>>
    for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<PReLUKernelOp, E>,
    LhsTape: Merge<R>,
{
    fn try_prelu(self, rhs: Tensor<S, E, D, R>) -> Result<Self, Self::Err> {
        try_binary_op(PReLUKernelOp, self, rhs)
    }
}

pub trait LeakyRelu: HasErr + HasDtype {
    fn try_leaky_relu(self, rhs: Self::Dtype) -> Result<Self, Self::Err>;

    fn leaky_relu(self, rhs: Self::Dtype) -> Self {
        self.try_leaky_relu(rhs).unwrap()
    }
}

impl<S: Shape, E: Dtype, D: UnaryKernel<LeakyReLUKernelOp<E>, E>, T: Tape<E, D>> LeakyRelu
    for Tensor<S, E, D, T>
{
    fn try_leaky_relu(self, alpha: E) -> Result<Self, Self::Err> {
        try_unary_op(LeakyReLUKernelOp { alpha }, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
