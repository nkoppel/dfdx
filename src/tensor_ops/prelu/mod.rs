mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::*;
use crate::{
    shapes::*,
    tensor::{Merge, Tape, Tensor},
};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct PReLUKernelOp;

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
}
