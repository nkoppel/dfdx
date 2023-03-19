use crate::tensor_ops::cpu_kernels::BinaryDerivative;
use num_traits::Float;

impl<F: Float> BinaryDerivative<F> for super::PReLUKernelOp {
    #[inline(always)]
    fn f(&self, &x: &F, &alpha: &F) -> F {
        if x < F::zero() {
            x * alpha
        } else {
            x
        }
    }
    #[inline(always)]
    fn dfdx(&self, x: &F, alpha: &F) -> F {
        if *x < F::zero() {
            *alpha
        } else {
            F::one()
        }
    }
    #[inline(always)]
    fn dfdy(&self, x: &F, _: &F) -> F {
        if *x < F::zero() {
            *x
        } else {
            F::zero()
        }
    }
}
