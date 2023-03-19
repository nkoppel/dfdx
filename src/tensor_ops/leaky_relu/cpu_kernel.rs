use crate::tensor_ops::cpu_kernels::UnaryDerivative;
use num_traits::Float;

impl<F: Float> UnaryDerivative<F> for super::LeakyReLUKernelOp<F> {
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        if x < F::zero() {
            x * self.alpha
        } else {
            x
        }
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        if *x < F::zero() {
            self.alpha
        } else {
            F::one()
        }
    }
}
