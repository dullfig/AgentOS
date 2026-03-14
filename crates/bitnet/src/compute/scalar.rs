//! Scalar (portable) compute backend.
//!
//! Wraps the existing I2S kernel — no SIMD, works on any platform.

use crate::tensor::TernaryTensor;
use crate::ops::matmul::ternary_matvec as scalar_ternary_matvec;
use super::ComputeBackend;

/// Portable scalar backend — the baseline.
#[derive(Debug, Clone, Copy)]
pub struct ScalarBackend;

impl ComputeBackend for ScalarBackend {
    fn name(&self) -> &str { "scalar" }

    fn ternary_matvec(&self, weights: &TernaryTensor, input: &[i8]) -> Vec<i32> {
        scalar_ternary_matvec(weights, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Ternary, TernaryTensor};

    fn weights_from_i8(values: &[i8], rows: usize, cols: usize) -> TernaryTensor {
        let ternary: Vec<Ternary> = values.iter().map(|&v| match v {
            -1 => Ternary::Neg,
             0 => Ternary::Zero,
             1 => Ternary::Pos,
             _ => panic!("not ternary"),
        }).collect();
        TernaryTensor::pack(&ternary, rows, cols)
    }

    #[test]
    fn scalar_matvec_basic() {
        let backend = ScalarBackend;
        let w = weights_from_i8(&[1, -1, 0, 1], 1, 4);
        let x = vec![10i8, 20, 30, 40];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![30]); // 10 - 20 + 0 + 40
    }
}
