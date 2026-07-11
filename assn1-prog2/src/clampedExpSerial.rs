//! Clamped exponentiation: `output[i] = min(values[i] ^ exponents[i], CLAMP_MAX)`.
//!
//! Ported from the CS149 assignment 1 C++ starter code. Two implementations
//! are provided against the same signature so they can be benchmarked
//! against each other:
//! - [`clamped_exp_serial`]: straightforward scalar loop.
//! - [`clamped_exp_simd`]: intended to use explicit SIMD; currently falls
//!   back to the serial path (see TODO below).

/// Upper bound that a computed result is clamped to.
const CLAMP_MAX: f32 = 9.999999;

/// Computes `values[i] ^ exponents[i]` for every element, clamping each
/// result to `CLAMP_MAX`, and writes the results into `output`.
///
/// # Panics
///
/// Panics if `values`, `exponents`, and `output` do not all have the same
/// length.
pub fn clamped_exp_serial(values: &[f32], exponents: &[i32], output: &mut [f32]) {
    assert_eq!(values.len(), exponents.len());
    assert_eq!(values.len(), output.len());

    for i in 0..values.len() {
        output[i] = clamped_power(values[i], exponents[i]);
    }
}

/// Computes `base ^ exponent`, clamped to [`CLAMP_MAX`].
///
/// # Panics
///
/// Panics if `exponent` is negative.
fn clamped_power(base: f32, exponent: i32) -> f32 {
    assert!(exponent >= 0);

    if exponent == 0 {
        return 1.0;
    }

    let mut result = base;
    for _ in 1..exponent {
        result *= base;
    }

    result.min(CLAMP_MAX)
}

/// SIMD counterpart to [`clamped_exp_serial`]. Same signature and same
/// per-element result, so the two can be swapped and benchmarked directly.
///
/// # Panics
///
/// Panics if `values`, `exponents`, and `output` do not all have the same
/// length (delegated to [`clamped_exp_serial`] for now).
///
/// # TODO
///
/// - Replace the fallback below with an explicit SIMD implementation
///   (`std::simd` or `std::arch` intrinsics), operating on chunks of the
///   input in lockstep.
/// - Handle the remainder (`values.len() % LANES != 0`) with a scalar tail
///   loop, reusing [`clamped_power`].
/// - Benchmark against [`clamped_exp_serial`] and record the speedup in the
///   crate README.
pub fn clamped_exp_simd(values: &[f32], exponents: &[i32], output: &mut [f32]) {
    clamped_exp_serial(values, exponents, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_exponent_yields_one() {
        assert_eq!(clamped_power(5.0, 0), 1.0);
    }

    #[test]
    fn exponent_of_one_yields_base() {
        assert_eq!(clamped_power(7.0, 1), 7.0);
    }

    #[test]
    fn positive_exponent_multiplies() {
        assert_eq!(clamped_power(2.0, 3), 8.0);
    }

    #[test]
    fn base_of_zero_yields_zero() {
        assert_eq!(clamped_power(0.0, 4), 0.0);
    }

    #[test]
    fn negative_base_with_odd_exponent_stays_negative() {
        assert_eq!(clamped_power(-2.0, 3), -8.0);
    }

    #[test]
    fn negative_base_with_even_exponent_is_positive() {
        assert_eq!(clamped_power(-2.0, 2), 4.0);
    }

    #[test]
    fn result_is_clamped() {
        assert_eq!(clamped_power(10.0, 2), CLAMP_MAX);
    }

    #[test]
    fn result_at_clamp_boundary_is_unchanged() {
        assert_eq!(clamped_power(CLAMP_MAX, 1), CLAMP_MAX);
    }

    #[test]
    fn negative_result_is_not_clamped() {
        assert_eq!(clamped_power(-100.0, 3), -1_000_000.0);
    }

    #[test]
    #[should_panic]
    fn negative_exponent_panics() {
        clamped_power(2.0, -1);
    }

    #[test]
    fn arrays_are_processed_elementwise() {
        let values = [2.0, 3.0, 0.0];
        let exponents = [3, 0, 5];
        let mut output = [0.0; 3];

        clamped_exp_serial(&values, &exponents, &mut output);

        assert_eq!(output, [8.0, 1.0, 0.0]);
    }

    #[test]
    fn empty_slices_are_a_no_op() {
        let values: [f32; 0] = [];
        let exponents: [i32; 0] = [];
        let mut output: [f32; 0] = [];

        clamped_exp_serial(&values, &exponents, &mut output);

        assert_eq!(output, []);
    }

    #[test]
    #[should_panic]
    fn mismatched_exponents_length_panics() {
        let values = [1.0, 2.0];
        let exponents = [1];
        let mut output = [0.0; 2];

        clamped_exp_serial(&values, &exponents, &mut output);
    }

    #[test]
    #[should_panic]
    fn mismatched_output_length_panics() {
        let values = [1.0, 2.0];
        let exponents = [1, 1];
        let mut output = [0.0; 1];

        clamped_exp_serial(&values, &exponents, &mut output);
    }

    #[test]
    fn simd_matches_serial() {
        let values = [2.0, 3.0, 10.0, 0.0];
        let exponents = [3, 0, 2, 5];

        let mut serial_output = [0.0; 4];
        clamped_exp_serial(&values, &exponents, &mut serial_output);

        let mut simd_output = [0.0; 4];
        clamped_exp_simd(&values, &exponents, &mut simd_output);

        assert_eq!(serial_output, simd_output);
    }
}
