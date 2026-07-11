//! Clamped exponentiation: `output[i] = min(values[i] ^ exponents[i], CLAMP_MAX)`.
//!
//! Ported from the CS149 assignment 1, program 2 (`prog2_vecintrin`) C++
//! starter code. That assignment asks for a vectorized implementation
//! built on "fake vector intrinsics" with a configurable `VECTOR_WIDTH`,
//! graded by two metrics rather than wall-clock time: total vector
//! instructions issued, and vector utilization (the fraction of lanes
//! that were active across those instructions). [`clamped_exp_vector`]
//! models that same lane/mask/instruction-count behavior in safe Rust.
//!
//! - [`clamped_exp_serial`]: the scalar reference ("gold") implementation.
//! - [`clamped_exp_vector`]: vectorized, correct for any combination of
//!   input length and `vector_width` (including widths that don't evenly
//!   divide the length).

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

/// Instrumentation mirroring CS149's fake vector-intrinsics statistics:
/// every simulated vector instruction counts once no matter how many
/// lanes are active, and utilization is the fraction of (instruction,
/// lane) pairs that were actually active.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct VectorStats {
    pub vector_width: usize,
    pub instructions: usize,
    active_lanes: usize,
}

impl VectorStats {
    fn record(&mut self, active_lane_count: usize) {
        self.instructions += 1;
        self.active_lanes += active_lane_count;
    }

    /// Fraction of lanes that were active across all instructions, in `[0, 1]`.
    pub fn utilization(&self) -> f64 {
        if self.instructions == 0 {
            return 0.0;
        }
        self.active_lanes as f64 / (self.instructions * self.vector_width) as f64
    }
}

/// Vectorized version of [`clamped_exp_serial`], processing `values` in
/// `vector_width`-wide chunks. Within a chunk, every lane advances in
/// lockstep behind a mask, modeling real SIMD hardware: a vector unit
/// issues one instruction stream for all lanes, so a lane that finishes
/// early still burns instructions masked-off while its chunk-mates catch
/// up. The final chunk is handled by masking off out-of-bounds lanes
/// rather than a separate scalar tail, so any `vector_width` works with
/// any input length.
///
/// Returns [`VectorStats`] recording the instructions issued and the
/// resulting lane utilization.
///
/// # Panics
///
/// Panics if `values`, `exponents`, and `output` do not all have the same
/// length, if any exponent is negative, or if `vector_width` is zero.
pub fn clamped_exp_vector(
    values: &[f32],
    exponents: &[i32],
    output: &mut [f32],
    vector_width: usize,
) -> VectorStats {
    assert_eq!(values.len(), exponents.len());
    assert_eq!(values.len(), output.len());
    assert!(exponents.iter().all(|&exponent| exponent >= 0));
    assert!(vector_width > 0);

    let mut stats = VectorStats {
        vector_width,
        ..Default::default()
    };

    let mut chunk_start = 0;
    while chunk_start < values.len() {
        let in_bounds_lanes = vector_width.min(values.len() - chunk_start);
        process_chunk(
            &values[chunk_start..chunk_start + in_bounds_lanes],
            &exponents[chunk_start..chunk_start + in_bounds_lanes],
            &mut output[chunk_start..chunk_start + in_bounds_lanes],
            vector_width,
            &mut stats,
        );
        chunk_start += vector_width;
    }

    stats
}

/// Runs one `vector_width`-wide lockstep chunk. `base`/`exponent`/`out`
/// may be shorter than `vector_width` on the final chunk; the unused
/// lanes are simply never marked active, matching how the real
/// intrinsics library uses a bounds mask (`_cs149_init_ones`) for a
/// short final chunk.
fn process_chunk(
    base: &[f32],
    exponent: &[i32],
    out: &mut [f32],
    vector_width: usize,
    stats: &mut VectorStats,
) {
    let in_bounds_lanes = base.len();

    let mut result = vec![1.0f32; vector_width];
    let mut remaining = vec![0i32; vector_width];
    remaining[..in_bounds_lanes].copy_from_slice(exponent);

    let mut mask: Vec<bool> = (0..vector_width)
        .map(|lane| lane < in_bounds_lanes && remaining[lane] > 0)
        .collect();
    stats.record(mask.iter().filter(|&&active| active).count());

    while mask.iter().any(|&active| active) {
        for lane in 0..vector_width {
            if mask[lane] {
                result[lane] *= base[lane];
            }
        }
        stats.record(mask.iter().filter(|&&active| active).count());

        for lane in 0..vector_width {
            if mask[lane] {
                remaining[lane] -= 1;
            }
        }
        stats.record(mask.iter().filter(|&&active| active).count());

        for lane in 0..vector_width {
            mask[lane] = mask[lane] && remaining[lane] > 0;
        }
        stats.record(mask.iter().filter(|&&active| active).count());
    }

    for lane in 0..in_bounds_lanes {
        out[lane] = result[lane].min(CLAMP_MAX);
    }
    stats.record(in_bounds_lanes);
}

/// Real hardware SIMD, as opposed to [`clamped_exp_vector`]'s simulation:
/// dispatches to an AVX2 implementation (8-wide `f32` lanes, matching the
/// myth machines' actual vector width) when the CPU supports it at run
/// time, falling back to [`clamped_exp_serial`] otherwise (e.g. non-x86_64
/// targets, or x86_64 CPUs predating AVX2).
///
/// # Panics
///
/// Panics if `values`, `exponents`, and `output` do not all have the same
/// length, or if any exponent is negative.
pub fn clamped_exp_simd(values: &[f32], exponents: &[i32], output: &mut [f32]) {
    assert_eq!(values.len(), exponents.len());
    assert_eq!(values.len(), output.len());
    assert!(exponents.iter().all(|&exponent| exponent >= 0));

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime `is_x86_feature_detected!` check above.
            unsafe { avx2::clamped_exp_avx2(values, exponents, output) };
            return;
        }
    }

    clamped_exp_serial(values, exponents, output);
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{CLAMP_MAX, clamped_power};
    use std::arch::x86_64::*;

    /// Lanes processed per AVX2 instruction (256 bits / 32-bit `f32`).
    const LANES: usize = 8;

    /// Real AVX2 vectorization of [`clamped_power`][super::clamped_power]:
    /// every lane multiplies in lockstep behind a compare-generated mask.
    /// The trip count for a chunk is the *largest* exponent among its
    /// lanes, computed once up front in scalar code; the vector loop then
    /// runs that many times unconditionally. (An earlier version instead
    /// re-checked "is any lane still active" via `_mm256_movemask_epi8`
    /// every iteration — that data-dependent branch cost more than the
    /// masking it was meant to skip, and measured slower than the plain
    /// scalar loop. Precomputing the trip count avoids the branch
    /// entirely.) Any leftover elements (`values.len() % LANES != 0`)
    /// fall back to the scalar [`clamped_power`] for the tail.
    ///
    /// # Safety
    ///
    /// Caller must ensure the AVX2 target feature is available (checked
    /// via `is_x86_feature_detected!("avx2")` by [`super::clamped_exp_simd`]),
    /// and that `values`, `exponents`, and `output` all have equal length.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn clamped_exp_avx2(values: &[f32], exponents: &[i32], output: &mut [f32]) {
        // SAFETY: all raw pointer/intrinsic calls stay within the bounds
        // established by the `chunk_start + LANES <= values.len()` guard,
        // and AVX2 availability is the caller's contract per this fn's
        // own safety section.
        unsafe {
            let clamp_max = _mm256_set1_ps(CLAMP_MAX);
            let ones_i32 = _mm256_set1_epi32(1);
            let zero_i32 = _mm256_setzero_si256();

            let mut chunk_start = 0;
            while chunk_start + LANES <= values.len() {
                let chunk_exponents = &exponents[chunk_start..chunk_start + LANES];
                let max_exponent = chunk_exponents.iter().copied().max().unwrap_or(0);

                let base = _mm256_loadu_ps(values.as_ptr().add(chunk_start));
                let mut remaining = _mm256_loadu_si256(chunk_exponents.as_ptr().cast::<__m256i>());
                let mut result = _mm256_set1_ps(1.0);

                for _ in 0..max_exponent {
                    let mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(remaining, zero_i32));
                    result = _mm256_blendv_ps(result, _mm256_mul_ps(result, base), mask);
                    remaining = _mm256_sub_epi32(remaining, ones_i32);
                }

                result = _mm256_min_ps(result, clamp_max);
                _mm256_storeu_ps(output.as_mut_ptr().add(chunk_start), result);

                chunk_start += LANES;
            }

            for i in chunk_start..values.len() {
                output[i] = clamped_power(values[i], exponents[i]);
            }
        }
    }
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

    fn assert_vector_matches_serial(values: &[f32], exponents: &[i32], vector_width: usize) {
        let mut serial_output = vec![0.0; values.len()];
        clamped_exp_serial(values, exponents, &mut serial_output);

        let mut vector_output = vec![0.0; values.len()];
        clamped_exp_vector(values, exponents, &mut vector_output, vector_width);

        assert_eq!(serial_output, vector_output, "vector_width={vector_width}");
    }

    #[test]
    fn vector_matches_serial_for_exact_multiple_of_width() {
        let values = [2.0, 3.0, 10.0, 0.0];
        let exponents = [3, 0, 2, 5];
        assert_vector_matches_serial(&values, &exponents, 4);
    }

    #[test]
    fn vector_matches_serial_with_ragged_tail() {
        // N=6 is not a multiple of a width-4 vector, exercising the
        // bounds mask on the final short chunk (mirrors `-s 3` in the
        // original assignment's correctness check).
        let values = [2.0, 3.0, 10.0, 0.0, 4.0, 1.5];
        let exponents = [3, 0, 2, 5, 1, 4];
        assert_vector_matches_serial(&values, &exponents, 4);
    }

    #[test]
    fn vector_matches_serial_across_widths() {
        let values = [2.0, 3.0, 10.0, 0.0, 4.0, 1.5, 7.0];
        let exponents = [3, 0, 2, 5, 1, 4, 2];

        for vector_width in [1, 2, 3, 4, 5, 8, 16] {
            assert_vector_matches_serial(&values, &exponents, vector_width);
        }
    }

    #[test]
    fn vector_matches_serial_on_empty_input() {
        assert_vector_matches_serial(&[], &[], 4);
    }

    #[test]
    #[should_panic]
    fn vector_negative_exponent_panics() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let exponents = [1, -1, 1, 1];
        let mut output = [0.0; 4];

        clamped_exp_vector(&values, &exponents, &mut output, 4);
    }

    #[test]
    #[should_panic]
    fn vector_width_of_zero_panics() {
        let values = [1.0];
        let exponents = [1];
        let mut output = [0.0];

        clamped_exp_vector(&values, &exponents, &mut output, 0);
    }

    #[test]
    fn wider_vectors_can_lower_utilization_under_divergent_exponents() {
        // Exponents deliberately diverge more as more lanes are grouped
        // together: every lane but one finishes in 1 iteration, and the
        // last lane needs `vector_width` iterations, so wider vectors
        // waste more masked-off instructions per useful op.
        let values = vec![2.0; 16];
        let mut narrow_exponents = vec![1; 4];
        narrow_exponents[3] = 4;
        let mut wide_exponents = vec![1; 16];
        wide_exponents[15] = 16;

        let mut narrow_output = vec![0.0; 4];
        let narrow_stats =
            clamped_exp_vector(&values[..4], &narrow_exponents, &mut narrow_output, 4);

        let mut wide_output = vec![0.0; 16];
        let wide_stats = clamped_exp_vector(&values, &wide_exponents, &mut wide_output, 16);

        assert!(wide_stats.utilization() < narrow_stats.utilization());
    }

    fn assert_simd_matches_serial(values: &[f32], exponents: &[i32]) {
        let mut serial_output = vec![0.0; values.len()];
        clamped_exp_serial(values, exponents, &mut serial_output);

        let mut simd_output = vec![0.0; values.len()];
        clamped_exp_simd(values, exponents, &mut simd_output);

        assert_eq!(serial_output, simd_output);
    }

    #[test]
    fn simd_matches_serial_for_exact_multiple_of_lanes() {
        let values = [2.0, 3.0, 10.0, 0.0, 1.5, 4.0, 2.5, 0.5];
        let exponents = [3, 0, 2, 5, 1, 4, 0, 6];
        assert_simd_matches_serial(&values, &exponents);
    }

    #[test]
    fn simd_matches_serial_with_scalar_tail() {
        // 10 elements is not a multiple of the AVX2 8-lane width, so this
        // exercises the scalar fallback for the last 2 elements.
        let values = [2.0, 3.0, 10.0, 0.0, 1.5, 4.0, 2.5, 0.5, 6.0, 1.0];
        let exponents = [3, 0, 2, 5, 1, 4, 0, 6, 2, 3];
        assert_simd_matches_serial(&values, &exponents);
    }

    #[test]
    fn simd_matches_serial_on_empty_input() {
        assert_simd_matches_serial(&[], &[]);
    }

    #[test]
    #[should_panic]
    fn simd_negative_exponent_panics() {
        let values = [1.0; 8];
        let mut exponents = [1; 8];
        exponents[3] = -1;
        let mut output = [0.0; 8];

        clamped_exp_simd(&values, &exponents, &mut output);
    }
}
