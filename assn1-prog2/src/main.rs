mod clamped_exp;

use clamped_exp::{clamped_exp_serial, clamped_exp_simd, clamped_exp_vector};
use std::time::Instant;
use tracing::info;

/// Matches `./myexp -s 10000` from the original CS149 assignment.
const N: usize = 10_000;

/// Widths swept in the original assignment's `#define VECTOR_WIDTH` study.
const VECTOR_WIDTHS: [usize; 4] = [2, 4, 8, 16];

/// Element count for the real-hardware SIMD benchmark. Large enough, and
/// with a uniform, compute-heavy exponent (see below), that AVX2's 8-wide
/// parallel multiplies actually dominate over memory traffic and masking
/// overhead.
const BENCH_N: usize = 10_000_000;

/// Uniform exponent for the benchmark. Real per-lane divergence (as used
/// in the vector-utilization sweep above) caps SIMD gains at the least-
/// converged lane; a uniform exponent removes that variable so the
/// benchmark isolates raw compute throughput.
const BENCH_EXPONENT: i32 = 20;

fn main() {
    tracing_subscriber::fmt::init();

    run_vector_utilization_sweep();
    run_simd_benchmark();
}

fn run_vector_utilization_sweep() {
    let values: Vec<f32> = (0..N).map(|i| 1.0 + (i % 5) as f32 * 0.5).collect();
    let exponents: Vec<i32> = (0..N).map(|i| (i % 4) as i32).collect();

    let mut gold = vec![0.0; N];
    clamped_exp_serial(&values, &exponents, &mut gold);

    for vector_width in VECTOR_WIDTHS {
        let mut output = vec![0.0; N];
        let stats = clamped_exp_vector(&values, &exponents, &mut output, vector_width);
        assert_eq!(
            output, gold,
            "vector_width={vector_width} diverged from gold"
        );

        info!(
            vector_width,
            total_vector_instructions = stats.instructions,
            vector_utilization = format!("{:.1}%", stats.utilization() * 100.0),
            "vectorized run complete"
        );
    }
}

fn run_simd_benchmark() {
    let values: Vec<f32> = (0..BENCH_N).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
    let exponents = vec![BENCH_EXPONENT; BENCH_N];

    let mut serial_output = vec![0.0; BENCH_N];
    let start = Instant::now();
    clamped_exp_serial(&values, &exponents, &mut serial_output);
    let serial_elapsed = start.elapsed();
    info!(?serial_elapsed, "serial run complete");

    let mut simd_output = vec![0.0; BENCH_N];
    let start = Instant::now();
    clamped_exp_simd(&values, &exponents, &mut simd_output);
    let simd_elapsed = start.elapsed();
    assert_eq!(serial_output, simd_output);

    info!(
        ?simd_elapsed,
        speedup = serial_elapsed.as_secs_f64() / simd_elapsed.as_secs_f64(),
        "avx2 simd run complete"
    );
}
