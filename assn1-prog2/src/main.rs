#[allow(non_snake_case)]
mod clampedExpSerial;

use clampedExpSerial::{clamped_exp_serial, clamped_exp_simd};
use std::time::Instant;
use tracing::info;

const N: usize = 4;

fn main() {
    tracing_subscriber::fmt::init();

    let values = [1.5, 2.0, 10.0, 3.0];
    let exponents = [2, 0, 3, 4];

    let mut serial_output = [0.0; N];
    let start = Instant::now();
    clamped_exp_serial(&values, &exponents, &mut serial_output);
    let serial_elapsed = start.elapsed();
    info!(?serial_elapsed, output = ?serial_output, "serial run complete");

    let mut simd_output = [0.0; N];
    let start = Instant::now();
    clamped_exp_simd(&values, &exponents, &mut simd_output);
    let simd_elapsed = start.elapsed();
    info!(
        ?simd_elapsed,
        output = ?simd_output,
        speedup = serial_elapsed.as_secs_f64() / simd_elapsed.as_secs_f64(),
        "simd run complete"
    );

    println!("{:?}", simd_output);
}
