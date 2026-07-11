mod mandelbrot;
mod ppm;

use std::io;
use std::time::{Duration, Instant};
use tracing::info;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;
const MAX_ITER: u32 = 1000;
const OUTPUT_PATH: &str = "a1-mandelbrot.ppm";

fn main() -> io::Result<()> {
    tracing_subscriber::fmt::init();

    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    info!(max_threads, "detected available parallelism");

    let start = Instant::now();
    mandelbrot::render(WIDTH, HEIGHT, MAX_ITER);
    let serial_elapsed = start.elapsed();
    info!(?serial_elapsed, "serial render complete");

    let (best_threads, best_elapsed) = sweep_thread_counts(max_threads, serial_elapsed)?;
    info!(
        best_threads,
        ?best_elapsed,
        best_speedup = serial_elapsed.as_secs_f64() / best_elapsed.as_secs_f64(),
        "fastest thread count"
    );

    println!("Mandelbrot image saved to {}", OUTPUT_PATH);
    Ok(())
}

/// Render with thread counts `1..=max_threads`, logging timing for each, and
/// write the image using the last (highest) thread count. Returns the
/// fastest thread count observed and its elapsed render time.
fn sweep_thread_counts(
    max_threads: usize,
    serial_elapsed: Duration,
) -> io::Result<(usize, Duration)> {
    let mut best_threads = 1;
    let mut best_elapsed = Duration::MAX;

    for num_threads in 1..=max_threads {
        let start = Instant::now();
        let pixels = mandelbrot::render_parallel(WIDTH, HEIGHT, MAX_ITER, num_threads);
        let parallel_elapsed = start.elapsed();

        info!(
            num_threads,
            ?parallel_elapsed,
            speedup = serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64(),
            "parallel render complete"
        );

        if parallel_elapsed < best_elapsed {
            best_threads = num_threads;
            best_elapsed = parallel_elapsed;
        }

        if num_threads == max_threads {
            let write_start = Instant::now();
            ppm::write(OUTPUT_PATH, WIDTH, HEIGHT, &pixels)?;
            info!(write_elapsed = ?write_start.elapsed(), "wrote PPM file (disk IO)");
        }
    }

    Ok((best_threads, best_elapsed))
}
