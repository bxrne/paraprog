use std::thread;
use std::time::{Duration, Instant};
use tracing::debug;

/// Compute the number of iterations before the point `(c_re, c_im)` escapes,
/// capped at `max_iter`. Points that never escape return `max_iter`.
pub fn escape_iterations(c_re: f64, c_im: f64, max_iter: u32) -> u32 {
    let mut z_re = 0.0;
    let mut z_im = 0.0;
    let mut iter = 0;

    while z_re * z_re + z_im * z_im <= 4.0 && iter < max_iter {
        let new_re = z_re * z_re - z_im * z_im + c_re;
        let new_im = 2.0 * z_re * z_im + c_im;
        z_re = new_re;
        z_im = new_im;
        iter += 1;
    }

    iter
}

/// Map an escape-iteration count to an RGB color. Points inside the set
/// (those reaching `max_iter`) are colored black.
pub fn color(iter: u32, max_iter: u32) -> [u8; 3] {
    if iter < max_iter {
        let t = iter as f64 / max_iter as f64;
        let r = (9.0 * (1.0 - t) * t * 255.0) as u8;
        let g = (15.0 * (1.0 - t) * t * 255.0) as u8;
        let b = (8.5 * (1.0 - t) * t * 255.0) as u8;
        [r, g, b]
    } else {
        [0, 0, 0]
    }
}

/// Render the Mandelbrot set into a flat RGB pixel buffer (`width * height * 3` bytes).
pub fn render(width: usize, height: usize, max_iter: u32) -> Vec<u8> {
    assert!(width > 0, "width must be positive");
    assert!(height > 0, "height must be positive");
    assert!(max_iter > 0, "max_iter must be positive");

    let mut pixels = vec![0u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let c_re = (x as f64 / width as f64) * 3.0 - 2.0; // Scale to [-2, 1]
            let c_im = (y as f64 / height as f64) * 2.0 - 1.0; // Scale to [-1, 1]
            let iter = escape_iterations(c_re, c_im, max_iter);
            let rgb = color(iter, max_iter);
            let idx = (y * width + x) * 3;
            pixels[idx] = rgb[0];
            pixels[idx + 1] = rgb[1];
            pixels[idx + 2] = rgb[2];
        }
    }

    pixels
}

/// Render the Mandelbrot set into a flat RGB pixel buffer (`width * height * 3` bytes),
/// splitting the work across `num_threads` threads (capped at `height`, since a
/// thread with no rows to render would do nothing).
///
/// Panics if `width`, `height`, `max_iter`, or `num_threads` is zero, or if a
/// worker thread panics while rendering.
///
/// Emits `tracing` events per worker (rows handled, pure compute time) plus a
/// scope-level event comparing wall time against the slowest worker, so the
/// gap between the two (thread spawn/join/scheduling "IO") is visible.
pub fn render_parallel(width: usize, height: usize, max_iter: u32, num_threads: usize) -> Vec<u8> {
    assert!(width > 0, "width must be positive");
    assert!(height > 0, "height must be positive");
    assert!(max_iter > 0, "max_iter must be positive");
    assert!(num_threads > 0, "num_threads must be positive");

    let alloc_start = Instant::now();
    let mut pixels = vec![0u8; width * height * 3];
    let alloc_elapsed = alloc_start.elapsed();

    let num_threads = num_threads.min(height);
    let rows_per_thread = height / num_threads;

    let scope_start = Instant::now();
    let worker_elapsed: Vec<_> = thread::scope(|scope| {
        let mut remaining = &mut pixels[..];
        let mut start_row = 0;
        let mut handles = Vec::with_capacity(num_threads);

        for thread_id in 0..num_threads {
            let end_row = if thread_id == num_threads - 1 {
                height
            } else {
                (thread_id + 1) * rows_per_thread
            };

            let rows_in_chunk = end_row - start_row;
            let (chunk, rest) = remaining.split_at_mut(rows_in_chunk * width * 3);
            remaining = rest;

            let handle = scope.spawn(move || {
                let compute_start = Instant::now();
                for y in start_row..end_row {
                    for x in 0..width {
                        let c_re = (x as f64 / width as f64) * 3.0 - 2.0; // Scale to [-2, 1]
                        let c_im = (y as f64 / height as f64) * 2.0 - 1.0; // Scale to [-1, 1]
                        let iter = escape_iterations(c_re, c_im, max_iter);
                        let rgb = color(iter, max_iter);
                        let idx = (y - start_row) * width * 3 + x * 3;
                        chunk[idx] = rgb[0];
                        chunk[idx + 1] = rgb[1];
                        chunk[idx + 2] = rgb[2];
                    }
                }
                let compute_elapsed = compute_start.elapsed();
                debug!(
                    thread_id,
                    rows = rows_in_chunk,
                    ?compute_elapsed,
                    "worker finished"
                );
                compute_elapsed
            });
            handles.push(handle);

            start_row = end_row;
        }

        handles
            .into_iter()
            .map(|h| h.join().expect("worker thread panicked while rendering"))
            .collect()
    });
    let scope_wall = scope_start.elapsed();

    let slowest = worker_elapsed.iter().max().copied().unwrap_or_default();
    let fastest = worker_elapsed.iter().min().copied().unwrap_or_default();
    let mean = worker_elapsed.iter().sum::<Duration>() / worker_elapsed.len() as u32;
    let overhead = scope_wall.saturating_sub(slowest);
    debug!(
        num_threads,
        ?alloc_elapsed,
        ?scope_wall,
        slowest_worker = ?slowest,
        fastest_worker = ?fastest,
        mean_worker = ?mean,
        skew = ?slowest.saturating_sub(fastest),
        spawn_join_overhead = ?overhead,
        "parallel scope complete"
    );

    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_iterations_inside() {
        let max_iter = 1000;
        // Points known to be inside the Mandelbrot set
        let inside_points = [(0.0, 0.0), (-0.5, 0.0), (-0.7, 0.1)];

        for &(c_re, c_im) in &inside_points {
            let iter = escape_iterations(c_re, c_im, max_iter);
            assert_eq!(
                iter, max_iter,
                "Point ({}, {}) should be inside the set",
                c_re, c_im
            );
        }
    }
    #[test]
    fn test_escape_iterations_outside() {
        let max_iter = 1000;
        // Points known to be outside the Mandelbrot set
        // These points should escape quickly
        let outside_points = [(2.0, 2.0), (-2.0, -1.5), (0.5, 0.5)];

        for &(c_re, c_im) in &outside_points {
            let iter = escape_iterations(c_re, c_im, max_iter);
            assert!(
                iter < max_iter,
                "Point ({}, {}) should be outside the set",
                c_re,
                c_im
            );
        }
    }

    #[test]
    fn test_color_mapping() {
        let max_iter = 1000;
        // Test color mapping for various iteration counts
        let test_cases = [
            (0, [0, 0, 0]),                                // Inside the set
            (max_iter / 2, color(max_iter / 2, max_iter)), // Mid-range color
            (max_iter - 1, color(max_iter - 1, max_iter)), // Just before max
            (max_iter, [0, 0, 0]),                         // Inside the set
            (max_iter + 1, [0, 0, 0]),                     // Outside the set
        ];

        for &(iter, expected_color) in &test_cases {
            let color_result = color(iter, max_iter);
            assert_eq!(
                color_result, expected_color,
                "Color mapping failed for iter {}",
                iter
            );
        }
    }

    #[test]
    fn test_render_parallel_matches_render() {
        let (width, height, max_iter) = (64, 48, 200);
        let serial = render(width, height, max_iter);
        for num_threads in [1, 2, 3, 4, 5] {
            assert_eq!(
                serial,
                render_parallel(width, height, max_iter, num_threads),
                "mismatch with num_threads = {}",
                num_threads
            );
        }
    }
}
