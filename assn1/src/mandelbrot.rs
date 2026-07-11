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
}
