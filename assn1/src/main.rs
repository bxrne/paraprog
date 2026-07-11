mod mandelbrot;
mod ppm;

fn main() {
    let width = 800;
    let height = 600;
    let max_iter = 1000;
    let fname = "a1-mandelbrot.ppm";

    let pixels = mandelbrot::render(width, height, max_iter);

    ppm::write(fname, width, height, &pixels).expect("Failed to write PPM file");
    println!("Mandelbrot image saved to {}", fname);
}
