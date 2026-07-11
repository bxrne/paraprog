#[allow(non_snake_case)]
mod clampedExpSerial;

use clampedExpSerial::clamped_exp_serial;

fn main() {
    let values = [1.5, 2.0, 10.0, 3.0];
    let exponents = [2, 0, 3, 4];
    let mut output = [0.0; 4];

    clamped_exp_serial(&values, &exponents, &mut output);

    println!("{:?}", output);
}
