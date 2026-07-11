ispc::ispc_module!(simple);

use std::time::Instant;
use tracing::info;

const N: usize = 1024;

fn simple_add_normal(a: &mut [f32], b: &[f32], out: &mut [f32]) {
    for i in 0..N {
        out[i] = a[i] + b[i];
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    let mut a = vec![1.0f32; N];
    let mut b = vec![2.0f32; N];
    let mut out = vec![0.0f32; N];

    let start = Instant::now();
    unsafe {
        simple::simple_add(a.as_mut_ptr(), b.as_mut_ptr(), out.as_mut_ptr(), N as i32);
    }
    let elapsed = start.elapsed();

    let gold: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(out, gold, "ispc simple_add diverged from scalar gold");

    info!(n = N, ?elapsed, "ispc simple_add complete");

    let start = Instant::now();
    simple_add_normal(&mut a, &b, &mut out);
    let elapsed = start.elapsed();

    let gold: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(out, gold, "scalar simple_add diverged from scalar gold");

    info!(n = N, ?elapsed, "scalar simple_add complete");
}
