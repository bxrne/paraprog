# cs149 - databass.dev bookclub

Rust workspace for CS149 assignments.

## Members

- `assn1-prog1` — parallel Mandelbrot renderer. Sweeps thread counts `1..=available_parallelism`, logs timing/speedup per count via `tracing`, writes the highest-thread-count render to `a1-mandelbrot.ppm`.
- `assn1-prog2` — clamped exponentiation (`output[i] = min(values[i] ^ exponents[i], 9.999999)`), from CS149 asst1 prog2 (`prog2_vecintrin`). `clamped_exp_serial` is the scalar reference. `clamped_exp_vector` models the assignment's fake vector intrinsics with a configurable lane count, correct for any input length and vector width; `main` sweeps `VECTOR_WIDTH` over `2, 4, 8, 16` and logs total vector instructions and vector utilization per width — matching the assignment's grading metrics. `clamped_exp_simd` is additionally a real hardware SIMD implementation using AVX2 intrinsics (8-wide `f32`, runtime-detected via `is_x86_feature_detected!`, scalar fallback otherwise), benchmarked against `clamped_exp_serial` in `main` for an actual wall-clock speedup.

## Running

```
cargo run -p assn1-prog1
cargo run -p assn1-prog2
```

Set `RUST_LOG=info` for per-run timing/utilization logs.

## Testing

```
cargo test
```
