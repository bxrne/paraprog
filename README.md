# cs149 - databass.dev bookclub

Rust workspace for CS149 assignments.

## Members

- `assn1-prog1` — parallel Mandelbrot renderer. Sweeps thread counts `1..=available_parallelism`, logs timing/speedup per count via `tracing`, writes the highest-thread-count render to `a1-mandelbrot.ppm`.
- `assn1-prog2` — clamped exponentiation (`output[i] = min(values[i] ^ exponents[i], 9.999999)`). Serial (`clamped_exp_serial`) and SIMD (`clamped_exp_simd`) implementations, logs timing per implementation via `tracing`.

## Running

```
cargo run -p assn1-prog1
cargo run -p assn1-prog2
```

Set `RUST_LOG=info` for per-run timing logs.

## Testing

```
cargo test
```
