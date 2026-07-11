# assn1-prog1: Parallel Fractal Generation Using Threads

> Parallelize Mandelbrot set generation across threads and reason about why speedup falls short of linear as thread count grows.

- `mandelbrot::render` is the serial reference; `mandelbrot::render_parallel` splits the image into row blocks across a caller-supplied thread count with no synchronization between threads, matching the assignment's "single static decomposition policy" constraint.
- `main` sweeps `1..=available_parallelism` threads, logs elapsed time and speedup per count via `tracing`, and writes the fastest run to `a1-mandelbrot.ppm`.
- Speedup is sublinear past the physical core count because hyperthreads share execution resources on a core rather than adding independent ones, and because a naive row-per-thread split gives some threads more escape-time work than others when the set's boundary is unevenly distributed across rows.

## Running

```
cargo run -p assn1-prog1
```

Set `RUST_LOG=info` for per-thread-count timing and speedup logs.

## Testing

```
cargo test -p assn1-prog1
```
