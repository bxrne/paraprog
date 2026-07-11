# assn1-prog2: Vectorizing Code Using SIMD Intrinsics

> Vectorize clamped exponentiation and explain how vector utilization changes as the simulated vector width changes.

- `clamped_exp_serial` is the scalar reference. `clamped_exp_vector` models the assignment's fake vector intrinsics (`CS149intrin.h`) with a runtime `vector_width` parameter rather than a compile-time `#define`, and stays correct for any input length by masking off out-of-bounds lanes on the final chunk instead of relying on a separate scalar tail.
- `main` sweeps `vector_width` over 2, 4, 8, and 16 and logs total vector instructions and vector utilization per width, which are the assignment's actual grading metrics rather than wall-clock time.
- Vector utilization plateaus once the width exceeds the largest exponent present in a chunk, because every lane in a chunk has to wait in lockstep for the slowest (highest-exponent) lane before the chunk can move on, so wider vectors just add more idle, masked-off lanes without any corresponding drop in instruction count.
- `clamped_exp_simd` is an additional, non-assignment implementation using real AVX2 intrinsics (`std::arch::x86_64`) with runtime feature detection and a scalar fallback, added on request to compare simulated vectorization against actual hardware SIMD.
- The AVX2 version is faster over the scalar reference. Three things eat into that: the compare-and-blend instructions needed to mask off lanes that have already finished add work that the scalar loop never has to do; the blend for a given iteration depends on that same iteration's multiply, which depends on the previous iteration's result, so the vector loop has a longer dependent instruction chain per step than a single scalar multiply; and moving the arrays through memory is a shared cost that both versions pay regardless of how parallel the arithmetic is.

## Running

```
cargo run -p assn1-prog2
```

Set `RUST_LOG=info` for per-width vector utilization logs and the serial-vs-AVX2 benchmark.

## Testing

```
cargo test -p assn1-prog2
```
