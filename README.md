# pixelfmt

Pixel format conversions in pure Rust with SIMD optimizations on x86\_64 and
aarch64. The performance goal is to approximately reach the memory bandwidth
limit and thus optimal performance when the input and output are not already
in CPU cache.
