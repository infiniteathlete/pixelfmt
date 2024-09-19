# pixelfmt

Pixel format conversions in pure Rust with SIMD optimizations on x86\_64 and
aarch64. The performance goal is to approximately reach the memory bandwidth
limit and thus optimal performance when the input and output are not already
in CPU cache.

Limitations and future work:

*   Supports exactly one conversion:
    [UYVY](https://fourcc.org/pixel-format/yuv-uyvy/) to
    [I420](https://fourcc.org/pixel-format/yuv-i420/).
    More will be added as needed.
*   Expects to process full horizontal lines. This is likely to
    change to allow working on cropped regions.
*   Does not support output to a frame with padding, as required by some
    APIs/devices.
*   The ARM NEON code is less optimized than the AVX2 code today.

You may find the notes in [`docs/simd.md`](docs/simd.md) helpful if you are new
to SIMD and thinking of contributing.

## Alternatives

The main alternative to `pixelfmt` is Chrome's C++
[libyuv](https://chromium.googlesource.com/libyuv/libyuv) library.
Rust bindings are available via the `yuv-sys` or `libyuv` crates.

Some reasons to prefer `pixelfmt`:

*   `pixelfmt` is pure Rust and thus may be easier and quicker to build,
    particularly when cross-compiling.
*   `pixelfmt` is less total code.
*   `pixelfmt`'s `uyvy_to_i420` implementation benchmarks as slightly faster than
    `libyuv`'s. This appears to come down to design choices: `libyuv` has
    separate routines to extract the Y and U+V planes, where `pixelfmt` gains
    some performance by doing it all with a single read of the UYVY data.
    `pixelfmt` also has a hand-optimized conversion function where `libyuv`
    uses a more generalized autovectorization approach.
*   `pixelfmt` uses safe Rust when possible, where `libyuv` is entirely written
    in (unsafe) C++. That said, `pixelfmt` still has a fair bit of `unsafe`
    logic for vendor SIMD intrinsics.
    
Some reasons to prefer `libyuv`:

*   `libyuv` is much more widely used.
*   `libyuv` is much more comprehensive and flexible.

## License

SPDX-License-Identifier: [MIT](https://spdx.org/licenses/MIT.html) OR [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

See [LICENSE-MIT.txt](LICENSE-MIT.txt) or [LICENSE-APACHE](LICENSE-APACHE.txt),
respectively.
