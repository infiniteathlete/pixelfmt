# SIMD notes

This project is the author's first use of explicit SIMD. Lessons learned:

*   Pixel format conversions are a straightforward use of SIMD because huge
    blocks of pixels are just shuffled around and lightly processed
    independently—no variable-width encodings, branching, or masking required.
    This does not require the level of sophistication you may have heard of
    in such works as [simdjson](https://github.com/simdjson/simdjson).
*   We get diminishing marginal benefits as we move from "scalar instructions
    only" to "autovectorized SIMD" to "novice-quality explicit SIMD" to "expert
    quality explicit SIMD". We don't claim to have reached expert quality here,
    yet appear to be approximately memory bandwidth-limited (thus have optimal
    performance, if not energy efficiency) in cold CPU cache situations.
*   x86_64 SIMD (including AVX2) is much more challenging to write than ARM SIMD
    (NEON). In particular, NEON makes it very straightforward to unzip parallel
    sequences, while it's a challenge in AVX2 to shuffle/permute things to the
    correct lanes. More on that below.
*   AVX2 is more powerful and potentially performant than NEON. For example, it
    supports masked operations which could be used to avoid needing a scalar
    fallback path for the last few pixels in each row.
*   [Godbolt's Compiler Explorer](https://godbolt.org/) can be useful.
    clang, g++, and Rust each have vector types and matching shuffle operations
    that are more general than e.g. AVX2 shuffle/permute intrinsics. In
    particular, one compiler shuffle call may produce a sequence of a few vendor
    shuffle/permute intrinsics/instructions with matching constants. It may not
    be possible to directly take advantage of this in the final code because
    these don't seem to combine well with vendor intrinsics, and because
    (in the case of Rust) they're not available on stable
    ([tracking issue](https://github.com/rust-lang/rust/issues/86656)).
    But Compiler Explorer allows you to write a kernel that does the shuffle you
    want and look at the resulting assembly; it's then straightforward to
    transcribe this to intrinsics use. This can be a bit tedious, but as the
    total lines of code here are quite minimal and could represent the majority
    of a program's CPU time if written without SIMD, it's worthwhile.

## x86-64 AVX2

A couple good references:

*   The [Intel intrinsics guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
    is the authoritative description of what x86_64 intrinsics/instructions
    are meant to do.
*   [uops.info](https://uops.info/) has the best information about the latency,
    throughput, etc. of the instructions on specific processors, based on
    Intel/AMD vendor documentation and microbenchmarks.

One key challenge with AVX2 is that its 256-bit registers are mostly separate
128-bit lanes. This comes up in a variety of ways:

*   many of the "swizzle" (shuffle, permute) operations have the phrase 
    "within 128-bit lanes" (e.g. `_mm256_shuffle_epi8` a.k.a. `vpshufb`).
    It's cheap and easy to move data within these
    lanes; crossing 128-bit lanes seems to be a separate step
    (typically via `_mm256_permute4x64_epi64` a.k.a. `vpermq`, or the more
    flexible but more expensive `_mm256_permutevar8x32_epi32` a.k.a. `vpermd`).
*   some operations combine the top or bottom 128-bit lanes from two different
    256-bit registers rather than processing a full single register (e.g.
    `_mm256_unpackhi_epi16` and `_mm256_unpacklo_epi16`). You then may need to
    use a couple `vpermq` operations to get them back in the order you expect.
    If you're persistent and lucky, you might find use them in a sequence that
    crosses them and then back again without the permutes.

There are many interesting discussions of this on the Internet, including
the following:

* https://stackoverflow.com/questions/66664367/interleave-two-vectors
* https://stackoverflow.com/questions/52982211/best-way-to-shuffle-across-avx-lanes
* https://community.intel.com/t5/Intel-ISA-Extensions/Cross-lane-operations-how/td-p/819068

As mentioned above, Compiler Explorer can be a big help with this challenge.
Here is an example in C++ using clang's
`__builtin_shufflevector` ([live in compiler explorer](https://godbolt.org/z/eqW5Tb6ea),
[clang docs](https://clang.llvm.org/docs/LanguageExtensions.html#vectors-and-extended-vectors)):

```cpp
#include <stdint.h>

typedef int8_t i8x32 __attribute__((__vector_size__(32)));
typedef int32_t i32x8 __attribute__((__vector_size__(32)));
typedef int64_t i64x4 __attribute__((__vector_size__(32)));
typedef int64_t i64x2 __attribute__((__vector_size__(16)));

void pre(i8x32 in, i8x32 *out) {
    *out = __builtin_shufflevector(in, in,
            // Lower 128-bit lane.
            1, 3, 5, 7, 9, 11, 13, 15, // lower half: 8 Y components.
            0, 4, 8, 12, 2, 6, 10, 14, // upper half: (4 * U), (4 * V).
            // Upper 128-bit lane (same layout).
            16+1, 16+3, 16+5, 16+7, 16+9, 16+11, 16+13, 16+15, // lower half: 8 Y components.
            16+0, 16+4, 16+8, 16+12, 16+2, 16+6, 16+10, 16+14 // upper half: (4 * U), (4 * V).
    );
}

void y(i64x4 data0, i64x4 data1, i64x4 *y) {
    *y = __builtin_shufflevector(data0, data1, 0, 2, 4, 6);
}

void double_block(i32x8 uv0, i32x8 uv1, i32x8 *u, i32x8 *v) {
    // (u0 u2 v0 v2) (u1 u3 v1 v3) (u4 u6 v4 v6) (u5 u7 v5 v7)
    *u = __builtin_shufflevector(uv0, uv1, 0, 4, 1, 5, 8, 12, 9, 13);
    *v = __builtin_shufflevector(uv0, uv1, 2, 6, 3, 7, 10, 14, 11, 15);
}

void single_block_split(i32x8 uv, i32x8 *uv_out) {
    // (u0 u2 v0 v2) (u1 u3 v1 v3)
    *uv_out = __builtin_shufflevector(uv, uv, 0, 4, 1, 5, 2, 6, 3, 7);
}
```

Compiler Explorer also supports similar operations with Rust's nightly compiler
and [`std::simd::simd_swizzle!`](https://doc.rust-lang.org/std/simd/macro.simd_swizzle.html).

# ARM NEON

The NEON portions of this code were much easier than AVX2 to write but may not
be optimal. In particular, looking at them with Compiler Explorer shows extra
register moves that seem unnecessary and likely add some latency. Dropping to
inline assembly would allow us to optimize further, at the minor cost of
requiring duplication to support both `aarch64` and (32-bit) `arm` platforms
if the latter is desired.

One reason no great effort has been made for optimization here is that on
macOS, the end goal may be to feed data to Video Toolbox. As that framework
accepts a variety of pixel formats and appears to convert them efficiently.
The author has not looked into whether this happens by SIMD code or by
dedicated silicon, but the result is good nonetheless.
