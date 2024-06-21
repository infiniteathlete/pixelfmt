// Copyright (C) 2024 Infinite Athlete, Inc. <av-eng@infiniteathlete.ai>
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [UYVY](https://fourcc.org/pixel-format/yuv-uyvy/) to
//! [I420](https://fourcc.org/pixel-format/yuv-i420/) conversion.
//!
//! Calling crates should use this solely via the `convert_uyvy_to_i420` re-export.
//! Other portions of this module are `pub` solely for use in the included benchmarks.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64;

use crate::{
    frame::{Frame, FrameMut},
    ConversionError,
};

/// Processes a block of 2 rows.
#[doc(hidden)]
pub trait RowProcessor: Copy + Clone + Sized + Send + Sync {
    /// Returns true if this block type is supported on this machine.
    fn new() -> Result<Self, ConversionError>;

    /// Processes a block `width` pixels wide, two rows high.
    ///
    /// # Safety
    ///
    /// Caller must ensure the following:
    /// * `top_uyvy_addr` and `bot_uyvy_addr` each contain `2 * width` bytes of initialized data.
    /// * `top_y_addr` and `bot_y_addr` are each valid destinations for `width` bytes.
    ///   They may however alias each other, which would not be allowed with `&mut`.
    /// * `u_addr` and `v_addr` are each valid destinations for `(width + 1) / 2` bytes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn process(
        self,
        width: usize,
        top_uyvy_addr: *const u8,
        bot_uyvy_addr: *const u8,
        top_y_addr: *mut u8,
        bot_y_addr: *mut u8,
        u_addr: *mut u8,
        v_addr: *mut u8,
    );
}

/// Converts [UYVY](https://fourcc.org/pixel-format/yuv-uyvy/) to [I420](https://fourcc.org/pixel-format/yuv-i420/).
///
/// `uyvy_in` must be fully initialized. `yuv_out` will be fully initialized on success.
pub fn convert<FI: Frame, FO: FrameMut>(
    uyvy_in: &FI,
    yuv_out: &mut FO,
) -> Result<(), ConversionError> {
    #[cfg(target_arch = "x86_64")]
    return convert_with::<ExplicitAvx2DoubleBlock, _, _>(uyvy_in, yuv_out);

    #[cfg(target_arch = "aarch64")]
    return convert_with::<ExplicitNeon, _, _>(uyvy_in, yuv_out);

    #[allow(unused)]
    Err(ConversionError("no block processor available"))
}

#[doc(hidden)]
pub fn convert_with<P: RowProcessor, FI: Frame, FO: FrameMut>(
    uyvy_in: &FI,
    yuv_out: &mut FO,
) -> Result<(), ConversionError> {
    assert!(uyvy_in.initialized());
    let (width, height) = uyvy_in.pixel_dimensions();
    if uyvy_in.format() != crate::PixelFormat::UYVY422
        || yuv_out.format() != crate::PixelFormat::I420
        || yuv_out.pixel_dimensions() != (width, height)
    {
        return Err(ConversionError("invalid arguments"));
    }
    let p = P::new()?;
    let pixels = width * height;
    let uyvy_planes = uyvy_in.planes();
    let [uyvy_in] = &uyvy_planes[..] else {
        panic!("uyvy_in must have one plane");
    };
    let mut yuv_planes = yuv_out.planes_mut();
    let [y_out, u_out, v_out] = &mut yuv_planes[..] else {
        panic!("yuv_out must have three planes");
    };
    let chroma_width = (width >> 1) + (width & 1);
    let chroma_rows = (height >> 1) + (height & 1);
    let chroma_size = chroma_width * chroma_rows;
    if y_out.stride() != width || u_out.stride() != chroma_width || v_out.stride() != chroma_width {
        return Err(ConversionError("output padding unsupported")); // TODO
    }
    let uyvy_stride = uyvy_in.stride();
    assert!(uyvy_stride >= width * 2);
    assert!(uyvy_in.len() >= height * uyvy_stride - (uyvy_stride - width * 2));
    let uyvy_in = uyvy_in.as_ptr();
    assert_eq!(y_out.len(), pixels);
    assert_eq!(u_out.len(), chroma_size);
    assert_eq!(v_out.len(), chroma_size);
    let y_out = y_out.as_mut_ptr();
    let u_out = u_out.as_mut_ptr();
    let v_out = v_out.as_mut_ptr();
    let mut r = 0;
    loop {
        if r + 2 > height {
            break;
        }
        unsafe {
            p.process(
                width,
                uyvy_in.add(r * uyvy_stride),
                uyvy_in.add((r + 1) * uyvy_stride),
                y_out.add(r * width),
                y_out.add((r + 1) * width),
                u_out.add((r >> 1) * chroma_width),
                v_out.add((r >> 1) * chroma_width),
            );
        }
        r += 2;
    }
    if r < height {
        // Process the last row, without subsampling vertically.
        unsafe {
            p.process(
                width,
                uyvy_in.add(r * uyvy_stride),
                uyvy_in.add(r * uyvy_stride), // aliased!
                y_out.add(r * width),
                y_out.add(r * width), // aliased!
                u_out.add((r >> 1) * chroma_width),
                v_out.add((r >> 1) * chroma_width),
            );
        }
    }
    drop(yuv_planes);
    unsafe { yuv_out.initialize() };
    Ok(())
}

#[allow(dead_code)] // occasionally useful for debugging in tests.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hexprint(v: std::arch::x86_64::__m256i) -> impl std::fmt::Display {
    use std::arch::x86_64::_mm256_extract_epi64;
    format!(
        "{:032x} {:032x}",
        u128::from(_mm256_extract_epi64(v, 3) as u64) << 64
            | u128::from(_mm256_extract_epi64(v, 2) as u64),
        u128::from(_mm256_extract_epi64(v, 1) as u64) << 64
            | u128::from(_mm256_extract_epi64(v, 0) as u64)
    )
}

unsafe fn fallback(
    width: usize,
    top_uyvy_addr: *const u8,
    bot_uyvy_addr: *const u8,
    top_y_addr: *mut u8,
    bot_y_addr: *mut u8,
    u_addr: *mut u8,
    v_addr: *mut u8,
) {
    for i in 0..width {
        std::ptr::write(
            top_y_addr.add(i),
            std::ptr::read(top_uyvy_addr.add(2 * i + 1)),
        );
        std::ptr::write(
            bot_y_addr.add(i),
            std::ptr::read(bot_uyvy_addr.add(2 * i + 1)),
        );
    }
    let avg = |a: u8, b: u8| ((u16::from(a) + u16::from(b) + 1) >> 1) as u8;
    let chroma_width = (width >> 1) + (width & 1);
    for i in 0..chroma_width {
        let top_u = std::ptr::read(top_uyvy_addr.add(4 * i));
        let bot_u = std::ptr::read(bot_uyvy_addr.add(4 * i));
        let top_v = std::ptr::read(top_uyvy_addr.add(4 * i + 2));
        let bot_v = std::ptr::read(bot_uyvy_addr.add(4 * i + 2));
        std::ptr::write(u_addr.add(i), avg(top_u, bot_u));
        std::ptr::write(v_addr.add(i), avg(top_v, bot_v));
    }
}

#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct ExplicitAvx2DoubleBlock(());

#[cfg(target_arch = "x86_64")]
impl RowProcessor for ExplicitAvx2DoubleBlock {
    #[inline]
    fn new() -> Result<Self, ConversionError> {
        if is_x86_feature_detected!("avx2") {
            Ok(Self(()))
        } else {
            Err(ConversionError("avx2 is not supported on this machine"))
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline(never)]
    unsafe fn process(
        self,
        width: usize,
        top_uyvy_addr: *const u8,
        bot_uyvy_addr: *const u8,
        top_y_addr: *mut u8,
        bot_y_addr: *mut u8,
        u_addr: *mut u8,
        v_addr: *mut u8,
    ) {
        // Put data[i] into 32-bit groups: lower 128-bits = (y0 y1 u0 v0) upper = (y2 y3 u1 v1).
        // Source indexes, applied to each 128-bit lane within the 256-bit register.
        let shuf_indices = x86_64::_mm256_broadcastsi128_si256(x86_64::_mm_setr_epi8(
            1, 3, 5, 7, 9, 11, 13, 15, // lower half: 8 Y components.
            0, 4, 8, 12, 2, 6, 10, 14, // upper half: (4 * U), (4 * V).
        ));

        // Process the nice blocks.
        const BLOCK_SIZE: usize = 64;
        let mut i = 0;
        loop {
            let top_uyvy_addr = top_uyvy_addr.add(2 * i);
            let bot_uyvy_addr = bot_uyvy_addr.add(2 * i);
            let top_y_addr = top_y_addr.add(i);
            let bot_y_addr = bot_y_addr.add(i);
            let u_addr = u_addr.add(i / 2);
            let v_addr = v_addr.add(i / 2);
            if i + BLOCK_SIZE > width {
                break;
            }
            let [top, bot] = [top_uyvy_addr, bot_uyvy_addr].map(|uyvy_addr| -> [_; 4] {
                std::array::from_fn(|i| {
                    // VMOVDQU (YMM, M256) on Zen2: lat <8, cpi 0.5
                    let raw = x86_64::_mm256_loadu_si256(uyvy_addr.add(32 * i) as _);
                    // VPSHUFB (YMM, YMM, YMM) on Zen2: lat 1; cpi 0.5; ports 1*FP12.
                    x86_64::_mm256_shuffle_epi8(raw, shuf_indices)
                })
            });
            for (data, addr) in [(top, top_y_addr), (bot, bot_y_addr)] {
                for i in [0, 1] {
                    // Put into 64-groups: y0 y2 y1 y3.
                    // VPUNPCKLQDQ (YMM, YMM, YMM) on Zen2: lat 1, cpi 0.50, ports 1*FP12
                    let swapped = x86_64::_mm256_unpacklo_epi64(data[2 * i], data[2 * i + 1]);

                    // Swap y2 and y1 to produce: y0 y1 y2 y3.
                    // VPERMQ (YMM, YMM, I8) on Zen2: lat 6, cpi 1.27
                    x86_64::_mm256_storeu_si256(
                        addr.add(32 * i) as _,
                        x86_64::_mm256_permute4x64_epi64::<0b11_01_10_00>(swapped),
                    );
                }
            }
            let uv: [_; 2] = std::array::from_fn(|i| {
                // unpackhi_epi32(data[0], data[1]) returns (u0 u2 v0 v2) (u1 u3 v1 v3).
                // VPUNPCKHDQ (YMM, YMM, YMM) on Zen2: lat 1, cpi 0.50, ports 1*FP12
                x86_64::_mm256_avg_epu8(
                    x86_64::_mm256_unpackhi_epi32(top[2 * i], top[2 * i + 1]),
                    x86_64::_mm256_unpackhi_epi32(bot[2 * i], bot[2 * i + 1]),
                )
            });
            let mix = x86_64::_mm256_permute2x128_si256::<0b_00_11_00_01>(uv[0], uv[1]);
            let uv0prime =
                x86_64::_mm256_inserti128_si256::<1>(uv[0], x86_64::_mm256_castsi256_si128(uv[1]));
            x86_64::_mm256_storeu_si256(u_addr as _, x86_64::_mm256_unpacklo_epi32(uv0prime, mix));
            x86_64::_mm256_storeu_si256(v_addr as _, x86_64::_mm256_unpackhi_epi32(uv0prime, mix));
            i += BLOCK_SIZE;
        }
        if i < width {
            fallback(
                width - i,
                top_uyvy_addr.add(2 * i),
                bot_uyvy_addr.add(2 * i),
                top_y_addr.add(i),
                bot_y_addr.add(i),
                u_addr.add(i / 2),
                v_addr.add(i / 2),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct ExplicitAvx2SingleBlock(());

#[cfg(target_arch = "x86_64")]
impl RowProcessor for ExplicitAvx2SingleBlock {
    #[inline]
    fn new() -> Result<Self, ConversionError> {
        if is_x86_feature_detected!("avx2") {
            Ok(Self(()))
        } else {
            Err(ConversionError("avx2 is not supported on this machine"))
        }
    }

    #[inline(never)]
    #[target_feature(enable = "avx2")]
    unsafe fn process(
        self,
        width: usize,
        top_uyvy_addr: *const u8,
        bot_uyvy_addr: *const u8,
        top_y_addr: *mut u8,
        bot_y_addr: *mut u8,
        u_addr: *mut u8,
        v_addr: *mut u8,
    ) {
        // Put data[i] into 32-bit groups: lower 128-bits = (y0 y1 u0 v0) upper = (y2 y3 u1 v1).
        // Source indexes, applied to each 128-bit lane within the 256-bit register.
        let shuf_indices = x86_64::_mm256_broadcastsi128_si256(x86_64::_mm_setr_epi8(
            1, 3, 5, 7, 9, 11, 13, 15, // lower half: 8 Y components.
            0, 4, 8, 12, 2, 6, 10, 14, // upper half: (4 * U), (4 * V).
        ));
        // Process the nice blocks.
        const BLOCK_SIZE: usize = 32;
        let mut i = 0;
        loop {
            let top_uyvy_addr = top_uyvy_addr.add(2 * i);
            let bot_uyvy_addr = bot_uyvy_addr.add(2 * i);
            let top_y_addr = top_y_addr.add(i);
            let bot_y_addr = bot_y_addr.add(i);
            let u_addr = u_addr.add(i / 2);
            let v_addr = v_addr.add(i / 2);
            if i + BLOCK_SIZE > width {
                break;
            }
            let [top, bot] = [top_uyvy_addr, bot_uyvy_addr].map(|uyvy_addr| -> [_; 2] {
                std::array::from_fn(|i| {
                    // VMOVDQU (YMM, M256) on Zen2: lat <8, cpi 0.5
                    let raw = x86_64::_mm256_loadu_si256(uyvy_addr.add(32 * i) as _);
                    // VPSHUFB (YMM, YMM, YMM) on Zen2: lat 1; cpi 0.5; ports 1*FP12.
                    x86_64::_mm256_shuffle_epi8(raw, shuf_indices)
                })
            });
            for (data, y_addr) in [(top, top_y_addr), (bot, bot_y_addr)] {
                let y = x86_64::_mm256_unpacklo_epi64(data[0], data[1]);
                // VMOVDQU (M256, YMM) on Zen2: ports 1*FP2.
                x86_64::_mm256_storeu_si256(
                    y_addr as _,
                    x86_64::_mm256_permute4x64_epi64::<0b11_01_10_00>(y),
                );
            }

            let uv = x86_64::_mm256_avg_epu8(
                x86_64::_mm256_unpackhi_epi32(top[0], top[1]),
                x86_64::_mm256_unpackhi_epi32(bot[0], bot[1]),
            );
            let p = x86_64::_mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            x86_64::_mm256_storeu2_m128i(
                v_addr as _,
                u_addr as _,
                x86_64::_mm256_permutevar8x32_epi32(uv, p),
            );
            i += BLOCK_SIZE;
        }
        if i < width {
            fallback(
                width - i,
                top_uyvy_addr.add(2 * i),
                bot_uyvy_addr.add(2 * i),
                top_y_addr.add(i),
                bot_y_addr.add(i),
                u_addr.add(i / 2),
                v_addr.add(i / 2),
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct ExplicitNeon(());

#[cfg(target_arch = "aarch64")]
impl RowProcessor for ExplicitNeon {
    fn new() -> Result<Self, ConversionError> {
        if std::arch::is_aarch64_feature_detected!("neon") {
            Ok(Self(()))
        } else {
            Err(ConversionError("neon unsupported on this machine"))
        }
    }

    #[inline(never)]
    #[target_feature(enable = "neon")]
    unsafe fn process(
        self,
        width: usize,
        top_uyvy_addr: *const u8,
        bot_uyvy_addr: *const u8,
        top_y_addr: *mut u8,
        bot_y_addr: *mut u8,
        u_addr: *mut u8,
        v_addr: *mut u8,
    ) {
        const BLOCK_SIZE: usize = 32;
        let mut i = 0;
        loop {
            if i + BLOCK_SIZE > width {
                break;
            }
            let top_uyvy = aarch64::vld4q_u8(top_uyvy_addr.add(2 * i));
            let bot_uyvy = aarch64::vld4q_u8(bot_uyvy_addr.add(2 * i));
            aarch64::vst2q_u8(
                top_y_addr.add(i),
                aarch64::uint8x16x2_t(top_uyvy.1, top_uyvy.3),
            );
            aarch64::vst2q_u8(
                bot_y_addr.add(i),
                aarch64::uint8x16x2_t(bot_uyvy.1, bot_uyvy.3),
            );
            aarch64::vst1q_u8(
                u_addr.add(i / 2),
                aarch64::vrhaddq_u8(top_uyvy.0, bot_uyvy.0),
            );
            aarch64::vst1q_u8(
                v_addr.add(i / 2),
                aarch64::vrhaddq_u8(top_uyvy.2, bot_uyvy.2),
            );
            i += BLOCK_SIZE;
        }
        if i < width {
            fallback(
                width - i,
                top_uyvy_addr.add(2 * i),
                bot_uyvy_addr.add(2 * i),
                top_y_addr.add(i),
                bot_y_addr.add(i),
                u_addr.add(i / 2),
                v_addr.add(i / 2),
            );
        }
    }
}

/// Creates a block processor that we hope the compiler will auto-vectorize.
///
/// The target feature(s) to require (and thus allow the compiler to use) are
/// given at macro expansion time; the block size in pixels is a const generic
/// in the generated struct/impl.
macro_rules! auto {
    {$ident:ident supported { $($supported:tt)+ } features { $($feature:literal)* }} => {
        #[doc(hidden)]
        #[derive(Copy, Clone)]
        pub struct $ident<const PIXELS: usize>([(); PIXELS]);

        impl<const PIXELS: usize> RowProcessor for $ident<PIXELS> {
            #[inline(always)]
            fn new() -> Result<Self, ConversionError> {
                if true && $($supported)+ {
                    Ok(Self(std::array::from_fn(|_| ())))
                } else {
                    Err(ConversionError(concat!(stringify!($ident), " unsupported on this machine")))
                }
            }

            #[inline(never)]
            $(#[target_feature(enable = $feature)])*
            unsafe fn process(
                self,
                width: usize,
                top_uyvy_addr: *const u8,
                bot_uyvy_addr: *const u8,
                top_y_addr: *mut u8,
                bot_y_addr: *mut u8,
                u_addr: *mut u8,
                v_addr: *mut u8,
            ) {
                let mut i = 0;

                // The u/v logic below doesn't handle an odd number of pixels per block.
                const { assert!(PIXELS % 2 == 0); }
                loop {
                    if i + PIXELS > width {
                        break;
                    }
                    let top_uyvy_addr = top_uyvy_addr.add(2 * i);
                    let bot_uyvy_addr = bot_uyvy_addr.add(2 * i);
                    let top_y_addr = top_y_addr.add(i);
                    let bot_y_addr = bot_y_addr.add(i);
                    let u_addr = u_addr.add(i / 2);
                    let v_addr = v_addr.add(i / 2);
                    for j in 0..PIXELS {
                        std::ptr::write(top_y_addr.add(j), std::ptr::read(top_uyvy_addr.add(2*j + 1)));
                        std::ptr::write(bot_y_addr.add(j), std::ptr::read(bot_uyvy_addr.add(2*j + 1)));
                    }
                    let avg = |a: u8, b: u8| { (u16::from(a) + u16::from(b) + 1 >> 1) as u8 };
                    for j in 0..PIXELS/2 {
                        let top_u = std::ptr::read(top_uyvy_addr.add(4*j));
                        let bot_u = std::ptr::read(bot_uyvy_addr.add(4*j));
                        let top_v = std::ptr::read(top_uyvy_addr.add(4*j + 2));
                        let bot_v = std::ptr::read(bot_uyvy_addr.add(4*j + 2));
                        std::ptr::write(u_addr.add(j), avg(top_u, bot_u));
                        std::ptr::write(v_addr.add(j), avg(top_v, bot_v));
                    }
                    i += PIXELS;
                }
                if i < width {
                    fallback(
                        width - i,
                        top_uyvy_addr.add(2 * i),
                        bot_uyvy_addr.add(2 * i),
                        top_y_addr.add(i),
                        bot_y_addr.add(i),
                        u_addr.add(i / 2),
                        v_addr.add(i / 2),
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
auto! {
    AutoAvx2Block
    supported { is_x86_feature_detected!("avx2") }
    features { "avx2" }
}

#[cfg(target_arch = "aarch64")]
auto! {
    AutoNeonBlock
    supported { std::arch::is_aarch64_feature_detected!("neon") }
    features { "neon" }
}

auto! {
    AutoVanillaBlock
    supported { true }
    features { }
}

#[cfg(test)]
mod tests {
    macro_rules! test_processor {
        ($processor: ty, $mod: ident, $pixels: expr) => {
            mod $mod {
                use super::super::RowProcessor as _;
                type P = $processor;

                /// Tests that a single `process` call produces the right `y` plane bytes.
                #[test]
                fn y() {
                    let p = P::new().unwrap();
                    const PIXELS: usize = $pixels;
                    let mut top_in = vec![0xff; PIXELS * 4];
                    let mut bot_in = vec![0xff; PIXELS * 4];
                    for i in 0..PIXELS {
                        top_in[2 * i + 1] = i as u8;
                        bot_in[2 * i + 1] = !(i as u8);
                    }
                    let mut top_y_out = vec![0xff; PIXELS];
                    let mut bot_y_out = vec![0xff; PIXELS];
                    let mut u_out = vec![0xff; PIXELS / 2];
                    let mut v_out = vec![0xff; PIXELS / 2];
                    unsafe {
                        p.process(
                            PIXELS,
                            top_in.as_ptr(),
                            bot_in.as_ptr(),
                            top_y_out.as_mut_ptr(),
                            bot_y_out.as_mut_ptr(),
                            u_out.as_mut_ptr(),
                            v_out.as_mut_ptr(),
                        )
                    }
                    let expected_top_y_out: [u8; PIXELS] = std::array::from_fn(|i| i as u8);
                    let expected_bot_y_out: [u8; PIXELS] = std::array::from_fn(|i| !(i as u8));
                    assert_eq!(&top_y_out[..], &expected_top_y_out[..]);
                    assert_eq!(&bot_y_out[..], &expected_bot_y_out[..]);
                }

                /// Tests that a single `process` call produces the right `u` and `v` plane bytes.
                #[test]
                fn uv() {
                    let p = P::new().unwrap();
                    const PIXELS: usize = $pixels;
                    let mut top_in = vec![0xff; PIXELS * 4];
                    let mut bot_in = vec![0xff; PIXELS * 4];
                    for i in 0..PIXELS {
                        // u values avg to 0x20 + i (rounding up).
                        top_in[4 * i] = 0x10 + i as u8;
                        bot_in[4 * i] = 0x30 + i as u8;

                        // v values avg to 0x90 + i.
                        top_in[4 * i + 2] = 0x80 + i as u8;
                        bot_in[4 * i + 2] = 0xa0 + i as u8;
                    }
                    let mut top_y_out = vec![0xff; PIXELS];
                    let mut bot_y_out = vec![0xff; PIXELS];
                    let mut u_out = vec![0xff; PIXELS / 2];
                    let mut v_out = vec![0xff; PIXELS / 2];
                    unsafe {
                        p.process(
                            PIXELS,
                            top_in.as_ptr(),
                            bot_in.as_ptr(),
                            top_y_out.as_mut_ptr(),
                            bot_y_out.as_mut_ptr(),
                            u_out.as_mut_ptr(),
                            v_out.as_mut_ptr(),
                        )
                    }
                    let expected_u_out: [u8; PIXELS / 2] = std::array::from_fn(|i| 0x20 + i as u8);
                    let expected_v_out: [u8; PIXELS / 2] = std::array::from_fn(|i| 0x90 + i as u8);
                    assert_eq!(&u_out[..], &expected_u_out[..]);
                    assert_eq!(&v_out[..], &expected_v_out[..]);
                }

                /// Tests a full realistic frame.
                #[cfg(not(miri))] // slow!
                #[test]
                fn full_frame() {
                    use crate::{
                        frame::{ConsecutiveFrame, Frame as _},
                        PixelFormat,
                    };

                    // Test input created with:
                    // ```
                    // ffmpeg -y -f lavfi -i testsrc=size=1280x720,format=uyvy422 -frames 1 in.yuv
                    // ffmpeg -y -pix_fmt uyvy422 -s 1280x720 -i in.yuv -pix_fmt yuv420p almost_out.yuv
                    // ```
                    // ffmpeg apparently rounds down in this conversion, but we follow the libyuv
                    // example of rounding up, so the output isn't actually from ffmpeg.
                    const WIDTH: usize = 1280;
                    const HEIGHT: usize = 720;
                    let uyvy_in = ConsecutiveFrame::new(PixelFormat::UYVY422, WIDTH, HEIGHT)
                        .with_storage(&include_bytes!("testdata/in.yuv")[..]);
                    let mut actual_out =
                        ConsecutiveFrame::new(PixelFormat::I420, WIDTH, HEIGHT).new_vec();
                    let expected_out = ConsecutiveFrame::new(PixelFormat::I420, WIDTH, HEIGHT)
                        .with_storage(&include_bytes!("testdata/out.yuv")[..]);
                    super::super::convert_with::<P, _, _>(&uyvy_in, &mut actual_out).unwrap();
                    // `assert_eq!` output is unhelpful on these large binary arrays.
                    // On failure, it might be better to write to a file and diff with better tools,
                    // e.g.: `diff -u <(xxd src/testdata/out.yuv) <(xxd actual_out_auto.yuv)`
                    // std::fs::write(
                    //     concat!("actual_out_", stringify!($mod), ".yuv"),
                    //     &actual_out.inner(),
                    // )
                    // .unwrap();
                    assert!(expected_out.planes() == actual_out.planes());
                }

                /// Tests a 3x3 frame, which is noteworthy in two ways.
                /// * It exercises the special aliasing last row case.
                /// * It exercises typical `RowProcessor`s' fallback paths.
                #[test]
                #[rustfmt::skip]
                fn size3x3() {
                    use crate::{frame::ConsecutiveFrame, PixelFormat};
                    let uyvy_in = ConsecutiveFrame::new(PixelFormat::UYVY422, 3, 3).with_storage(&[
                        // U0-1  Y0    V0-1  Y1    U2-2  Y2    V2-2  Yx
                           0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // top row
                           0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, // middle row
                           0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, // bottom row
                    ][..]);
                    let expected_out = ConsecutiveFrame::new(PixelFormat::I420, 3, 3).with_storage(&[
                        // Y0    Y1    Y2
                        0x01, 0x03, 0x05, // top row
                        0x0a, 0x0c, 0x0e, // middle row
                        0x13, 0x15, 0x17, // bottom row

                        // U0-1 U2-2
                        0x05, 0x09, // top+middle rows
                        0x12, 0x16, // bottom row

                        // V0-1 V2-2
                        0x07,   0x0b, // top+middle rows
                        0x14,   0x18, // bottom row
                    ][..]);
                    let mut actual_out =
                        ConsecutiveFrame::new(PixelFormat::I420, 3, 3).new_vec();
                    super::super::convert_with::<P, _, _>(&uyvy_in, &mut actual_out).unwrap();
                    assert_eq!(expected_out.inner(), actual_out.inner());
                }
            }
        };
    }

    #[cfg(target_arch = "x86_64")]
    #[cfg(not(miri))] // vendor instrinsics are unsupported on miri.
    test_processor!(
        super::super::ExplicitAvx2DoubleBlock,
        explicit_double_avx2,
        64
    );

    #[cfg(target_arch = "x86_64")]
    #[cfg(not(miri))] // vendor instrinsics are unsupported on miri.
    test_processor!(
        super::super::ExplicitAvx2SingleBlock,
        explicit_single_avx2,
        32
    );

    #[cfg(target_arch = "x86_64")]
    #[cfg(not(miri))] // vendor instrinsics are unsupported on miri.
    test_processor!(super::super::AutoAvx2Block<32>, auto_avx2, 32);

    #[cfg(target_arch = "aarch64")]
    test_processor!(super::super::AutoNeonBlock<64>, auto_neon, 64);

    #[cfg(target_arch = "aarch64")]
    #[cfg(not(miri))] // vendor instrinsics are unsupported on miri.
    test_processor!(super::super::ExplicitNeon, explicit_neon, 32);

    test_processor!(super::super::AutoVanillaBlock<32>, auto, 32);
}
