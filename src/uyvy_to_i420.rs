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
use std::mem::MaybeUninit;

#[derive(Debug)]
pub struct Unsupported;

/// Processes a block `PIXELS` columns wide, 2 rows high.
#[doc(hidden)]
pub trait BlockProcessor: Copy + Clone + Sized + Send + Sync {
    /// The width of this block in pixels.
    const PIXELS: usize;

    /// Returns true if this block type is supported on this machine.
    fn new() -> Result<Self, Unsupported>;

    /// Processes a block `PIXELS` wide, two rows high.
    ///
    /// # Safety
    ///
    /// Caller must ensure the following:
    /// * `top_uyvy_addr` and `bot_uyvy_addr` each contain `2 * PIXELS` bytes of initialized data.
    /// * `top_y_addr` and `bot_y_addr` are each valid destinations for `PIXELS` bytes.
    /// * `u_addr` and `v_addr` are each valid destinations for `PIXELS / 2` bytes.
    unsafe fn process(
        self,
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
/// `uyvy_in` is of the shape `(height, width / 2, 4)` and channels are (U, Y, V, Y).
/// Guaranteed to initialize and fill the supplied output planes on success.
/// `y_out` is of the shape `(height, width)`; `u_out` and `v_out` are each of
/// the shape `(height / 2, width / 2)`.
pub fn convert(
    uyvy_in: &[u8],
    width: usize,
    height: usize,
    y_out: &mut [MaybeUninit<u8>],
    u_out: &mut [MaybeUninit<u8>],
    v_out: &mut [MaybeUninit<u8>],
) -> Result<(), Unsupported> {
    #[cfg(target_arch = "x86_64")]
    return convert_with::<ExplicitAvx2DoubleBlock>(uyvy_in, width, height, y_out, u_out, v_out);

    #[cfg(target_arch = "aarch64")]
    return convert_with::<ExplicitNeon>(uyvy_in, width, height, y_out, u_out, v_out);

    #[allow(unused)]
    Err(Unsupported)
}

#[doc(hidden)]
pub fn convert_with<P: BlockProcessor>(
    uyvy_in: &[u8],
    width: usize,
    height: usize,
    y_out: &mut [MaybeUninit<u8>],
    u_out: &mut [MaybeUninit<u8>],
    v_out: &mut [MaybeUninit<u8>],
) -> Result<(), Unsupported> {
    if width % P::PIXELS != 0 || height % 2 != 0 {
        return Err(Unsupported);
    }
    let p = P::new()?;
    let pixels = width * height;
    assert!(uyvy_in.len() == pixels * 2);
    assert!(y_out.len() == pixels);
    assert!(u_out.len() == pixels / 4);
    assert!(v_out.len() == pixels / 4);
    let uyvy_row_stride = 2 * width; // TODO: support line padding?
    for r in (0..height).step_by(2) {
        for c in (0..width).step_by(P::PIXELS) {
            unsafe {
                let y_out = y_out.as_mut_ptr().cast::<u8>();
                p.process(
                    uyvy_in.as_ptr().add(r * uyvy_row_stride + 2 * c),
                    uyvy_in.as_ptr().add((r + 1) * uyvy_row_stride + 2 * c),
                    y_out.add(r * width + c),
                    y_out.add((r + 1) * width + c),
                    u_out
                        .as_mut_ptr()
                        .cast::<u8>()
                        .add((r >> 1) * (width >> 1) + (c >> 1)),
                    v_out
                        .as_mut_ptr()
                        .cast::<u8>()
                        .add((r >> 1) * (width >> 1) + (c >> 1)),
                );
            }
        }
    }
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

#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct ExplicitAvx2DoubleBlock(());

#[cfg(target_arch = "x86_64")]
impl BlockProcessor for ExplicitAvx2DoubleBlock {
    const PIXELS: usize = 64;

    #[inline]
    fn new() -> Result<Self, Unsupported> {
        if is_x86_feature_detected!("avx2") {
            Ok(Self(()))
        } else {
            Err(Unsupported)
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline(never)]
    unsafe fn process(
        self,
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
    }
}

#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct ExplicitAvx2SingleBlock(());

#[cfg(target_arch = "x86_64")]
impl BlockProcessor for ExplicitAvx2SingleBlock {
    const PIXELS: usize = 32;

    #[inline]
    fn new() -> Result<Self, Unsupported> {
        if is_x86_feature_detected!("avx2") {
            Ok(Self(()))
        } else {
            Err(Unsupported)
        }
    }

    #[inline(never)]
    #[target_feature(enable = "avx2")]
    unsafe fn process(
        self,
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
    }
}

#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct ExplicitNeon(());

#[cfg(target_arch = "aarch64")]
impl BlockProcessor for ExplicitNeon {
    const PIXELS: usize = 32;

    fn new() -> Result<Self, Unsupported> {
        if std::arch::is_aarch64_feature_detected!("neon") {
            Ok(Self(()))
        } else {
            Err(Unsupported)
        }
    }

    #[inline(never)]
    #[target_feature(enable = "neon")]
    unsafe fn process(
        self,
        top_uyvy_addr: *const u8,
        bot_uyvy_addr: *const u8,
        top_y_addr: *mut u8,
        bot_y_addr: *mut u8,
        u_addr: *mut u8,
        v_addr: *mut u8,
    ) {
        let top_uyvy = aarch64::vld4q_u8(top_uyvy_addr);
        let bot_uyvy = aarch64::vld4q_u8(bot_uyvy_addr);
        aarch64::vst2q_u8(top_y_addr, aarch64::uint8x16x2_t(top_uyvy.1, top_uyvy.3));
        aarch64::vst2q_u8(bot_y_addr, aarch64::uint8x16x2_t(bot_uyvy.1, bot_uyvy.3));
        aarch64::vst1q_u8(u_addr, aarch64::vrhaddq_u8(top_uyvy.0, bot_uyvy.0));
        aarch64::vst1q_u8(v_addr, aarch64::vrhaddq_u8(top_uyvy.2, bot_uyvy.2));
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

        impl<const PIXELS: usize> BlockProcessor for $ident<PIXELS> {
            const PIXELS: usize = PIXELS;

            #[inline(always)]
            fn new() -> Result<Self, Unsupported> {
                if true && $($supported)+ {
                    Ok(Self(std::array::from_fn(|_| ())))
                } else {
                    Err(Unsupported)
                }
            }

            #[inline(never)]
            $(#[target_feature(enable = $feature)])*
            unsafe fn process(
                self,
                top_uyvy_addr: *const u8,
                bot_uyvy_addr: *const u8,
                top_y_addr: *mut u8,
                bot_y_addr: *mut u8,
                u_addr: *mut u8,
                v_addr: *mut u8,
            ) {
                for i in 0..PIXELS {
                    std::ptr::write(top_y_addr.add(i), std::ptr::read(top_uyvy_addr.add(2*i + 1)));
                    std::ptr::write(bot_y_addr.add(i), std::ptr::read(bot_uyvy_addr.add(2*i + 1)));
                }
                let avg = |a: u8, b: u8| { (u16::from(a) + u16::from(b) + 1 >> 1) as u8 };
                for i in 0..Self::PIXELS/2 {
                    let top_u = std::ptr::read(top_uyvy_addr.add(4*i));
                    let bot_u = std::ptr::read(bot_uyvy_addr.add(4*i));
                    let top_v = std::ptr::read(top_uyvy_addr.add(4*i + 2));
                    let bot_v = std::ptr::read(bot_uyvy_addr.add(4*i + 2));
                    std::ptr::write(u_addr.add(i), avg(top_u, bot_u));
                    std::ptr::write(v_addr.add(i), avg(top_v, bot_v));
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
    macro_rules! test_block {
        ($processor: ty, $mod: ident) => {
            mod $mod {
                use super::super::BlockProcessor as _;
                type P = $processor;

                /// Tests that a single `process` call produces the right `y` plane bytes.
                #[test]
                fn y() {
                    let p = P::new().unwrap();
                    let mut top_in = vec![0xff; P::PIXELS * 4];
                    let mut bot_in = vec![0xff; P::PIXELS * 4];
                    for i in 0..P::PIXELS {
                        top_in[2 * i + 1] = i as u8;
                        bot_in[2 * i + 1] = !(i as u8);
                    }
                    let mut top_y_out = vec![0xff; P::PIXELS];
                    let mut bot_y_out = vec![0xff; P::PIXELS];
                    let mut u_out = vec![0xff; P::PIXELS / 2];
                    let mut v_out = vec![0xff; P::PIXELS / 2];
                    unsafe {
                        p.process(
                            top_in.as_ptr(),
                            bot_in.as_ptr(),
                            top_y_out.as_mut_ptr(),
                            bot_y_out.as_mut_ptr(),
                            u_out.as_mut_ptr(),
                            v_out.as_mut_ptr(),
                        )
                    }
                    let expected_top_y_out: [u8; P::PIXELS] = std::array::from_fn(|i| i as u8);
                    let expected_bot_y_out: [u8; P::PIXELS] = std::array::from_fn(|i| !(i as u8));
                    assert_eq!(&top_y_out[..], &expected_top_y_out[..]);
                    assert_eq!(&bot_y_out[..], &expected_bot_y_out[..]);
                }

                /// Tests that a single `process` call produces the right `u` and `v` plane bytes.
                #[test]
                fn uv() {
                    let p = P::new().unwrap();
                    let mut top_in = vec![0xff; P::PIXELS * 4];
                    let mut bot_in = vec![0xff; P::PIXELS * 4];
                    for i in 0..P::PIXELS {
                        // u values avg to 0x20 + i (rounding up).
                        top_in[4 * i] = 0x10 + i as u8;
                        bot_in[4 * i] = 0x30 + i as u8;

                        // v values avg to 0x90 + i.
                        top_in[4 * i + 2] = 0x80 + i as u8;
                        bot_in[4 * i + 2] = 0xa0 + i as u8;
                    }
                    let mut top_y_out = vec![0xff; P::PIXELS];
                    let mut bot_y_out = vec![0xff; P::PIXELS];
                    let mut u_out = vec![0xff; P::PIXELS / 2];
                    let mut v_out = vec![0xff; P::PIXELS / 2];
                    unsafe {
                        p.process(
                            top_in.as_ptr(),
                            bot_in.as_ptr(),
                            top_y_out.as_mut_ptr(),
                            bot_y_out.as_mut_ptr(),
                            u_out.as_mut_ptr(),
                            v_out.as_mut_ptr(),
                        )
                    }
                    let expected_u_out: [u8; P::PIXELS / 2] =
                        std::array::from_fn(|i| 0x20 + i as u8);
                    let expected_v_out: [u8; P::PIXELS / 2] =
                        std::array::from_fn(|i| 0x90 + i as u8);
                    assert_eq!(&u_out[..], &expected_u_out[..]);
                    assert_eq!(&v_out[..], &expected_v_out[..]);
                }

                /// Tests a full realistic frame.
                #[test]
                fn full_frame() {
                    // Test input created with:
                    // ```
                    // ffmpeg -y -f lavfi -i testsrc=size=1280x720,format=uyvy422 -frames 1 in.yuv
                    // ffmpeg -y -pix_fmt uyvy422 -s 1280x720 -i in.yuv -pix_fmt yuv420p almost_out.yuv
                    // ```
                    // ffmpeg apparently rounds down in this conversion, but we follow the libyuv
                    // example of rounding up, so the output isn't actually from ffmpeg.
                    const WIDTH: usize = 1280;
                    const HEIGHT: usize = 720;
                    let uyvy_in = include_bytes!("testdata/in.yuv");
                    let expected_out = include_bytes!("testdata/out.yuv");
                    let mut actual_out = Vec::with_capacity(3 * WIDTH * HEIGHT / 2);
                    let (mut y_out, uv_out) =
                        actual_out.spare_capacity_mut().split_at_mut(WIDTH * HEIGHT);
                    let (mut u_out, mut v_out) = uv_out.split_at_mut(WIDTH * HEIGHT / 4);
                    super::super::convert_with::<P>(
                        uyvy_in, WIDTH, HEIGHT, &mut y_out, &mut u_out, &mut v_out,
                    )
                    .unwrap();
                    unsafe {
                        actual_out.set_len(3 * WIDTH * HEIGHT / 2);
                    }
                    // `assert_eq!` output is unhelpful on these large binary arrays.
                    // On failure, it might be better to write to a file and diff with better tools,
                    // e.g.: `diff -u <(xxd src/testdata/out.yuv) <(xxd actual_out_auto.yuv)`
                    // std::fs::write(
                    //     concat!("actual_out_", stringify!($mod), ".yuv"),
                    //     &actual_out[..],
                    // )
                    // .unwrap();
                    assert!(&expected_out[..] == &actual_out[..]);
                }
            }
        };
    }

    #[cfg(target_arch = "x86_64")]
    test_block!(super::super::ExplicitAvx2DoubleBlock, explicit_double_avx2);

    #[cfg(target_arch = "x86_64")]
    test_block!(super::super::ExplicitAvx2SingleBlock, explicit_single_avx2);

    #[cfg(target_arch = "x86_64")]
    test_block!(super::super::AutoAvx2Block<32>, auto_avx2);

    #[cfg(target_arch = "aarch64")]
    test_block!(super::super::AutoNeonBlock<64>, auto_neon);

    #[cfg(target_arch = "aarch64")]
    test_block!(super::super::ExplicitNeon, explicit_neon);

    test_block!(super::super::AutoVanillaBlock<32>, auto);
}
