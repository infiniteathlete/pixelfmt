#[cfg(target_arch = "aarch64")]
use std::arch::aarch64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64;
use std::mem::MaybeUninit;

#[derive(Debug)]
pub struct Unsupported;

/// Processes a block `PIXELS` wide, 2 high.
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
    /// * `top_uyvy_addr` and `bot_uyvy_addr` contain `2 * PIXELS` bytes of initialized data.
    /// * `top_y_addr` and `bot_y_addr` are valid destinations for `PIXELS` bytes.
    /// * `u_addr` and `v_addr` are valid destinations for `PIXELS / 2` bytes.
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
/// `data` is of the shape `(height, width / 2, 4)` and channels are (U, Y, V, Y).
///
/// Returns planes `Y` of the shape `(height, width)`, `U` of the shape `(height / 2, width / 2)`,
/// and `V` of the shape `(height / 2, width / 2)`.
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
                p.process(
                    uyvy_in.as_ptr().add(r * uyvy_row_stride + 2 * c),
                    uyvy_in.as_ptr().add((r + 1) * uyvy_row_stride + 2 * c),
                    y_out.as_mut_ptr().cast::<u8>().add(r * width + c),
                    y_out.as_mut_ptr().cast::<u8>().add((r + 1) * width + c),
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

/// AVX2 block processor.
///
/// The following references are helpful in writing and understanding `x86_64`
/// SIMD code:
///
/// *   The [Intel Instrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
///     describes the purpose of these intrinsics.
/// *   [uops.info](https://uops.info/) gives timing information on a variety of Intel and AMD
///     chips based on both documentation and measurements. The Zen 2-focused notes below came from
///     this source.
///
/// A major complication of AVX2 code is that the 256-bit registers are divided
/// into separate 128-bit "lower" and "upper" lanes. There are many operations
/// that shuffle within these lanes but few that cross them. This leads to
/// arcane sequences of intrinsics.
///
/// Note that while LLVM doesn't autovectorize a whole algorithm well (including using specialty
/// instrinsics), it is good at shuffling. It's quite helpful to use the
/// [Godbolt compiler explorer](https://godbolt.org/) to see how it would do a
/// particular shuffle operation, phrased as either a short C++ program using
/// `__builtin_shufflevector` or a short Rust program using
/// `std::simd::simd_swizzle!` (which is unstable anyway as of 2024-04-22).
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
            01, 03, 05, 07, 09, 11, 13, 15, // lower half: 8 Y components.
            00, 04, 08, 12, 02, 06, 10, 14, // upper half: (4 * U), (4 * V).
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

/// Explicit NEON implementation.
///
/// This is refreshingly simple compared to the AVX2 implementation above; NEON
/// is not as flexible but makes it quite easy to unzip `(u, y, v, y)``
/// sequences.  One performance note is that [Godbolt compiler
/// explorer](https://godbolt.org/) appears to show an extra vector register
/// move between operations per iteration that is likely costing some
/// performance. Inline assembly might do better.
///
/// The best approach on `aarch64` may not be on-CPU anyway. Apple's Video
/// Toolbox and Rockchip MPP both support pixel format conversions, likely via
/// dedicated video hardware.
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
        aarch64::vst1q_u8(u_addr, aarch64::vhaddq_u8(top_uyvy.0, bot_uyvy.0));
        aarch64::vst1q_u8(v_addr, aarch64::vhaddq_u8(top_uyvy.2, bot_uyvy.2));
    }
}

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
                for (uyvy_addr, y_addr) in [(top_uyvy_addr, top_y_addr), (bot_uyvy_addr, bot_y_addr)] {
                    for i in 0..PIXELS {
                        std::ptr::write(y_addr.add(i), std::ptr::read(uyvy_addr.add(2*i + 1)))
                    }
                }
                let avg = |a: u8, b: u8| { (u16::from(a) + u16::from(b) >> 1) as u8 };
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
    features { "avx" }
}

#[cfg(target_arch = "aarch64")]
auto! {
    AutoNeonBlock
    supported { std::arch::is_aarch64_feature_detected!("neon") }
    features { "neon" }
}

auto! { AutoVanillaBlock supported { true } features { } }

#[cfg(test)]
mod tests {
    macro_rules! test_block {
        ($processor: ty, $mod: ident) => {
            mod $mod {
                use super::super::BlockProcessor as _;
                type P = $processor;

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

                #[test]
                fn uv() {
                    let p = P::new().unwrap();
                    let mut top_in = vec![0xff; P::PIXELS * 4];
                    let mut bot_in = vec![0xff; P::PIXELS * 4];
                    for i in 0..P::PIXELS {
                        // u values avg to 0x20 + i.
                        top_in[4 * i] = 0x10 + i as u8;
                        bot_in[4 * i] = 0x30 + i as u8;

                        // v values avg to 0x50 + i.
                        top_in[4 * i + 2] = 0x40 + i as u8;
                        bot_in[4 * i + 2] = 0x60 + i as u8;
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
                        std::array::from_fn(|i| 0x50 + i as u8);
                    assert_eq!(&u_out[..], &expected_u_out[..]);
                    assert_eq!(&v_out[..], &expected_v_out[..]);
                }
            }
        };
    }

    #[cfg(target_arch = "x86_64")]
    test_block!(super::super::ExplicitAvx2DoubleBlock, explicit_double_avx2);

    #[cfg(target_arch = "x86_64")]
    test_block!(super::super::AutoAvx2Block<64>, auto_avx2);

    #[cfg(target_arch = "aarch64")]
    test_block!(super::super::AutoNeonBlock<64>, auto_avx2);

    #[cfg(target_arch = "aarch64")]
    test_block!(super::super::ExplicitNeon, explicit_neon);

    test_block!(super::super::AutoVanillaBlock<64>, auto);

    #[test]
    fn full_frame() {
        // Created with:
        // ```
        // ffmpeg -y -f lavfi -i testsrc=size=1280x720,format=uyvy422 -frames 1 in.yuv
        // ffmpeg -y -pix_fmt uyvy422 -s 1280x720 -i in.yuv -pix_fmt yuv420p out.yuv
        // ```
        const WIDTH: usize = 1280;
        const HEIGHT: usize = 720;
        let uyvy_in = include_bytes!("testdata/in.yuv");
        let expected_out = include_bytes!("testdata/out.yuv");
        let mut actual_out = Vec::with_capacity(3 * WIDTH * HEIGHT / 2);
        let (mut y_out, uv_out) = actual_out.spare_capacity_mut().split_at_mut(WIDTH * HEIGHT);
        let (mut u_out, mut v_out) = uv_out.split_at_mut(WIDTH * HEIGHT / 4);
        super::convert(uyvy_in, WIDTH, HEIGHT, &mut y_out, &mut u_out, &mut v_out).unwrap();
        unsafe {
            actual_out.set_len(3 * WIDTH * HEIGHT / 2);
        }

        // `assert_eq!` output is unhelpful on these large binary arrays.
        // On failure, it might be better to write to a file and diff with better tools.
        // std::fs::write("actual_out.yuv", &actual_out[..]).unwrap();
        assert!(&expected_out[..] == &actual_out[..]);
    }
}
