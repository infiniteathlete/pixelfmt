// Copyright (C) 2024 Infinite Athlete, Inc. <av-eng@infiniteathlete.ai>
// SPDX-License-Identifier: MIT OR Apache-2.0

use core::panic;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use pixelfmt::{
    frame::{ConsecutiveFrame, Frame, FrameBuf, VecFrameBuf},
    PixelFormat,
};

/// Reads and writes about the right amount of stuff with `memcpy`.
///
/// This produces nonsense; it's useful only as a memory bandwidth baseline.
fn memcpy_baseline<FI: Frame, BO: FrameBuf>(input: &FI, mut output: BO) -> BO::Frame {
    let input_planes = input.planes();
    let [input] = &input_planes[..] else {
        panic!("expected exactly one plane");
    };
    let mut output_planes = output.planes();
    let [y_out, u_out, v_out] = &mut output_planes[..] else {
        panic!("expected exactly three planes");
    };
    let (y_in, rest) = input.data.split_at(input.data.len() / 2);
    let (u_in_all, v_in_all) = rest.split_at(rest.len() / 2);
    let (u_in_1, u_in_2) = u_in_all.split_at(u_in_all.len() / 2);
    let (v_in_1, v_in_2) = v_in_all.split_at(v_in_all.len() / 2);
    assert_eq!(y_out.data.len(), y_in.len());
    assert_eq!(u_out.data.len(), u_in_1.len());
    assert_eq!(u_out.data.len(), u_in_2.len());
    assert_eq!(v_out.data.len(), v_in_1.len());
    assert_eq!(v_out.data.len(), v_in_2.len());

    // TODO: it'd be nice to use `MaybeUninit::copy_from_slice` and
    // `MaybeUninit::slice_as_mut`, but they're unstable.
    // https://github.com/rust-lang/rust/issues/79995
    unsafe {
        std::ptr::copy_nonoverlapping(y_in.as_ptr(), y_out.data.as_mut_ptr().cast(), y_in.len());
        std::ptr::copy_nonoverlapping(
            u_in_1.as_ptr(),
            u_out.data.as_mut_ptr().cast(),
            u_in_1.len(),
        );
        std::ptr::copy_nonoverlapping(
            u_in_2.as_ptr(),
            u_out.data.as_mut_ptr().cast(),
            u_in_2.len(),
        );
        std::ptr::copy_nonoverlapping(
            v_in_1.as_ptr(),
            v_out.data.as_mut_ptr().cast(),
            v_in_1.len(),
        );
        std::ptr::copy_nonoverlapping(
            v_in_2.as_ptr(),
            v_out.data.as_mut_ptr().cast(),
            v_in_2.len(),
        );
        drop(output_planes);
        output.finish()
    }
}

fn libyuv<FI: Frame, BO: FrameBuf>(input: &FI, mut output: BO) -> BO::Frame {
    let input_planes = input.planes();
    let (width, height) = input.pixel_dimensions();
    let [uyvy] = &input_planes[..] else {
        panic!("expected exactly one plane");
    };
    assert_eq!(uyvy.data.len(), width * height * 2);
    let mut output_planes = output.planes();
    let [y_out, u_out, v_out] = &mut output_planes[..] else {
        panic!("expected exactly three planes");
    };
    assert_eq!(y_out.data.len(), width * height);
    assert_eq!(u_out.data.len(), width * height / 4);
    assert_eq!(v_out.data.len(), width * height / 4);
    unsafe {
        assert_eq!(
            0,
            yuv_sys::rs_UYVYToI420(
                uyvy.data.as_ptr(),
                (width * 2) as i32,
                y_out.data.as_mut_ptr().cast(),
                width as i32,
                u_out.data.as_mut_ptr().cast(),
                (width / 2) as i32,
                v_out.data.as_mut_ptr().cast(),
                (width / 2) as i32,
                width as i32,
                height as i32,
            )
        );
        drop(output_planes);
        output.finish()
    }
}

/// Common implementation between `bench_cold` and `bench_hot`.
fn bench_common<const FRAMES_PER_ITER: usize>(
    mut g: criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;
    let inputs: [_; FRAMES_PER_ITER] = std::array::from_fn(|_| {
        ConsecutiveFrame::new_unpadded(
            pixelfmt::PixelFormat::UYVY422,
            WIDTH,
            HEIGHT,
            // This dummy frame is filled with 1s rather than 0s so that it has
            // to occupy distinct physical memory rather than take advantage of
            // the zero page optimization on Linux. [1] Distinct physical memory
            // is important to stress the CPU cache as real images would.
            //
            // [1] https://lwn.net/Articles/517465/
            vec![1u8; WIDTH * HEIGHT * 2],
        )
    });
    g.throughput(criterion::Throughput::Bytes(
        (inputs.len() * (WIDTH * HEIGHT * 7) / 2) as u64,
    ));
    macro_rules! bench_block {
        ($name:literal, $impl:ty) => {
            g.bench_function($name, |b| {
                b.iter(|| {
                    for i in &inputs {
                        black_box(
                            convert_with::<$impl, _, _>(
                                i,
                                VecFrameBuf::new(PixelFormat::I420, WIDTH, HEIGHT, 0),
                            )
                            .unwrap(),
                        );
                    }
                })
            });
        };
    }
    use pixelfmt::uyvy_to_i420::*;
    g.bench_function("memcpy_baseline", |b| {
        b.iter(|| {
            for i in &inputs {
                black_box(memcpy_baseline(
                    i,
                    VecFrameBuf::new(PixelFormat::I420, WIDTH, HEIGHT, 0),
                ));
            }
        })
    });
    g.bench_function("libyuv", |b| {
        b.iter(|| {
            for i in &inputs {
                black_box(libyuv(
                    i,
                    VecFrameBuf::new(PixelFormat::I420, WIDTH, HEIGHT, 0),
                ));
            }
        })
    });
    #[cfg(target_arch = "x86_64")]
    bench_block!("explicit_avx2_double", ExplicitAvx2DoubleBlock);
    #[cfg(target_arch = "x86_64")]
    bench_block!("explicit_avx2_single", ExplicitAvx2SingleBlock);
    #[cfg(target_arch = "x86_64")]
    bench_block!("auto_avx2_64", AutoAvx2Block<64>);
    #[cfg(target_arch = "aarch64")]
    bench_block!("explicit_neon", ExplicitNeon);
    #[cfg(target_arch = "aarch64")]
    bench_block!("auto_neon_64", AutoNeonBlock<64>);
    bench_block!("auto_vanilla_64", AutoVanillaBlock<64>);
    g.finish();
}

/// Cold benchmark: each iteration processes enough data to be unlikely to fit in cache.
fn bench_cold(c: &mut Criterion) {
    bench_common::<32>(c.benchmark_group("cold"));
}

/// Hot benchmark: each iteration operates on a single frame that likely fits in the CPU's LLC.
fn bench_hot(c: &mut Criterion) {
    bench_common::<1>(c.benchmark_group("hot"));
}

criterion_group!(benches, bench_cold, bench_hot);

criterion_main!(benches);
