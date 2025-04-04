// Copyright (C) 2024 Infinite Athlete, Inc. <av-eng@infiniteathlete.ai>
// SPDX-License-Identifier: MIT OR Apache-2.0

use core::panic;

use criterion::{criterion_group, criterion_main, Criterion};
use pixelfmt::frame::{ConsecutiveFrame, Frame, FrameMut};

/// Reads and writes about the right amount of stuff with `memcpy`.
///
/// This produces nonsense; it's useful only as a memory bandwidth baseline.
fn memcpy_baseline(input: &ConsecutiveFrame<Vec<u8>>, output: &mut ConsecutiveFrame<Vec<u8>>) {
    let input_planes = input.planes();
    let [input] = &input_planes[..] else {
        panic!("expected exactly one plane");
    };
    let mut output_planes = output.planes_mut();
    let [y_out, u_out, v_out] = &mut output_planes[..] else {
        panic!("expected exactly three planes");
    };
    let (y_in, rest) = input.split_at(input.len() / 2);
    let (u_in_all, v_in_all) = rest.split_at(rest.len() / 2);
    let (u_in_1, u_in_2) = u_in_all.split_at(u_in_all.len() / 2);
    let (v_in_1, v_in_2) = v_in_all.split_at(v_in_all.len() / 2);
    y_out.copy_from_slice(y_in);
    u_out.copy_from_slice(u_in_1);
    u_out.copy_from_slice(u_in_2);
    v_out.copy_from_slice(v_in_1);
    v_out.copy_from_slice(v_in_2);
}

fn libyuv<FI: Frame, FO: FrameMut>(input: &FI, output: &mut FO) {
    let input_planes = input.planes();
    let (width, height) = input.pixel_dimensions();
    let [uyvy] = &input_planes[..] else {
        panic!("expected exactly one plane");
    };
    assert_eq!(uyvy.len(), width * height * 2);
    assert_eq!(uyvy.stride(), width * 2);
    let mut output_planes = output.planes_mut();
    let [y_out, u_out, v_out] = &mut output_planes[..] else {
        panic!("expected exactly three planes");
    };
    assert_eq!(y_out.len(), width * height);
    assert_eq!(y_out.stride(), width);
    assert_eq!(u_out.len(), width * height / 4);
    assert_eq!(u_out.stride(), width / 2);
    assert_eq!(v_out.len(), width * height / 4);
    assert_eq!(v_out.stride(), width / 2);
    unsafe {
        assert_eq!(
            0,
            yuv_sys::rs_UYVYToI420(
                uyvy.as_ptr(),
                uyvy.stride() as i32,
                y_out.as_mut_ptr(),
                y_out.stride() as i32,
                u_out.as_mut_ptr(),
                u_out.stride() as i32,
                v_out.as_mut_ptr(),
                v_out.stride() as i32,
                width as i32,
                height as i32,
            )
        );
        drop(output_planes);
    }
}

/// Common implementation between `bench_cold` and `bench_hot`.
fn bench_common<const FRAMES_PER_ITER: usize>(
    mut g: criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;
    struct Operation {
        input: ConsecutiveFrame<Vec<u8>>,
        output: ConsecutiveFrame<Vec<u8>>,
    }
    let mut ops: [Operation; FRAMES_PER_ITER] = std::array::from_fn(|_| Operation {
        input: ConsecutiveFrame::new(pixelfmt::PixelFormat::UYVY422, WIDTH, HEIGHT).with_storage(
            // This dummy frame is filled with 1s rather than 0s so that it has
            // to occupy distinct physical memory rather than take advantage of
            // the zero page optimization on Linux. [1] Distinct physical memory
            // is important to stress the CPU cache as real images would.
            //
            // [1] https://lwn.net/Articles/517465/
            vec![1u8; WIDTH * HEIGHT * 2],
        ),
        output: ConsecutiveFrame::new(pixelfmt::PixelFormat::I420, WIDTH, HEIGHT).new_vec(),
    });
    g.throughput(criterion::Throughput::Bytes(
        ((WIDTH * HEIGHT * 7) / 2) as u64,
    ));
    macro_rules! bench {
        ($name:literal, $f:expr) => {
            g.bench_function($name, |b| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        for op in &mut ops {
                            $f(op);
                        }
                    }
                    start.elapsed() / FRAMES_PER_ITER as u32
                });
            });
        };
    }
    macro_rules! bench_block {
        ($name:literal, $p:expr) => {
            let p = $p;
            bench!($name, |op: &mut Operation| convert_with(
                p,
                &op.input,
                &mut op.output
            )
            .unwrap());
        };
    }
    use pixelfmt::uyvy_to_i420::*;
    bench!("memcpy_baseline", |op: &mut Operation| memcpy_baseline(
        &op.input,
        &mut op.output
    ));
    bench!("libyuv", |op: &mut Operation| libyuv(
        &op.input,
        &mut op.output
    ));
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        bench_block!(
            "explicit_avx2_double",
            ExplicitAvx2DoubleBlock::try_new().unwrap()
        );
        bench_block!(
            "explicit_avx2_single",
            ExplicitAvx2SingleBlock::try_new().unwrap()
        );
        bench_block!("auto_avx2_64", AutoAvx2Block::<64>::try_new().unwrap());
    }
    #[cfg(target_arch = "x86_64")]
    bench_block!("explicit_sse2", ExplicitSse2::new());
    #[cfg(target_arch = "aarch64")]
    bench_block!("explicit_neon", ExplicitNeon::try_new().unwrap());
    #[cfg(target_arch = "aarch64")]
    bench_block!("auto_neon_64", AutoNeonBlock::<64>::try_new().unwrap());
    bench_block!(
        "auto_vanilla_64",
        AutoVanillaBlock::<64>::try_new().unwrap()
    );
    g.finish();
}

/// Cold benchmark: each iteration processes enough data to be unlikely to fit in cache.
fn bench_cold(c: &mut Criterion) {
    bench_common::<128>(c.benchmark_group("cold"));
}

/// Hot benchmark: each iteration operates on a single frame that likely fits in the CPU's LLC.
fn bench_hot(c: &mut Criterion) {
    bench_common::<1>(c.benchmark_group("hot"));
}

criterion_group!(benches, bench_cold, bench_hot);

criterion_main!(benches);
