// Copyright (C) 2024 Infinite Athlete <av-eng@infiniteathlete.ai>
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

/// Reads and writes about the right amount of stuff with `memcpy`.
///
/// This produces nonsense; it's useful only as a memory bandwidth baseline.
fn memcpy_baseline(input: &[u8]) -> [Vec<u8>; 3] {
    let (y, rest) = input.split_at(input.len() / 2);
    let (u_all, v_all) = rest.split_at(rest.len() / 2);
    let (u1, u2) = u_all.split_at(u_all.len() / 2);
    let (v1, v2) = v_all.split_at(v_all.len() / 2);
    let y = y.to_owned();
    let mut u = u1.to_owned();
    u.copy_from_slice(u2); // overwrite to simulate the average operation.
    let mut v = v1.to_owned();
    v.copy_from_slice(v2); // overwrite to simulate the average operation.
    assert_eq!(y.len(), input.len() / 2);
    assert_eq!(u.len(), input.len() / 8);
    assert_eq!(v.len(), input.len() / 8);
    [y, u, v]
}

fn libyuv(input: &[u8], width: usize, height: usize) -> [Vec<u8>; 3] {
    let mut y = Vec::with_capacity(width * height);
    let mut u = Vec::with_capacity(width * height / 4);
    let mut v = Vec::with_capacity(width * height / 4);
    unsafe {
        yuv_sys::rs_UYVYToI420(
            input.as_ptr(),
            (width * 2) as i32,
            y.as_mut_ptr(),
            width as i32,
            u.as_mut_ptr(),
            (width / 2) as i32,
            v.as_mut_ptr(),
            (width / 2) as i32,
            width as i32,
            height as i32,
        );
        y.set_len(y.capacity());
        u.set_len(u.capacity());
        v.set_len(v.capacity());
    }
    [y, u, v]
}

/// Common implementation between `bench_cold` and `bench_hot`.
fn bench_common<const FRAMES_PER_ITER: usize>(
    mut g: criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;
    let inputs: [_; FRAMES_PER_ITER] = std::array::from_fn(|_| vec![1u8; WIDTH * HEIGHT * 2]);
    g.throughput(criterion::Throughput::Bytes(
        (inputs.len() * (WIDTH * HEIGHT * 7) / 2) as u64,
    ));
    macro_rules! bench_block {
        ($name:literal, $impl:ty) => {
            g.bench_function($name, |b| {
                b.iter(|| {
                    for i in &inputs {
                        let mut y = Vec::with_capacity(WIDTH * HEIGHT);
                        let mut u = Vec::with_capacity(WIDTH * HEIGHT / 4);
                        let mut v = Vec::with_capacity(WIDTH * HEIGHT / 4);
                        black_box(
                            convert_with::<$impl>(
                                i,
                                WIDTH,
                                HEIGHT,
                                y.spare_capacity_mut(),
                                u.spare_capacity_mut(),
                                v.spare_capacity_mut(),
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
                black_box(memcpy_baseline(i));
            }
        })
    });
    g.bench_function("libyuv", |b| {
        b.iter(|| {
            for i in &inputs {
                black_box(libyuv(i, WIDTH, HEIGHT));
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
