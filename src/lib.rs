// Copyright (C) 2024 Infinite Athlete, Inc. <av-eng@infiniteathlete.ai>
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pixel format conversions.

#[doc(hidden)] // `pub` only for benchmarks.
pub mod uyvy_to_i420;

pub mod frame;

pub use uyvy_to_i420::convert as convert_uyvy_to_i420;

/// Re-export of the `arrayvec` version used by this crate.
///
/// [`arrayvec::ArrayVec`] is exposed in e.g. [`crate::frame::Frame::planes`], and callers may
/// wish to use matching types.
pub use arrayvec;

/// The maximum number of image planes defined by any supported [`PixelFormat`].
pub const MAX_PLANES: usize = 3;

/// Error type for pixel format conversions.
#[derive(Clone, Debug)]
pub struct ConversionError(&'static str);

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for ConversionError {}

/// Pixel format: layout of pixels in memory, defining the number/meaning
/// of planes including the size of each sample in bits.
///
/// YUV color ranges (e.g. full/JPEG vs limited/MPEG) are not defined here.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// [UYVY](https://fourcc.org/pixel-format/yuv-uyvy/).
    ///
    /// Matches ffmpeg's `AV_PIX_FMT_UYVY422`: packed YUV 4:2:2, 16bpp, Cb Y0 Cr Y1.
    UYVY422,

    /// [I420](https://fourcc.org/pixel-format/yuv-i420/).
    ///
    /// Matches ffmpeg's `AV_PIX_FMT_YUV420P`: planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples).
    I420,
}

/// Dimensions of a particular image plane.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct PlaneDims {
    /// The stride for a row, in bytes. This may include extra padding.
    pub stride: usize,

    /// The number of rows. This often matches the image height, but some
    /// chroma planes may be subsampled.
    pub rows: usize,
}

impl PixelFormat {
    /// Returns the number of planes for this format.
    #[inline]
    pub fn num_planes(self) -> usize {
        match self {
            PixelFormat::UYVY422 => 1,
            PixelFormat::I420 => 3,
        }
    }

    /// Returns the plane dimensions at minimum stride (no extra bytes for padding).
    pub fn min_plane_dims(self, width: usize, height: usize) -> impl Iterator<Item = PlaneDims> {
        let mut sizes = arrayvec::ArrayVec::<PlaneDims, MAX_PLANES>::new();
        match self {
            PixelFormat::UYVY422 => {
                sizes.push(PlaneDims {
                    stride: width.checked_shl(1).expect("stride should not overflow"),
                    rows: height,
                });
            }
            PixelFormat::I420 => {
                sizes.push(PlaneDims {
                    // Y plane.
                    stride: width,
                    rows: height,
                });
                let chroma_plane_size = PlaneDims {
                    // U/V planes.
                    // Overflow-safe divide by two that rounds up.
                    stride: (width >> 1) + (width & 1),
                    rows: (height >> 1) + (height & 1),
                };
                sizes.push(chroma_plane_size);
                sizes.push(chroma_plane_size);
            }
        }
        debug_assert_eq!(sizes.len(), self.num_planes());
        sizes.into_iter()
    }

    /// Returns human-readable names of the planes for this format.
    pub fn plane_names(self) -> &'static [&'static str] {
        match self {
            PixelFormat::UYVY422 => &["YUYV"],
            PixelFormat::I420 => &["Y", "U", "V"],
        }
    }
}
