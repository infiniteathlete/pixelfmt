// Copyright (C) 2024 Infinite Athlete, Inc. <av-eng@infiniteathlete.ai>
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Image frame abstractions.
//!
//! ```
//! # use pixelfmt::{PixelFormat, frame::{ConsecutiveFrame, Frame as _}};
//! // References a 1920x1080 I420 frame held in a slice.
//! let data = &[0u8; 1920 * 1080 * 3 / 2];
//! let frame = ConsecutiveFrame::new(PixelFormat::I420, 1920, 1080).with_storage(&data[..]);
//!
//! // Allocates a frame which can be used to store a 1920x1080 I420 image.
//! let mut out = ConsecutiveFrame::new(PixelFormat::I420, 1920, 1080).new_vec();
//! ```

use std::ops::{Deref, DerefMut};

use arrayvec::ArrayVec;

use crate::{PixelFormat, PlaneDims, MAX_PLANES};

/// Read access to a raw image frame.
///
/// Mutation is via the separate [`FrameMut`] trait.
pub trait Frame {
    /// Returns the pixel format of the image.
    fn format(&self) -> PixelFormat;

    /// Returns the `(width, height)` of the image.
    fn pixel_dimensions(&self) -> (usize, usize);

    /// Returns the (image format-defined) planes for read/shared access.
    fn planes(&self) -> ArrayVec<FramePlaneRef, MAX_PLANES>;
}

/// Raw image frame (write access).
///
/// See [`Frame`] for more information.
pub trait FrameMut: Frame {
    /// Returns the (image format-defined) planes for mutation/exclusive access.
    fn planes_mut(&mut self) -> ArrayVec<FramePlaneMut, MAX_PLANES>;
}

/// Provides read-only access to a given image plane.
pub struct FramePlaneRef<'a> {
    data: &'a [u8],
    stride: usize,
}

impl<'p> FramePlaneRef<'p> {
    /// Creates a new `FramePlaneRef`.
    #[inline]
    pub fn new(data: &'p mut [u8], stride: usize) -> Self {
        Self { data, stride }
    }

    /// Returns the stride of the plane in bytes.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the data as a slice.
    ///
    /// Unlike `Deref`, the returned slice can outlive the `FramePlaneRef`.
    #[inline]
    pub fn as_slice(&self) -> &'p [u8] {
        self.data
    }
}

impl Deref for FramePlaneRef<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl PartialEq for FramePlaneRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.stride == other.stride && self.data == other.data
    }
}

impl Eq for FramePlaneRef<'_> {}

/// Provides write access to a given image plane.
pub struct FramePlaneMut<'a> {
    data: &'a mut [u8],
    stride: usize,
}

impl<'a> FramePlaneMut<'a> {
    /// Creates a new `FramePlaneMut`.
    #[inline]
    pub fn new(data: &'a mut [u8], stride: usize) -> Self {
        Self { data, stride }
    }

    /// Returns the stride of the plane in bytes.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the data as a mutable slice.
    ///
    /// Note: unlike `DerefMut`, the returned slice can outlive the `FramePlaneMut`.
    #[inline]
    pub fn into_slice(self) -> &'a mut [u8] {
        self.data
    }
}

impl Deref for FramePlaneMut<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl DerefMut for FramePlaneMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl PartialEq for FramePlaneMut<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.stride == other.stride && self.data == other.data
    }
}

impl Eq for FramePlaneMut<'_> {}

/// A frame which stores all planes consecutively.
#[derive(Clone)]
pub struct ConsecutiveFrame<S> {
    format: PixelFormat,
    width: usize,
    height: usize,

    /// Planes' dimensions. Invariants:
    /// * planes beyond those required by the format have `stride == rows == 0`.
    /// * `stride` is sufficient for `width` pixels; may have extra padding.
    /// * `rows` is correct for `height` with no extra padding.
    /// * if `storage` is not `()`, `stride * rows` summed across all planes
    ///   does not overflow. (This invariant is not checked during the builder
    ///   phase.)
    dims: [PlaneDims; MAX_PLANES],

    /// A [`Storage`] with sufficient capacity for `Self::total_size`, or `()`.
    storage: S,
}

impl<S> ConsecutiveFrame<S> {
    fn total_size(&self) -> usize {
        let mut total_size = 0usize;
        for dims in self.dims.iter() {
            total_size = total_size
                .checked_add(dims.stride * dims.rows)
                .expect("total frame size should not overflow");
        }
        total_size
    }
}

/// Operations for building a `ConsecutiveFrame`.
impl ConsecutiveFrame<()> {
    /// Returns a new builder for a consecutive frame with the given dimensions.
    ///
    /// To obtain a useful `Frame` impl, call `new_vec` or `with_storage`.
    pub fn new(format: PixelFormat, width: usize, height: usize) -> Self {
        let mut dims = [PlaneDims::default(); MAX_PLANES];
        for (dim, min) in dims.iter_mut().zip(format.min_plane_dims(width, height)) {
            *dim = min;
        }
        ConsecutiveFrame {
            format,
            width,
            height,
            dims,
            storage: (),
        }
    }

    /// Pads each plane's row stride in bytes to be a multiple of the next
    /// multiple of two greater than or equal to `padding`. Panics on overflow.
    pub fn with_padding(mut self, padding: usize) -> Self {
        let padding_mask = padding
            .checked_next_power_of_two()
            .expect("padding should not overflow")
            - 1;
        for dims in self.dims.iter_mut() {
            dims.stride = dims
                .stride
                .checked_add(padding_mask)
                .expect("plane stride with padding should not overflow")
                & !padding_mask;
        }
        self
    }

    /// Returns an uninitialized frame backed by a newly allocated `Vec<u8>`.
    ///
    /// Panics on overflow or allocation failure.
    pub fn new_vec(self) -> ConsecutiveFrame<Vec<u8>> {
        ConsecutiveFrame {
            format: self.format,
            width: self.width,
            height: self.height,
            dims: self.dims,
            storage: vec![0; self.total_size()],
        }
    }

    /// Returns a frame backend by `storage`.
    ///
    /// Panics on overflow or if `storage` is too small.
    pub fn with_storage<S: Deref<Target = [u8]>>(self, storage: S) -> ConsecutiveFrame<S> {
        assert!(
            storage.len() >= self.total_size(),
            "storage={} < total_size={}",
            storage.len(),
            self.total_size()
        );
        ConsecutiveFrame {
            format: self.format,
            width: self.width,
            height: self.height,
            dims: self.dims,
            storage,
        }
    }
}

impl<S> ConsecutiveFrame<S> {
    pub fn inner(&self) -> &S {
        &self.storage
    }

    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    pub fn into_inner(self) -> S {
        self.storage
    }
}

impl<S: Deref<Target = [u8]>> Frame for ConsecutiveFrame<S> {
    #[inline]
    fn format(&self) -> PixelFormat {
        self.format
    }

    #[inline]
    fn pixel_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn planes(&self) -> ArrayVec<FramePlaneRef, MAX_PLANES> {
        let mut storage = self.storage.deref();
        let mut planes = ArrayVec::new();
        for dims in self.dims.iter().take(self.format.num_planes()) {
            let (data, rest) = storage.split_at(dims.stride * dims.rows);
            storage = rest;
            planes.push(FramePlaneRef {
                data,
                stride: dims.stride,
            });
        }
        planes
    }
}

impl<S: DerefMut<Target = [u8]>> FrameMut for ConsecutiveFrame<S> {
    fn planes_mut(&mut self) -> ArrayVec<FramePlaneMut, MAX_PLANES> {
        let mut storage = self.storage.deref_mut();
        let mut planes = ArrayVec::new();
        for dims in self.dims.iter().take(self.format.num_planes()) {
            let (data, rest) = storage.split_at_mut(dims.stride * dims.rows);
            storage = rest;
            planes.push(FramePlaneMut {
                data,
                stride: dims.stride,
            });
        }
        planes
    }
}
