// Copyright (C) 2024 Infinite Athlete, Inc. <av-eng@infiniteathlete.ai>
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Image frame abstractions.

use std::{mem::MaybeUninit, ops::Deref};

use arrayvec::ArrayVec;

use crate::{PixelFormat, PlaneDims, MAX_PLANES};

/// A fully initialized image frame.
pub trait Frame {
    /// Returns the pixel format of the image.
    fn format(&self) -> PixelFormat;

    /// Returns the `(width, height)` of the image.
    fn pixel_dimensions(&self) -> (usize, usize);

    /// Returns the (image format-defined) planes.
    fn planes(&self) -> ArrayVec<FramePlane, MAX_PLANES>;
}

#[derive(Eq, PartialEq)] // TODO: should `Eq`/`PartialEq` ignore padding bytes?
pub struct FramePlane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
}

/// A buffer in which an image frame can be placed.
///
/// # Safety
///
/// Implementors must return the same buffer for each call to `planes`, so that
/// callers can meaningfully fill the buffer as required by `finish`.
pub unsafe trait FrameBuf {
    type Frame: Frame;

    /// Returns the pixel format of the image.
    fn format(&self) -> PixelFormat;

    /// Returns the `(width, height)` of the image.
    fn pixel_dimensions(&self) -> (usize, usize);

    /// Returns the (pixel format-defined) planes.
    fn planes(&mut self) -> ArrayVec<FrameBufPlane, MAX_PLANES>;

    /// Returns a fully initialized frame from this buffer.
    ///
    /// # Safety
    ///
    /// Caller asserts that it has filled in all bytes of the buffers returned
    /// by `planes`, including padding.
    unsafe fn finish(self) -> Self::Frame;
}

pub struct FrameBufPlane<'a> {
    pub data: &'a mut [MaybeUninit<u8>],
    pub stride: usize,
}

/// A frame buffer which stores all planes consecutively in a single `Vec`.
pub struct VecFrameBuf {
    format: PixelFormat,
    width: usize,
    height: usize,

    /// Planes' dimensions.
    /// * planes beyond those required by the format have `stride == rows == 0`.
    /// * `stride` is sufficient for `width` pixels; may have extra padding.
    /// * `rows` is correct for `height` with no extra padding.
    /// * `stride * rows` summmed across all planes does not overflow.
    dims: [PlaneDims; MAX_PLANES],

    /// A `Vec` with length 0 and sufficient capacity for all planes.
    data: Vec<u8>,
}

impl VecFrameBuf {
    /// Creates a new frame buffer.
    ///
    /// Pads each plane's row stride in bytes to be a multiple of the next
    /// multiple of two greater than or equal to `padding`. Panics on overflow.
    pub fn new(format: PixelFormat, width: usize, height: usize, padding: usize) -> Self {
        let mut total_bytes = 0usize;
        let mut dims = [PlaneDims::default(); MAX_PLANES];
        let padding_mask = padding
            .checked_next_power_of_two()
            .expect("padding should not overflow")
            - 1;
        for (dims, min) in dims.iter_mut().zip(format.min_plane_dims(width, height)) {
            dims.stride = min
                .stride
                .checked_add(padding_mask)
                .expect("plane stride with padding should not overflow")
                & !padding_mask;
            dims.rows = min.rows;
            let plane_bytes = dims
                .stride
                .checked_mul(dims.rows)
                .expect("plane size should not overflow");
            total_bytes = total_bytes
                .checked_add(plane_bytes)
                .expect("total frame size should not overflow");
        }
        let data = Vec::with_capacity(total_bytes);
        Self {
            format,
            width,
            height,
            dims,
            data,
        }
    }
}

unsafe impl FrameBuf for VecFrameBuf {
    type Frame = ConsecutiveFrame<Vec<u8>>;

    #[inline]
    fn format(&self) -> PixelFormat {
        self.format
    }

    #[inline]
    fn pixel_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn planes(&mut self) -> ArrayVec<FrameBufPlane, MAX_PLANES> {
        let mut remaining = self.data.spare_capacity_mut();
        let mut planes = ArrayVec::new();
        for dims in self.dims.iter().take(self.format.num_planes()) {
            let (head, tail) = remaining.split_at_mut(dims.stride * dims.rows);
            planes.push(FrameBufPlane {
                data: head,
                stride: dims.stride,
            });
            remaining = tail;
        }
        planes
    }

    #[inline]
    unsafe fn finish(mut self) -> Self::Frame {
        // SAFETY: `self.data` has sufficient size, and the caller has
        // promised the spare bytes have been initialized.
        unsafe {
            self.data
                .set_len(self.dims.iter().map(|d| d.stride * d.rows).sum());
        }
        ConsecutiveFrame {
            format: self.format,
            width: self.width,
            height: self.height,
            dims: self.dims,
            data: self.data,
        }
    }
}

impl From<ConsecutiveFrame<Vec<u8>>> for VecFrameBuf {
    #[inline]
    fn from(mut value: ConsecutiveFrame<Vec<u8>>) -> Self {
        value.data.clear();
        VecFrameBuf {
            format: value.format,
            width: value.width,
            height: value.height,
            dims: value.dims,
            data: value.data,
        }
    }
}

/// A frame in which all planes are stored consecutively.
pub struct ConsecutiveFrame<T> {
    format: PixelFormat,
    width: usize,
    height: usize,

    /// Planes' dimensions. Invariants are the same as for `VecFrameBuf`.
    dims: [PlaneDims; MAX_PLANES],

    /// Data; must be exactly the right size for the planes.
    data: T,
}

impl<T: Deref<Target = [u8]>> ConsecutiveFrame<T> {
    /// Returns a new consecutive frame with the given dimensions.
    ///
    /// This is the most general form that allows the caller to specify padding.
    /// Panics if the arguments are inconsistent.
    pub fn new(
        format: PixelFormat,
        width: usize,
        height: usize,
        dims: &[PlaneDims],
        data: T,
    ) -> Self {
        assert_eq!(format.num_planes(), dims.len());
        let mut expected_len = 0usize;
        for (dims, min) in dims.iter().zip(format.min_plane_dims(width, height)) {
            assert!(dims.stride >= min.stride);
            assert!(dims.rows == min.rows);
            let plane_len = dims
                .stride
                .checked_mul(dims.rows)
                .expect("plane size should not overflow");
            expected_len = expected_len
                .checked_add(plane_len)
                .expect("total frame size should not overflow");
        }
        assert_eq!(data.len(), expected_len);
        Self {
            format,
            width,
            height,
            dims: std::array::from_fn(|i| dims.get(i).copied().unwrap_or_default()),
            data,
        }
    }

    /// Returns a new consecutive frame with no padding.
    ///
    /// Panics if the arguments are inconsistent.
    pub fn new_unpadded(format: PixelFormat, width: usize, height: usize, data: T) -> Self {
        let mut expected_len = 0usize;
        let mut dims = [PlaneDims::default(); MAX_PLANES];
        for (i, min) in format.min_plane_dims(width, height).enumerate() {
            dims[i] = min;
            let plane_len = min
                .stride
                .checked_mul(min.rows)
                .expect("plane size should not overflow");
            expected_len = expected_len
                .checked_add(plane_len)
                .expect("total frame size should not overflow");
        }
        assert_eq!(data.len(), expected_len);
        Self {
            format,
            width,
            height,
            dims,
            data,
        }
    }

    /// Returns a reference to this (potentially owned) frame.
    ///
    /// This can be useful to avoid monomorphization overhead or simply to
    /// avoid handing off ownership.
    #[inline]
    pub fn as_ref(&self) -> ConsecutiveFrame<&[u8]> {
        ConsecutiveFrame {
            format: self.format,
            width: self.width,
            height: self.height,
            dims: self.dims,
            data: &self.data,
        }
    }

    /// Returns the underlying storage for the planes.
    #[inline]
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Consumes the frame and returns the underlying storage for the planes.
    #[inline]
    pub fn into_data(self) -> T {
        self.data
    }
}

impl<T: Deref<Target = [u8]>> Frame for ConsecutiveFrame<T> {
    #[inline]
    fn format(&self) -> PixelFormat {
        self.format
    }

    #[inline]
    fn pixel_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn planes(&self) -> ArrayVec<FramePlane, MAX_PLANES> {
        let mut remaining = &self.data[..];
        let mut planes = ArrayVec::new();
        for dims in self.dims.iter().take(self.format.num_planes()) {
            let (head, tail) = remaining.split_at(dims.stride * dims.rows);
            planes.push(FramePlane {
                data: head,
                stride: dims.stride,
            });
            remaining = tail;
        }
        planes
    }
}

impl<T: Deref<Target = [u8]>> PartialEq for ConsecutiveFrame<T> {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format
            && self.width == other.width
            && self.height == other.height
            && self.planes() == other.planes()
    }
}

impl<T: Deref<Target = [u8]>> Eq for ConsecutiveFrame<T> {}
