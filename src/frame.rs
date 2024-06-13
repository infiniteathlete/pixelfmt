// Copyright (C) 2024 Infinite Athlete, Inc. <av-eng@infiniteathlete.ai>
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Image frame abstractions.
//!
//! ```
//! # use pixelfmt::{PixelFormat, frame::{ConsecutiveFrame, Frame as _}};
//! // References a 1920x1080 I420 frame held in a slice.
//! let data = &[0u8; 1920 * 1080 * 3 / 2];
//! let frame = ConsecutiveFrame::new(PixelFormat::I420, 1920, 1080).with_storage(&data[..]);
//! assert!(frame.initialized());
//!
//! // Allocates a frame which can be used to store a 1920x1080 I420 image.
//! let mut out = ConsecutiveFrame::new(PixelFormat::I420, 1920, 1080).new_vec();
//! assert!(!out.initialized());
//! ```

use std::{marker::PhantomData, mem::MaybeUninit};

use arrayvec::ArrayVec;

use crate::{PixelFormat, PlaneDims, MAX_PLANES};

/// Read access to a raw image frame.
///
/// Mutation is via the separate [`FrameMut`] trait.
///
/// # Initialization
///
/// For efficiency reasons, frames may not be fully initialized when created.
/// The `Frame` is responsible for tracking its own initialization status and
/// will panic if asked to give a reference to data within any of its planes
/// prior to initialization.
///
/// Access to potentially uninitialized data requires raw pointers. In
/// particular, `FrameMut` does *not* provide access via
/// `&mut MaybeUninit<[u8]>` because copying from uninitialized memory to this
/// buffer would fill a `ConsecutiveFrame<&mut [u8]>` with uninitialized data,
/// which would be unsound.
///
/// # Safety
///
/// The implementor is responsible for the validity of the raw pointers returned
/// via `planes` and `planes_mut`.
pub unsafe trait Frame {
    /// Returns the pixel format of the image.
    fn format(&self) -> PixelFormat;

    /// Returns the `(width, height)` of the image.
    fn pixel_dimensions(&self) -> (usize, usize);

    /// Returns true if this frame has been fully initialized.
    fn initialized(&self) -> bool;

    /// Marks this frame as fully initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the frame is fully initialized, including
    /// any padding bytes.
    unsafe fn initialize(&mut self);

    /// Returns the (image format-defined) planes for read/shared access.
    fn planes(&self) -> ArrayVec<FramePlaneRef, MAX_PLANES>;
}

/// Raw image frame (write access).
///
/// See [`Frame`] for more information.
///
/// # Safety
///
/// As with `Frame`.
pub unsafe trait FrameMut: Frame {
    /// Returns the (image format-defined) planes for mutation/exclusive access.
    fn planes_mut(&mut self) -> ArrayVec<FramePlaneMut, MAX_PLANES>;
}

/// Provides read-only access to a given image plane.
pub struct FramePlaneRef<'a> {
    data: *const u8,
    stride: usize,
    len: usize,
    initialized: bool,
    _phantom: PhantomData<&'a [u8]>,
}

impl FramePlaneRef<'_> {
    /// Creates a new `FramePlaneRef`.
    ///
    /// # Safety
    ///
    /// The caller is responsible for the validity of all arguments and for
    /// bounding the returned lifetime.
    #[inline]
    pub unsafe fn new(data: *const u8, stride: usize, len: usize) -> Self {
        Self {
            data,
            stride,
            len,
            initialized: false,
            _phantom: PhantomData,
        }
    }

    /// Returns the stride of the plane in bytes.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the total length of the plane in bytes.
    #[allow(clippy::len_without_is_empty)] // empty frames are silly.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the plane's data as a slice.
    ///
    /// Panics if the frame has not been initialized.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        assert!(self.initialized);
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Returns a raw pointer to the plane's data, which may be uninitialized.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.data
    }
}

/// Tests for equality of two `FramePlaneRef`s; this will panic if either is uninitialized.
impl PartialEq for FramePlaneRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        // TODO: skip padding bytes?
        self.stride == other.stride && self.len == other.len && self.as_slice() == other.as_slice()
    }
}

impl Eq for FramePlaneRef<'_> {}

/// Provides write access to a given image plane.
pub struct FramePlaneMut<'a> {
    data: *mut u8,
    stride: usize,
    len: usize,

    /// See `FramePlaneRef::initialized`.
    initialized: bool,
    _phantom: PhantomData<&'a mut [u8]>,
}

impl FramePlaneMut<'_> {
    /// Creates a new `FramePlaneMut`.
    ///
    /// # Safety
    ///
    /// The caller is responsible for the validity of all arguments and for
    /// bounding the returned lifetime.
    #[inline]
    pub unsafe fn new(data: *mut u8, stride: usize, len: usize) -> Self {
        Self {
            data,
            stride,
            len,
            initialized: false,
            _phantom: PhantomData,
        }
    }

    /// Returns the stride of the plane in bytes.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the total length of the plane in bytes.
    #[allow(clippy::len_without_is_empty)] // empty frames are silly.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the plane's data as a slice.
    ///
    /// Panics if the frame has not been initialized.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        assert!(self.initialized);
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Returns a raw pointer to the plane's data, which may be uninitialized.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.data
    }

    /// Returns the plane's data as a mutable slice.
    ///
    /// Panics if the frame has not been initialized.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        assert!(self.initialized);
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
    }

    /// Returns a raw mutable pointer to the plane's data, which may be uninitialized.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.data
    }
}

/// Tests for equality of two `FramePlaneMut`s; this will panic if either is uninitialized.
impl PartialEq for FramePlaneMut<'_> {
    fn eq(&self, other: &Self) -> bool {
        // TODO: skip padding bytes?
        self.stride == other.stride && self.len == other.len && self.as_slice() == other.as_slice()
    }
}

impl Eq for FramePlaneMut<'_> {}

/// Read access to a backing buffer for a [`ConsecutiveFrame`].
///
/// This specifically does *not* use the `MaybeUninit` type to avoid providing
/// a way to transition bytes from initialized back to uninitialized.
///
/// # Safety
///
/// * The raw pointer accessors must return stable pointers valid for accessing
///   at least `len` bytes of data.
/// * If `PREINITIALIZED` is true, or after the caller writes valid data via
///   `as_mut_ptr`, the data must be initialized.
pub unsafe trait Storage {
    /// Checks if this is a valid storage for `len` bytes, and returns if it
    /// is known to be initialized.
    fn check_len(&self, len: usize) -> bool;

    /// Returns a raw pointer to the start of the storage.
    fn as_ptr(&self) -> *const u8;

    /// Notes that this storage is initialized, up to length `len`.
    ///
    /// This may be a no-op, but in the case of `Vec<u8>` it issues a
    /// `Vec::set_len` call so that `ConsecutiveFrame::into_inner` returns a
    /// `Vec` with the correct length. Note that the `len` argument here may
    /// be less than `Vec::capacity`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the storage is initialized.
    #[allow(unused_variables)]
    unsafe fn initialize(&mut self, len: usize) {}
}

/// Write access to a backing buffer for a [`ConsecutiveFrame`].
///
/// # Safety
///
/// As in [`Storage`].
pub unsafe trait StorageMut: Storage {
    /// Returns a raw pointer to the start of the storage.
    fn as_mut_ptr(&mut self) -> *mut u8;
}

unsafe impl Storage for Vec<u8> {
    #[inline]
    fn check_len(&self, len: usize) -> bool {
        assert!(len <= self.capacity());
        len <= self.len()
    }

    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.as_ptr()
    }
}
unsafe impl StorageMut for Vec<u8> {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
}

macro_rules! impl_slice_storage {
    ($t:ty { preinitialized=$preinitialized:expr }) => {
        unsafe impl Storage for &[$t] {
            fn check_len(&self, len: usize) -> bool {
                assert!(len <= <[$t]>::len(self));
                $preinitialized
            }

            #[inline]
            fn as_ptr(&self) -> *const u8 {
                <[$t]>::as_ptr(self).cast()
            }
        }
        unsafe impl Storage for &mut [$t] {
            fn check_len(&self, len: usize) -> bool {
                assert!(len <= <[$t]>::len(self));
                $preinitialized
            }

            #[inline]
            fn as_ptr(&self) -> *const u8 {
                <[$t]>::as_ptr(self).cast()
            }
        }
        unsafe impl StorageMut for &mut [$t] {
            #[inline]
            fn as_mut_ptr(&mut self) -> *mut u8 {
                <[$t]>::as_mut_ptr(self).cast()
            }
        }
    };
}

impl_slice_storage!(u8 { preinitialized=true });
impl_slice_storage!(MaybeUninit<u8> { preinitialized=false });

/// A frame which stores all planes consecutively.
pub struct ConsecutiveFrame<S> {
    initialized: bool,
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
            initialized: false,
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
            initialized: false,
            format: self.format,
            width: self.width,
            height: self.height,
            dims: self.dims,
            storage: Vec::with_capacity(self.total_size()),
        }
    }

    /// Returns a frame backend by `storage`.
    ///
    /// Whether the frame is considered initialized depends on the type of `storage`:
    /// * `&[u8]` or `&mut [u8]`: initialized.
    /// * `&[MaybeUninit<u8>]` or `&mut [MaybeUninit<u8>]`: uninitialized.
    /// * `Vec<u8>`: initialized if `Vec::len` is sufficient.
    ///
    /// Panics on overflow or if `storage` is too small.
    pub fn with_storage<S: Storage>(self, storage: S) -> ConsecutiveFrame<S> {
        ConsecutiveFrame {
            initialized: storage.check_len(self.total_size()),
            format: self.format,
            width: self.width,
            height: self.height,
            dims: self.dims,
            storage,
        }
    }
}

impl<S: Storage> ConsecutiveFrame<S> {
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

unsafe impl<S: Storage> Frame for ConsecutiveFrame<S> {
    #[inline]
    fn initialized(&self) -> bool {
        self.initialized
    }

    #[inline]
    fn format(&self) -> PixelFormat {
        self.format
    }

    #[inline]
    fn pixel_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn planes(&self) -> ArrayVec<FramePlaneRef, MAX_PLANES> {
        let ptr = self.storage.as_ptr();
        let mut off = 0;
        let mut planes = ArrayVec::new();
        for dims in self.dims.iter().take(self.format.num_planes()) {
            planes.push(FramePlaneRef {
                // SAFETY: the invariants on `dim` ensure `data` + `len` are valid.
                data: unsafe { ptr.byte_add(off) },
                stride: dims.stride,
                len: dims.stride * dims.rows,
                initialized: self.initialized,
                _phantom: PhantomData,
            });
            off += dims.stride * dims.rows;
        }
        planes
    }

    #[inline]
    unsafe fn initialize(&mut self) {
        if !self.initialized {
            self.storage.initialize(self.total_size());
            self.initialized = true;
        }
    }
}

unsafe impl<S: StorageMut> FrameMut for ConsecutiveFrame<S> {
    fn planes_mut(&mut self) -> ArrayVec<FramePlaneMut, MAX_PLANES> {
        let ptr = self.storage.as_mut_ptr();
        let mut off = 0;
        let mut planes = ArrayVec::new();
        for dims in self.dims.iter().take(self.format.num_planes()) {
            planes.push(FramePlaneMut {
                // SAFETY: this math is valid because of the invariants on `dims`.
                data: unsafe { ptr.byte_add(off) },
                stride: dims.stride,
                len: dims.stride * dims.rows,
                initialized: self.initialized,
                _phantom: PhantomData,
            });
            off += dims.stride * dims.rows;
        }
        planes
    }
}
