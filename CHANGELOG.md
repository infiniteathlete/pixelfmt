# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1](https://github.com/infiniteathlete/pixelfmt/compare/v0.2.0...v0.2.1) - 2025-02-27

### Fixed

- offer longer lifetime slice accessors

### Other

- fix clippy 1.85 warning
# version 0.2.0 (2025-01-10)

* BREAKING: simplified by dropping the concept of an uninitialized frame.
  This concept was meant to improve performance by avoiding the need to zero
  memory prior to writing into a frame. However, this was only effective for
  smaller frames. With common memory allocators such as glibc's, large
  allocations such as those backing 8K frames are not reused, meaning each comes
  from a fresh `mmap` call and is zeroed by the OS on first access anyway. The
  overhead of the `mmap`/`munmap` and minor page faults is significant, and the
  best way to avoid it is for callers to pool/reuse frames. When they do so,
  there's no benefit to supporting uninitialized frames. Dropping this support
  allows us to have purely safe interfaces.

# version 0.1.0 (2024-10-22)

* initial version.
