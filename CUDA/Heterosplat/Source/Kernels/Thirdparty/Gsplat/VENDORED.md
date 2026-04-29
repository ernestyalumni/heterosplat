# Vendored gsplat kernels

Source: https://github.com/nerfstudio-project/gsplat @ `53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5`
Imported: 2026-04-28
License: Apache 2.0 (see `LICENSE`, `NOTICE`).

## What lives here

| File | Origin | Status |
|---|---|---|
| `Common.h` | upstream `cuda/include/Common.h` | **patched** — torch macros stripped (see `NOTICE`) |
| `Config.h` | upstream `cuda/csrc/Config.h` | as-is |
| `Utils.cuh` | upstream `cuda/include/Utils.cuh` | as-is |
| `QuatScaleToCovarKernels.cuh` | extracted from `cuda/csrc/QuatScaleToCovarCUDA.cu` | **patched** — only `__global__` templates kept; launchers dropped |
| `SphericalHarmonicsKernels.cuh` | extracted from `cuda/csrc/SphericalHarmonicsCUDA.cu` | **patched** — `__device__` helpers + `__global__` templates kept; launchers dropped; `gpuAtomicAdd` → `atomicAdd` |
| `IntersectTileKernels.cuh` | extracted from `cuda/csrc/IntersectTile.cu` | **patched** — AccuTile helpers + `intersect_tile_kernel` kept; ATen launcher, CUB sort helpers, and `intersect_offset` dropped |

## What's NOT here (and why)

- **`launch_*` functions** — they take `at::Tensor`. Replaced by raw-pointer launchers in `../../Heterosplat/`.
- **2DGS / 3DGUT / lidar / external-distortion / camera-wrappers** — out of scope per PLAN.md.

## How to add another kernel

For each new kernel `Foo` from gsplat:
1. Open upstream `FooCUDA.cu`.
2. Copy the templated `__global__` kernel(s) into `FooKernels.cuh` here, **without** the `launch_*` host functions or `AT_DISPATCH_*` macros.
3. Note the modification in `NOTICE`.
4. Write a raw-pointer launcher under `../../Heterosplat/Foo.{h,cu}`.

## Bumping upstream

Don't, unless there's a concrete bugfix we need. Once kernels pass our oracle tests, we're frozen at the SHA above. If you do bump:
1. Diff `Utils.cuh` and `Common.h` against upstream — re-apply the patches in `NOTICE`.
2. Re-run the full `Check` test suite against captured Python+gsplat fixtures.
3. Update the SHA + date above.
