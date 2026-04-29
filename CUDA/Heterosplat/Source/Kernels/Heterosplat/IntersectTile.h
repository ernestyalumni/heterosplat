#ifndef KERNELS_HETEROSPLAT_INTERSECT_TILE_H
#define KERNELS_HETEROSPLAT_INTERSECT_TILE_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `intersect_tile` forward kernel.
///
/// Maps projected 2D Gaussians to touched image tiles. This exposes gsplat's
/// low-level two-pass primitive without allocating, prefix-summing, or sorting:
/// first call with `cum_tiles_per_gauss == nullptr` to fill
/// `tiles_per_gauss`, then prefix-sum that array on the caller side and call
/// again with `cum_tiles_per_gauss`, `isect_ids`, and `flatten_ids`.
///
/// If `conics` and `opacities` are both non-null, the kernel uses gsplat's
/// AccuTile/SNUGBOX ellipse intersection path. Otherwise it falls back to the
/// axis-aligned radius box path.
//------------------------------------------------------------------------------
void launch_intersect_tile_forward(
  bool packed,
  std::uint32_t I,
  std::uint32_t N,
  std::uint32_t nnz,
  const std::int64_t* image_ids,
  const std::int64_t* gaussian_ids,
  const float* means2d,
  const std::int32_t* radii,
  const float* depths,
  const float* conics,
  const float* opacities,
  const std::int64_t* cum_tiles_per_gauss,
  std::uint32_t tile_size,
  std::uint32_t tile_width,
  std::uint32_t tile_height,
  std::int32_t* tiles_per_gauss,
  std::int64_t* isect_ids,
  std::int32_t* flatten_ids,
  cudaStream_t stream);

} // namespace Heterosplat
} // namespace Kernels

#endif // KERNELS_HETEROSPLAT_INTERSECT_TILE_H
