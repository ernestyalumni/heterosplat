#ifndef KERNELS_HETEROSPLAT_INTERSECT_OFFSET_H
#define KERNELS_HETEROSPLAT_INTERSECT_OFFSET_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `intersect_offset` forward kernel.
///
/// Converts sorted intersection ids (from `intersect_tile` + radix sort) into
/// per-image, per-tile start-offsets. Each entry `offsets[image * n_tiles + tile]`
/// gives the index into the sorted intersection arrays where that (image, tile)
/// pair begins.
///
/// \param n_isects   Total number of intersections (length of `isect_ids`).
/// \param isect_ids  Sorted intersection ids. [n_isects]
/// \param I          Number of images.
/// \param tile_width   Tile grid width.
/// \param tile_height  Tile grid height.
/// \param offsets    Output tile offsets. [I * tile_height * tile_width]
///                   Must be pre-allocated by the caller.
/// \param stream     CUDA stream.
//------------------------------------------------------------------------------
void launch_intersect_offset_forward(
  std::uint32_t n_isects,
  const std::int64_t* isect_ids,
  std::uint32_t I,
  std::uint32_t tile_width,
  std::uint32_t tile_height,
  std::int32_t* offsets,
  cudaStream_t stream);

} // namespace Heterosplat
} // namespace Kernels

#endif // KERNELS_HETEROSPLAT_INTERSECT_OFFSET_H
