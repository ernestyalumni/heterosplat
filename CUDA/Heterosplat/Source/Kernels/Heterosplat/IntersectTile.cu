#include "Kernels/Heterosplat/IntersectTile.h"
#include "Kernels/Thirdparty/Gsplat/IntersectTileKernels.cuh"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

namespace
{
constexpr std::uint32_t kThreadsPerBlock {256};

std::uint32_t gsplat_bit_budget(const std::uint32_t value)
{
  assert(value > 0);
  return static_cast<std::uint32_t>(std::floor(std::log2(value))) + 1u;
}
} // namespace

void launch_intersect_tile_forward(
  const bool packed,
  const std::uint32_t I,
  const std::uint32_t N,
  const std::uint32_t nnz,
  const std::int64_t* image_ids,
  const std::int64_t* gaussian_ids,
  const float* means2d,
  const std::int32_t* radii,
  const float* depths,
  const float* conics,
  const float* opacities,
  const std::int64_t* cum_tiles_per_gauss,
  const std::uint32_t tile_size,
  const std::uint32_t tile_width,
  const std::uint32_t tile_height,
  std::int32_t* tiles_per_gauss,
  std::int64_t* isect_ids,
  std::int32_t* flatten_ids,
  cudaStream_t stream)
{
  assert(I > 0);
  assert(tile_size > 0);
  assert(tile_width > 0);
  assert(tile_height > 0);
  assert(means2d != nullptr);
  assert(radii != nullptr);
  assert(depths != nullptr);

  const bool first_pass {cum_tiles_per_gauss == nullptr};
  if (first_pass)
  {
    assert(tiles_per_gauss != nullptr);
  }
  else
  {
    assert(isect_ids != nullptr);
    assert(flatten_ids != nullptr);
  }
  if (packed)
  {
    assert(image_ids != nullptr);
  }

  const std::uint32_t number_of_elements {packed ? nnz : I * N};
  if (number_of_elements == 0)
  {
    return;
  }

  const std::uint32_t number_of_tiles {tile_width * tile_height};
  const std::uint32_t image_n_bits {gsplat_bit_budget(I)};
  const std::uint32_t tile_n_bits {gsplat_bit_budget(number_of_tiles)};
  assert(image_n_bits + tile_n_bits <= 32);

  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {
    (number_of_elements + kThreadsPerBlock - 1u) / kThreadsPerBlock};

  gsplat::intersect_tile_kernel<float><<<grid, threads, 0, stream>>>(
    packed,
    I,
    N,
    nnz,
    image_ids,
    gaussian_ids,
    means2d,
    radii,
    depths,
    conics,
    opacities,
    cum_tiles_per_gauss,
    tile_size,
    tile_width,
    tile_height,
    tile_n_bits,
    image_n_bits,
    tiles_per_gauss,
    isect_ids,
    flatten_ids);
}

} // namespace Heterosplat
} // namespace Kernels
