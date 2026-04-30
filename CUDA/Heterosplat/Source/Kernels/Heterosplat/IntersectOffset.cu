#include "Kernels/Heterosplat/IntersectOffset.h"
#include "Kernels/Thirdparty/Gsplat/IntersectOffsetKernels.cuh"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
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

void launch_intersect_offset_forward(
  const std::uint32_t n_isects,
  const std::int64_t* isect_ids,
  const std::uint32_t I,
  const std::uint32_t tile_width,
  const std::uint32_t tile_height,
  std::int32_t* offsets,
  cudaStream_t stream)
{
  assert(I > 0);
  assert(tile_width > 0);
  assert(tile_height > 0);
  assert(offsets != nullptr);

  const std::uint32_t n_tiles {tile_width * tile_height};

  if (n_isects == 0)
  {
    cudaMemsetAsync(offsets, 0, I * n_tiles * sizeof(std::int32_t), stream);
    return;
  }

  assert(isect_ids != nullptr);

  const std::uint32_t tile_n_bits {gsplat_bit_budget(n_tiles)};

  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {(n_isects + kThreadsPerBlock - 1u) / kThreadsPerBlock};

  gsplat::intersect_offset_kernel<<<grid, threads, 0, stream>>>(
    n_isects,
    isect_ids,
    I,
    n_tiles,
    tile_n_bits,
    offsets);
}

} // namespace Heterosplat
} // namespace Kernels
