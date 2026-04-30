#include "Kernels/Heterosplat/RasterizeToPixels3DGS.h"
#include "Kernels/Thirdparty/Gsplat/RasterizeToPixels3DGSKernels.cuh"

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

namespace
{

constexpr std::uint32_t kColorDim {3};

} // namespace

void launch_rasterize_to_pixels_3dgs_forward(
  const std::uint32_t I,
  const std::uint32_t N,
  const std::uint32_t n_isects,
  const bool packed,
  const float* means2d,
  const float* conics,
  const float* colors,
  const float* opacities,
  const float* backgrounds,
  const bool* masks,
  const std::uint32_t image_width,
  const std::uint32_t image_height,
  const std::uint32_t tile_size,
  const std::int32_t* tile_offsets,
  const std::int32_t* flatten_ids,
  float* render_colors,
  float* render_alphas,
  std::int32_t* last_ids,
  cudaStream_t stream)
{
  assert(tile_size > 0);

  if (I == 0 || (n_isects == 0 && masks == nullptr))
  {
    return;
  }

  assert(means2d != nullptr);
  assert(conics != nullptr);
  assert(colors != nullptr);
  assert(opacities != nullptr);
  assert(tile_offsets != nullptr);
  assert(render_colors != nullptr);
  assert(render_alphas != nullptr);
  assert(last_ids != nullptr);

  const std::uint32_t tile_width {
    (image_width + tile_size - 1) / tile_size};
  const std::uint32_t tile_height {
    (image_height + tile_size - 1) / tile_size};

  const dim3 threads {tile_size, tile_size, 1};
  const dim3 grid {I, tile_height, tile_width};

  const std::int64_t shmem_size {
    static_cast<std::int64_t>(tile_size * tile_size) *
    static_cast<std::int64_t>(
      sizeof(std::int32_t) + sizeof(gsplat::vec3) + sizeof(gsplat::vec3))};

  cudaFuncSetAttribute(
    gsplat::rasterize_to_pixels_3dgs_fwd_kernel<kColorDim, float>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    static_cast<int>(shmem_size));

  gsplat::rasterize_to_pixels_3dgs_fwd_kernel<kColorDim, float>
    <<<grid, threads, static_cast<unsigned int>(shmem_size), stream>>>(
      I, N, n_isects, packed,
      reinterpret_cast<const gsplat::vec2*>(means2d),
      reinterpret_cast<const gsplat::vec3*>(conics),
      colors, opacities, backgrounds, masks,
      image_width, image_height, tile_size, tile_width, tile_height,
      tile_offsets, flatten_ids,
      render_colors, render_alphas, last_ids);
}

void launch_rasterize_to_pixels_3dgs_backward(
  const std::uint32_t I,
  const std::uint32_t N,
  const std::uint32_t n_isects,
  const bool packed,
  const float* means2d,
  const float* conics,
  const float* colors,
  const float* opacities,
  const float* backgrounds,
  const bool* masks,
  const std::uint32_t image_width,
  const std::uint32_t image_height,
  const std::uint32_t tile_size,
  const std::int32_t* tile_offsets,
  const std::int32_t* flatten_ids,
  const float* render_alphas,
  const std::int32_t* last_ids,
  const float* v_render_colors,
  const float* v_render_alphas,
  float* v_means2d_abs,
  float* v_means2d,
  float* v_conics,
  float* v_colors,
  float* v_opacities,
  cudaStream_t stream)
{
  assert(means2d != nullptr);
  assert(conics != nullptr);
  assert(colors != nullptr);
  assert(opacities != nullptr);
  assert(tile_offsets != nullptr);
  assert(render_alphas != nullptr);
  assert(last_ids != nullptr);
  assert(v_render_colors != nullptr);
  assert(v_render_alphas != nullptr);
  assert(v_means2d != nullptr);
  assert(v_conics != nullptr);
  assert(v_colors != nullptr);
  assert(v_opacities != nullptr);
  assert(tile_size > 0);

  if (n_isects == 0 || I == 0)
  {
    return;
  }

  const std::uint32_t tile_width {
    (image_width + tile_size - 1) / tile_size};
  const std::uint32_t tile_height {
    (image_height + tile_size - 1) / tile_size};

  const dim3 threads {tile_size, tile_size, 1};
  const dim3 grid {I, tile_height, tile_width};

  const std::int64_t shmem_size {
    static_cast<std::int64_t>(tile_size * tile_size) *
    static_cast<std::int64_t>(
      sizeof(std::int32_t) + sizeof(gsplat::vec3) + sizeof(gsplat::vec3) +
      sizeof(float) * kColorDim)};

  cudaFuncSetAttribute(
    gsplat::rasterize_to_pixels_3dgs_bwd_kernel<kColorDim, float>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    static_cast<int>(shmem_size));

  gsplat::rasterize_to_pixels_3dgs_bwd_kernel<kColorDim, float>
    <<<grid, threads, static_cast<unsigned int>(shmem_size), stream>>>(
      I, N, n_isects, packed,
      reinterpret_cast<const gsplat::vec2*>(means2d),
      reinterpret_cast<const gsplat::vec3*>(conics),
      colors, opacities, backgrounds, masks,
      image_width, image_height, tile_size, tile_width, tile_height,
      tile_offsets, flatten_ids,
      render_alphas, last_ids,
      v_render_colors, v_render_alphas,
      v_means2d_abs != nullptr
        ? reinterpret_cast<gsplat::vec2*>(v_means2d_abs)
        : nullptr,
      reinterpret_cast<gsplat::vec2*>(v_means2d),
      reinterpret_cast<gsplat::vec3*>(v_conics),
      v_colors, v_opacities);
}

} // namespace Heterosplat
} // namespace Kernels
