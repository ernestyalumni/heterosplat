#ifndef KERNELS_HETEROSPLAT_RASTERIZE_TO_PIXELS_3DGS_H
#define KERNELS_HETEROSPLAT_RASTERIZE_TO_PIXELS_3DGS_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `rasterize_to_pixels_3dgs` fwd.
///
/// Per-tile alpha-compositing rasterizer. Each CUDA block covers one tile
/// (tile_size x tile_size pixels). Threads cooperatively load Gaussians from
/// shared memory in front-to-back depth order and composite into per-pixel
/// colour and alpha accumulators.
///
/// The colour channel count CDIM is a compile-time constant. We instantiate
/// CDIM=3 (RGB). For other channel counts, add explicit instantiations in
/// RasterizeToPixels3DGS.cu.
///
/// \param I             Number of images.
/// \param N             Number of Gaussians per image (dense mode). Unused
///                      in packed mode.
/// \param n_isects      Total intersection count (length of flatten_ids).
/// \param packed        If true, means2d/conics/colors/opacities are [nnz,*];
///                      otherwise [I,N,*].
/// \param means2d       Projected 2D means [I,N,2] or [nnz,2].
/// \param conics        Inverse 2D covariance upper-tri [I,N,3] or [nnz,3].
/// \param colors        Per-Gaussian colour [I,N,CDIM] or [nnz,CDIM].
/// \param opacities     Per-Gaussian opacity [I,N] or [nnz].
/// \param backgrounds   Optional per-image background colour [I,CDIM]; null
///                      means black.
/// \param masks         Optional per-tile mask [I,tile_h,tile_w]; null =
///                      rasterize all tiles. False tiles get background only.
/// \param tile_offsets   Per-tile start index into sorted isects [I,tile_h,tile_w].
/// \param flatten_ids   Sorted Gaussian indices [n_isects].
/// \param render_colors Output [I,image_h,image_w,CDIM].
/// \param render_alphas Output [I,image_h,image_w].
/// \param last_ids      Output per-pixel index of last contributing Gaussian
///                      [I,image_h,image_w].
/// \param stream        CUDA stream; nullptr = default.
//------------------------------------------------------------------------------
void launch_rasterize_to_pixels_3dgs_forward(
  std::uint32_t I,
  std::uint32_t N,
  std::uint32_t n_isects,
  bool packed,
  const float* means2d,
  const float* conics,
  const float* colors,
  const float* opacities,
  const float* backgrounds,
  const bool* masks,
  std::uint32_t image_width,
  std::uint32_t image_height,
  std::uint32_t tile_size,
  const std::int32_t* tile_offsets,
  const std::int32_t* flatten_ids,
  float* render_colors,
  float* render_alphas,
  std::int32_t* last_ids,
  cudaStream_t stream);

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `rasterize_to_pixels_3dgs` bwd.
///
/// Back-to-front traversal recomputing alpha-compositing weights. Produces
/// gradients for means2d, conics, colors, and opacities via warp-level
/// reduction + atomicAdd.
///
/// Caller must pre-zero all gradient output buffers (atomicAdd accumulates).
///
/// \param v_means2d_abs  Optional absolute-value gradient for means2d
///                       (used for adaptive densification); null to skip.
//------------------------------------------------------------------------------
void launch_rasterize_to_pixels_3dgs_backward(
  std::uint32_t I,
  std::uint32_t N,
  std::uint32_t n_isects,
  bool packed,
  const float* means2d,
  const float* conics,
  const float* colors,
  const float* opacities,
  const float* backgrounds,
  const bool* masks,
  std::uint32_t image_width,
  std::uint32_t image_height,
  std::uint32_t tile_size,
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
  cudaStream_t stream);

} // namespace Heterosplat
} // namespace Kernels

#endif // KERNELS_HETEROSPLAT_RASTERIZE_TO_PIXELS_3DGS_H
