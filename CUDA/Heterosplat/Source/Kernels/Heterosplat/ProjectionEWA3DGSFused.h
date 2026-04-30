#ifndef KERNELS_HETEROSPLAT_PROJECTION_EWA_3DGS_FUSED_H
#define KERNELS_HETEROSPLAT_PROJECTION_EWA_3DGS_FUSED_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `projection_ewa_3dgs_fused` fwd.
///
/// Projects 3D Gaussians into 2D via EWA splatting. For each batch element
/// and camera, transforms Gaussian means/covariances to camera space, projects
/// to 2D, computes tight bounding-box radii, and writes conics (inverse 2D
/// covariance upper triangle).
///
/// Covariance input: either `covars` (upper triangle [B,N,6]) or both
/// `quats` [B,N,4] and `scales` [B,N,3]. The unused path's pointers are null.
///
/// Camera model: 0=PINHOLE, 1=ORTHO, 2=FISHEYE.
///
/// \param compensations  Optional output [B,C,N]; null to skip.
//------------------------------------------------------------------------------
void launch_projection_ewa_3dgs_fused_forward(
  std::uint32_t B,
  std::uint32_t C,
  std::uint32_t N,
  const float* means,
  const float* covars,
  const float* quats,
  const float* scales,
  const float* opacities,
  const float* viewmats,
  const float* Ks,
  std::uint32_t image_width,
  std::uint32_t image_height,
  float eps2d,
  float near_plane,
  float far_plane,
  float radius_clip,
  std::uint32_t camera_model,
  std::int32_t* radii,
  float* means2d,
  float* depths,
  float* conics,
  float* compensations,
  cudaStream_t stream);

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `projection_ewa_3dgs_fused` bwd.
///
/// Backpropagates through the EWA projection. Gradient outputs from the
/// rasterizer are `v_means2d`, `v_depths`, `v_conics`, and optionally
/// `v_compensations`. The kernel accumulates into `v_means` (and either
/// `v_covars` or `v_quats`+`v_scales`) via warp-level reduction + atomicAdd.
///
/// Caller must zero `v_means`, `v_covars`/`v_quats`/`v_scales`, and
/// `v_viewmats` before calling (atomicAdd accumulates).
//------------------------------------------------------------------------------
void launch_projection_ewa_3dgs_fused_backward(
  std::uint32_t B,
  std::uint32_t C,
  std::uint32_t N,
  const float* means,
  const float* covars,
  const float* quats,
  const float* scales,
  const float* viewmats,
  const float* Ks,
  std::uint32_t image_width,
  std::uint32_t image_height,
  float eps2d,
  std::uint32_t camera_model,
  const std::int32_t* radii,
  const float* conics,
  const float* compensations,
  const float* v_means2d,
  const float* v_depths,
  const float* v_conics,
  const float* v_compensations,
  float* v_means,
  float* v_covars,
  float* v_quats,
  float* v_scales,
  float* v_viewmats,
  cudaStream_t stream);

} // namespace Heterosplat
} // namespace Kernels

#endif // KERNELS_HETEROSPLAT_PROJECTION_EWA_3DGS_FUSED_H
