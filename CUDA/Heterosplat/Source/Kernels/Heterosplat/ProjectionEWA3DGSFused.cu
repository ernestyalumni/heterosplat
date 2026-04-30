#include "Kernels/Heterosplat/ProjectionEWA3DGSFused.h"
#include "Kernels/Thirdparty/Gsplat/ProjectionEWA3DGSFusedKernels.cuh"

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

namespace
{
constexpr std::uint32_t kThreadsPerBlock {256};
} // namespace

void launch_projection_ewa_3dgs_fused_forward(
  const std::uint32_t B,
  const std::uint32_t C,
  const std::uint32_t N,
  const float* means,
  const float* covars,
  const float* quats,
  const float* scales,
  const float* opacities,
  const float* viewmats,
  const float* Ks,
  const std::uint32_t image_width,
  const std::uint32_t image_height,
  const float eps2d,
  const float near_plane,
  const float far_plane,
  const float radius_clip,
  const std::uint32_t camera_model,
  std::int32_t* radii,
  float* means2d,
  float* depths,
  float* conics,
  float* compensations,
  cudaStream_t stream)
{
  assert(means != nullptr);
  assert(viewmats != nullptr);
  assert(Ks != nullptr);
  assert(radii != nullptr);
  assert(means2d != nullptr);
  assert(depths != nullptr);
  assert(conics != nullptr);
  assert(covars != nullptr || (quats != nullptr && scales != nullptr));

  const std::uint32_t n_elements {B * C * N};
  if (n_elements == 0)
  {
    return;
  }

  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {(n_elements + kThreadsPerBlock - 1u) / kThreadsPerBlock};

  gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>
    <<<grid, threads, 0, stream>>>(
    B,
    C,
    N,
    means,
    covars,
    quats,
    scales,
    opacities,
    viewmats,
    Ks,
    image_width,
    image_height,
    eps2d,
    near_plane,
    far_plane,
    radius_clip,
    static_cast<gsplat::CameraModelType>(camera_model),
    radii,
    means2d,
    depths,
    conics,
    compensations);
}

void launch_projection_ewa_3dgs_fused_backward(
  const std::uint32_t B,
  const std::uint32_t C,
  const std::uint32_t N,
  const float* means,
  const float* covars,
  const float* quats,
  const float* scales,
  const float* viewmats,
  const float* Ks,
  const std::uint32_t image_width,
  const std::uint32_t image_height,
  const float eps2d,
  const std::uint32_t camera_model,
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
  cudaStream_t stream)
{
  assert(means != nullptr);
  assert(viewmats != nullptr);
  assert(Ks != nullptr);
  assert(radii != nullptr);
  assert(conics != nullptr);
  assert(v_means2d != nullptr);
  assert(v_depths != nullptr);
  assert(v_conics != nullptr);
  assert(covars != nullptr || (quats != nullptr && scales != nullptr));

  const std::uint32_t n_elements {B * C * N};
  if (n_elements == 0)
  {
    return;
  }

  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {(n_elements + kThreadsPerBlock - 1u) / kThreadsPerBlock};

  gsplat::projection_ewa_3dgs_fused_bwd_kernel<float>
    <<<grid, threads, 0, stream>>>(
    B,
    C,
    N,
    means,
    covars,
    quats,
    scales,
    viewmats,
    Ks,
    image_width,
    image_height,
    eps2d,
    static_cast<gsplat::CameraModelType>(camera_model),
    radii,
    conics,
    compensations,
    v_means2d,
    v_depths,
    v_conics,
    v_compensations,
    v_means,
    v_covars,
    v_quats,
    v_scales,
    v_viewmats);
}

} // namespace Heterosplat
} // namespace Kernels
