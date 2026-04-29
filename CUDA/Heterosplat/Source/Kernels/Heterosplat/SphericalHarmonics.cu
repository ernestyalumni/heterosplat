#include "Kernels/Heterosplat/SphericalHarmonics.h"
#include "Kernels/Thirdparty/Gsplat/SphericalHarmonicsKernels.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

namespace
{
//------------------------------------------------------------------------------
/// gsplat upstream uses dim3(256). Match exactly so any future oracle
/// fixture comparison stays bit-for-bit aligned with their output.
//------------------------------------------------------------------------------
constexpr std::uint32_t kThreadsPerBlock {256};
} // namespace

void launch_spherical_harmonics_forward(
  const std::uint32_t N,
  const std::uint32_t K,
  const std::uint32_t degrees_to_use,
  const float* dirs,
  const float* coeffs,
  const bool* masks,
  float* colors,
  cudaStream_t stream)
{
  // Parallelise over (Gaussian, channel) pairs: total work = N * 3.
  const std::uint32_t total_threads {N * 3u};
  if (total_threads == 0)
  {
    return;
  }
  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {(total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock};

  // The vendored kernel takes vec3* for dirs; reinterpret-cast the float*
  // input. Layout is identical (3 contiguous floats per direction).
  gsplat::spherical_harmonics_fwd_kernel<float><<<grid, threads, 0, stream>>>(
    N,
    K,
    degrees_to_use,
    reinterpret_cast<const gsplat::vec3*>(dirs),
    coeffs,
    masks,
    colors);
}

void launch_spherical_harmonics_backward(
  const std::uint32_t N,
  const std::uint32_t K,
  const std::uint32_t degrees_to_use,
  const float* dirs,
  const float* coeffs,
  const bool* masks,
  const float* v_colors,
  float* v_coeffs,
  float* v_dirs,
  cudaStream_t stream)
{
  const std::uint32_t total_threads {N * 3u};
  if (total_threads == 0)
  {
    return;
  }
  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {(total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock};

  gsplat::spherical_harmonics_bwd_kernel<float><<<grid, threads, 0, stream>>>(
    N,
    K,
    degrees_to_use,
    reinterpret_cast<const gsplat::vec3*>(dirs),
    coeffs,
    masks,
    v_colors,
    v_coeffs,
    v_dirs);
}

} // namespace Heterosplat
} // namespace Kernels
