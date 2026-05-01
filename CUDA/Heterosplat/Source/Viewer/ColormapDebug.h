#ifndef VIEWER_COLORMAP_DEBUG_H
#define VIEWER_COLORMAP_DEBUG_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Viewer
{

/// Apply a per-axis colormap to Gaussian SH DC coefficients for debug
/// visualization. Maps the selected coordinate axis to a turbo-like colormap,
/// overwriting the DC term of each Gaussian's SH coefficients.
///
/// \param N         Number of Gaussians.
/// \param means     [N, 3] positions (device).
/// \param min_x, max_x, min_y, max_y, min_z, max_z  Bounding box.
/// \param axis      0 = X, 1 = Y, 2 = Z.
/// \param K         SH coefficient count per Gaussian, (sh_degree+1)^2.
/// \param sh_coeffs [N, K, 3] SH coefficients (device, modified in-place).
/// \param stream    CUDA stream.
void launch_colormap_per_axis(
  std::uint32_t N,
  const float* means,
  float min_x, float max_x,
  float min_y, float max_y,
  float min_z, float max_z,
  std::uint32_t axis,
  std::uint32_t K,
  float* sh_coeffs,
  cudaStream_t stream);

} // namespace Viewer

#endif // VIEWER_COLORMAP_DEBUG_H
