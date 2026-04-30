#ifndef IO_PLY_WRITER_H
#define IO_PLY_WRITER_H

#include <cstdint>
#include <string>

namespace IO
{

/// Write Gaussian splat parameters in the standard 3DGS PLY format
/// (binary little-endian, compatible with SuperSplat / antimatter15).
///
/// All input arrays are host pointers. The caller is responsible for
/// downloading GPU buffers before calling.
///
/// \param path         Output .ply file path.
/// \param num_gaussians  Number of Gaussians.
/// \param means        [N, 3] — world-space positions.
/// \param sh_degree    SH degree (0..3). Determines coefficient count.
/// \param sh_coeffs    [N, K, 3] — SH coefficients. K = (sh_degree+1)^2.
///                     DC is the first coefficient per channel.
/// \param opacities    [N] — raw opacity (sigmoid-inverse, i.e. logit).
/// \param scales       [N, 3] — raw scales (log-space).
/// \param quats        [N, 4] — quaternions (w, x, y, z).
void write_gaussians_ply(
  const std::string& path,
  std::uint32_t num_gaussians,
  const float* means,
  std::uint32_t sh_degree,
  const float* sh_coeffs,
  const float* opacities,
  const float* scales,
  const float* quats);

} // namespace IO

#endif // IO_PLY_WRITER_H
