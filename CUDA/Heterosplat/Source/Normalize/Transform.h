#ifndef NORMALIZE_TRANSFORM_H
#define NORMALIZE_TRANSFORM_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Normalize
{

/// Apply a similarity transform (rotation + uniform scale + translation) to
/// Gaussian splat parameters in-place.
///
/// The transform is parameterized as: p' = scale * R * p + t
/// where R is a 3x3 rotation matrix, scale is a uniform scalar, and t is a
/// 3D translation vector.
///
/// Effect on Gaussian parameters:
///   means:     mean' = scale * R * mean + t
///   quats:     quat' = R_quat * quat  (quaternion composition)
///   log_scales: log_scale' = log_scale + log(scale)  (uniform scaling in log-space)
///   opacities:  unchanged
///   sh_coeffs:  DC unchanged, higher-order rotated (skipped for simplicity;
///               rotation of SH basis requires Wigner D-matrices — deferred)
///
/// \param N             Number of Gaussians.
/// \param rotation      Row-major 3x3 rotation matrix (9 floats, device ptr).
/// \param scale         Uniform scale factor.
/// \param translation   3D translation (3 floats, device ptr).
/// \param means         [N, 3] in-place (device).
/// \param quats         [N, 4] in-place (device), (w, x, y, z).
/// \param log_scales    [N, 3] in-place (device).
/// \param stream        CUDA stream.
void launch_apply_similarity_transform(
  std::uint32_t N,
  const float* rotation,
  float scale,
  const float* translation,
  float* means,
  float* quats,
  float* log_scales,
  cudaStream_t stream);

/// Apply a full 4x4 homogeneous transform to Gaussian means only.
/// For non-rigid transforms where rotation extraction is ambiguous.
///
/// \param N         Number of Gaussians.
/// \param transform Row-major 4x4 matrix (16 floats, device ptr).
/// \param means     [N, 3] in-place (device).
/// \param stream    CUDA stream.
void launch_apply_homogeneous_transform_means(
  std::uint32_t N,
  const float* transform,
  float* means,
  cudaStream_t stream);

} // namespace Normalize

#endif // NORMALIZE_TRANSFORM_H
