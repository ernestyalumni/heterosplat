#include "Normalize/Transform.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace Normalize
{

namespace
{

constexpr std::uint32_t kThreadsPerBlock {256};

// Quaternion multiplication: result = a * b
// Convention: (w, x, y, z)
__device__ void quat_multiply(
  const float aw, const float ax, const float ay, const float az,
  const float bw, const float bx, const float by, const float bz,
  float& rw, float& rx, float& ry, float& rz)
{
  rw = aw * bw - ax * bx - ay * by - az * bz;
  rx = aw * bx + ax * bw + ay * bz - az * by;
  ry = aw * by - ax * bz + ay * bw + az * bx;
  rz = aw * bz + ax * by - ay * bx + az * bw;
}

__global__ void apply_similarity_transform_kernel(
  const std::uint32_t N,
  const float* __restrict__ rotation,
  const float scale,
  const float* __restrict__ translation,
  float* __restrict__ means,
  float* __restrict__ quats,
  float* __restrict__ log_scales)
{
  const std::uint32_t n {blockIdx.x * blockDim.x + threadIdx.x};
  if (n >= N) return;

  // Load rotation matrix (row-major 3x3)
  const float r00 {rotation[0]}, r01 {rotation[1]}, r02 {rotation[2]};
  const float r10 {rotation[3]}, r11 {rotation[4]}, r12 {rotation[5]};
  const float r20 {rotation[6]}, r21 {rotation[7]}, r22 {rotation[8]};

  // Transform means: mean' = scale * R * mean + t
  const float mx {means[n * 3 + 0]};
  const float my {means[n * 3 + 1]};
  const float mz {means[n * 3 + 2]};

  means[n * 3 + 0] = scale * (r00 * mx + r01 * my + r02 * mz) + translation[0];
  means[n * 3 + 1] = scale * (r10 * mx + r11 * my + r12 * mz) + translation[1];
  means[n * 3 + 2] = scale * (r20 * mx + r21 * my + r22 * mz) + translation[2];

  // Convert rotation matrix to quaternion for composition.
  // Shepperd's method for robust matrix-to-quaternion conversion.
  float rw, rx, ry, rz;
  const float trace {r00 + r11 + r22};
  if (trace > 0.0f)
  {
    const float s {0.5f / sqrtf(trace + 1.0f)};
    rw = 0.25f / s;
    rx = (r21 - r12) * s;
    ry = (r02 - r20) * s;
    rz = (r10 - r01) * s;
  }
  else if (r00 > r11 && r00 > r22)
  {
    const float s {2.0f * sqrtf(1.0f + r00 - r11 - r22)};
    rw = (r21 - r12) / s;
    rx = 0.25f * s;
    ry = (r01 + r10) / s;
    rz = (r02 + r20) / s;
  }
  else if (r11 > r22)
  {
    const float s {2.0f * sqrtf(1.0f + r11 - r00 - r22)};
    rw = (r02 - r20) / s;
    rx = (r01 + r10) / s;
    ry = 0.25f * s;
    rz = (r12 + r21) / s;
  }
  else
  {
    const float s {2.0f * sqrtf(1.0f + r22 - r00 - r11)};
    rw = (r10 - r01) / s;
    rx = (r02 + r20) / s;
    ry = (r12 + r21) / s;
    rz = 0.25f * s;
  }

  // Normalize the rotation quaternion
  const float r_inv {rsqrtf(rw * rw + rx * rx + ry * ry + rz * rz)};
  rw *= r_inv; rx *= r_inv; ry *= r_inv; rz *= r_inv;

  // Compose quaternions: quat' = R_quat * quat
  const float qw {quats[n * 4 + 0]};
  const float qx {quats[n * 4 + 1]};
  const float qy {quats[n * 4 + 2]};
  const float qz {quats[n * 4 + 3]};

  float nw, nx, ny, nz;
  quat_multiply(rw, rx, ry, rz, qw, qx, qy, qz, nw, nx, ny, nz);

  // Normalize result
  const float q_inv {rsqrtf(nw * nw + nx * nx + ny * ny + nz * nz)};
  quats[n * 4 + 0] = nw * q_inv;
  quats[n * 4 + 1] = nx * q_inv;
  quats[n * 4 + 2] = ny * q_inv;
  quats[n * 4 + 3] = nz * q_inv;

  // Scale log_scales: log_scale' = log_scale + log(scale)
  const float log_s {logf(scale)};
  log_scales[n * 3 + 0] += log_s;
  log_scales[n * 3 + 1] += log_s;
  log_scales[n * 3 + 2] += log_s;
}

__global__ void apply_homogeneous_transform_means_kernel(
  const std::uint32_t N,
  const float* __restrict__ transform,
  float* __restrict__ means)
{
  const std::uint32_t n {blockIdx.x * blockDim.x + threadIdx.x};
  if (n >= N) return;

  // Row-major 4x4 homogeneous transform
  const float mx {means[n * 3 + 0]};
  const float my {means[n * 3 + 1]};
  const float mz {means[n * 3 + 2]};

  const float w {transform[12] * mx + transform[13] * my
    + transform[14] * mz + transform[15]};
  const float inv_w {1.0f / w};

  means[n * 3 + 0] = (transform[0] * mx + transform[1] * my
    + transform[2] * mz + transform[3]) * inv_w;
  means[n * 3 + 1] = (transform[4] * mx + transform[5] * my
    + transform[6] * mz + transform[7]) * inv_w;
  means[n * 3 + 2] = (transform[8] * mx + transform[9] * my
    + transform[10] * mz + transform[11]) * inv_w;
}

} // namespace

void launch_apply_similarity_transform(
  const std::uint32_t N,
  const float* rotation,
  const float scale,
  const float* translation,
  float* means,
  float* quats,
  float* log_scales,
  cudaStream_t stream)
{
  if (N == 0) return;
  const dim3 grid {(N + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  apply_similarity_transform_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
    N, rotation, scale, translation, means, quats, log_scales);
}

void launch_apply_homogeneous_transform_means(
  const std::uint32_t N,
  const float* transform,
  float* means,
  cudaStream_t stream)
{
  if (N == 0) return;
  const dim3 grid {(N + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  apply_homogeneous_transform_means_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
    N, transform, means);
}

} // namespace Normalize
