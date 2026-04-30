#include "Training/Activations.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace Training
{

namespace
{

constexpr std::uint32_t kThreadsPerBlock {256};

__global__ void sigmoid_forward_kernel(
  const std::uint32_t count,
  const float* __restrict__ logits,
  float* __restrict__ out)
{
  const std::uint32_t idx {blockIdx.x * blockDim.x + threadIdx.x};
  if (idx >= count) return;
  out[idx] = 1.0f / (1.0f + expf(-logits[idx]));
}

__global__ void exp_forward_kernel(
  const std::uint32_t count,
  const float* __restrict__ log_vals,
  float* __restrict__ out)
{
  const std::uint32_t idx {blockIdx.x * blockDim.x + threadIdx.x};
  if (idx >= count) return;
  out[idx] = expf(log_vals[idx]);
}

__global__ void sigmoid_backward_chain_kernel(
  const std::uint32_t count,
  const float* __restrict__ actual,
  const float* __restrict__ grad_actual,
  float* __restrict__ grad_logit)
{
  const std::uint32_t idx {blockIdx.x * blockDim.x + threadIdx.x};
  if (idx >= count) return;
  const float s {actual[idx]};
  grad_logit[idx] = grad_actual[idx] * s * (1.0f - s);
}

__global__ void exp_backward_chain_kernel(
  const std::uint32_t count,
  const float* __restrict__ actual,
  const float* __restrict__ grad_actual,
  float* __restrict__ grad_log)
{
  const std::uint32_t idx {blockIdx.x * blockDim.x + threadIdx.x};
  if (idx >= count) return;
  grad_log[idx] = grad_actual[idx] * actual[idx];
}

__global__ void normalize_quaternions_kernel(
  const std::uint32_t N,
  float* __restrict__ quats)
{
  const std::uint32_t n {blockIdx.x * blockDim.x + threadIdx.x};
  if (n >= N) return;
  const float w {quats[n * 4 + 0]};
  const float x {quats[n * 4 + 1]};
  const float y {quats[n * 4 + 2]};
  const float z {quats[n * 4 + 3]};
  const float inv_norm {rsqrtf(w * w + x * x + y * y + z * z)};
  quats[n * 4 + 0] = w * inv_norm;
  quats[n * 4 + 1] = x * inv_norm;
  quats[n * 4 + 2] = y * inv_norm;
  quats[n * 4 + 3] = z * inv_norm;
}

__global__ void compute_view_directions_kernel(
  const std::uint32_t N,
  const float* __restrict__ means,
  const float cam_x,
  const float cam_y,
  const float cam_z,
  float* __restrict__ dirs)
{
  const std::uint32_t n {blockIdx.x * blockDim.x + threadIdx.x};
  if (n >= N) return;
  float dx {cam_x - means[n * 3 + 0]};
  float dy {cam_y - means[n * 3 + 1]};
  float dz {cam_z - means[n * 3 + 2]};
  const float inv_norm {rsqrtf(dx * dx + dy * dy + dz * dz + 1e-12f)};
  dirs[n * 3 + 0] = dx * inv_norm;
  dirs[n * 3 + 1] = dy * inv_norm;
  dirs[n * 3 + 2] = dz * inv_norm;
}

} // namespace

void launch_sigmoid_forward(
  const std::uint32_t count,
  const float* logits,
  float* out,
  cudaStream_t stream)
{
  if (count == 0) return;
  const dim3 grid {(count + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  sigmoid_forward_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
    count, logits, out);
}

void launch_exp_forward(
  const std::uint32_t count,
  const float* log_vals,
  float* out,
  cudaStream_t stream)
{
  if (count == 0) return;
  const dim3 grid {(count + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  exp_forward_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
    count, log_vals, out);
}

void launch_sigmoid_backward_chain(
  const std::uint32_t count,
  const float* actual,
  const float* grad_actual,
  float* grad_logit,
  cudaStream_t stream)
{
  if (count == 0) return;
  const dim3 grid {(count + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  sigmoid_backward_chain_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
    count, actual, grad_actual, grad_logit);
}

void launch_exp_backward_chain(
  const std::uint32_t count,
  const float* actual,
  const float* grad_actual,
  float* grad_log,
  cudaStream_t stream)
{
  if (count == 0) return;
  const dim3 grid {(count + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  exp_backward_chain_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
    count, actual, grad_actual, grad_log);
}

void launch_normalize_quaternions(
  const std::uint32_t N,
  float* quats,
  cudaStream_t stream)
{
  if (N == 0) return;
  const dim3 grid {(N + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  normalize_quaternions_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
    N, quats);
}

void launch_compute_view_directions(
  const std::uint32_t N,
  const float* means,
  const float cam_x,
  const float cam_y,
  const float cam_z,
  float* dirs,
  cudaStream_t stream)
{
  if (N == 0) return;
  const dim3 grid {(N + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  compute_view_directions_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
    N, means, cam_x, cam_y, cam_z, dirs);
}

} // namespace Training
