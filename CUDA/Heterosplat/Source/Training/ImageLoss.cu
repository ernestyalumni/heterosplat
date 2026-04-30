#include "Training/ImageLoss.h"

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

namespace Training
{

namespace
{

constexpr std::uint32_t kThreadsPerBlock {256};

__global__ void l1_loss_forward_kernel(
  const std::uint32_t count,
  const float* __restrict__ rendered,
  const float* __restrict__ target,
  float* __restrict__ partial_sums,
  const float inv_count)
{
  extern __shared__ float shmem[];

  const std::uint32_t tid {threadIdx.x};
  const std::uint32_t idx {blockIdx.x * blockDim.x + tid};

  float val {0.0f};
  if (idx < count)
  {
    val = fabsf(rendered[idx] - target[idx]);
  }

  shmem[tid] = val;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      shmem[tid] += shmem[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    atomicAdd(partial_sums, shmem[0] * inv_count);
  }
}

__global__ void l1_loss_backward_kernel(
  const std::uint32_t count,
  const float* __restrict__ rendered,
  const float* __restrict__ target,
  float* __restrict__ grad_rendered,
  const float inv_count)
{
  const std::uint32_t idx {blockIdx.x * blockDim.x + threadIdx.x};
  if (idx >= count)
  {
    return;
  }

  const float diff {rendered[idx] - target[idx]};
  grad_rendered[idx] = (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f))
                       * inv_count;
}

} // namespace

void launch_l1_loss(
  const std::uint32_t num_pixels,
  const float* rendered,
  const float* target,
  float* loss,
  float* grad_rendered,
  cudaStream_t stream)
{
  assert(rendered != nullptr);
  assert(target != nullptr);
  assert(loss != nullptr);

  const std::uint32_t count {num_pixels * 3};

  if (count == 0)
  {
    cudaMemsetAsync(loss, 0, sizeof(float), stream);
    return;
  }

  const float inv_count {1.0f / static_cast<float>(count)};

  cudaMemsetAsync(loss, 0, sizeof(float), stream);

  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {(count + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  const std::size_t shmem_size {kThreadsPerBlock * sizeof(float)};

  l1_loss_forward_kernel<<<grid, threads, shmem_size, stream>>>(
    count, rendered, target, loss, inv_count);

  if (grad_rendered != nullptr)
  {
    l1_loss_backward_kernel<<<grid, threads, 0, stream>>>(
      count, rendered, target, grad_rendered, inv_count);
  }
}

} // namespace Training
