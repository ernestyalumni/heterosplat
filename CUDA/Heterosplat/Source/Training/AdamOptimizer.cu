#include "Training/AdamOptimizer.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace Training
{

namespace
{

constexpr std::uint32_t kThreadsPerBlock {256};

__global__ void adam_update_kernel(
  const std::uint32_t count,
  float* __restrict__ params,
  const float* __restrict__ grads,
  float* __restrict__ first_moments,
  float* __restrict__ second_moments,
  const float learning_rate,
  const float beta1,
  const float beta2,
  const float epsilon,
  const float bias_correction1,
  const float bias_correction2)
{
  const std::uint32_t idx {blockIdx.x * blockDim.x + threadIdx.x};
  if (idx >= count)
  {
    return;
  }

  const float grad {grads[idx]};

  float m {beta1 * first_moments[idx] + (1.0f - beta1) * grad};
  float v {beta2 * second_moments[idx] + (1.0f - beta2) * grad * grad};

  first_moments[idx] = m;
  second_moments[idx] = v;

  const float m_hat {m / bias_correction1};
  const float v_hat {v / bias_correction2};

  params[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
}

} // namespace

void launch_adam_update(
  const std::uint32_t count,
  float* params,
  const float* grads,
  float* first_moments,
  float* second_moments,
  const float learning_rate,
  const float beta1,
  const float beta2,
  const float epsilon,
  const std::uint32_t step,
  cudaStream_t stream)
{
  assert(params != nullptr);
  assert(grads != nullptr);
  assert(first_moments != nullptr);
  assert(second_moments != nullptr);
  assert(step >= 1);

  if (count == 0)
  {
    return;
  }

  const float bias_correction1 {1.0f - std::pow(beta1, static_cast<float>(step))};
  const float bias_correction2 {1.0f - std::pow(beta2, static_cast<float>(step))};

  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {(count + kThreadsPerBlock - 1u) / kThreadsPerBlock};

  adam_update_kernel<<<grid, threads, 0, stream>>>(
    count, params, grads, first_moments, second_moments,
    learning_rate, beta1, beta2, epsilon,
    bias_correction1, bias_correction2);
}

} // namespace Training
