#ifndef TRAINING_ADAM_OPTIMIZER_H
#define TRAINING_ADAM_OPTIMIZER_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Training
{

/// Per-element Adam update on GPU.
///
/// Implements: m = beta1*m + (1-beta1)*grad
///             v = beta2*v + (1-beta2)*grad^2
///             m_hat = m / (1 - beta1^t)
///             v_hat = v / (1 - beta2^t)
///             param -= lr * m_hat / (sqrt(v_hat) + eps)
///
/// \param count       Number of elements to update.
/// \param params      [count] — parameter values (read-write).
/// \param grads       [count] — gradients (read-only).
/// \param first_moments  [count] — Adam m state (read-write).
/// \param second_moments [count] — Adam v state (read-write).
/// \param learning_rate  Scalar learning rate.
/// \param beta1       First moment decay (default 0.9).
/// \param beta2       Second moment decay (default 0.999).
/// \param epsilon     Numerical stability (default 1e-8).
/// \param step        Current optimizer step (1-indexed, for bias correction).
/// \param stream      CUDA stream.
void launch_adam_update(
  std::uint32_t count,
  float* params,
  const float* grads,
  float* first_moments,
  float* second_moments,
  float learning_rate,
  float beta1,
  float beta2,
  float epsilon,
  std::uint32_t step,
  cudaStream_t stream);

} // namespace Training

#endif // TRAINING_ADAM_OPTIMIZER_H
