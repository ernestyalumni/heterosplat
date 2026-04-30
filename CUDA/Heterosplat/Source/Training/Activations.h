#ifndef TRAINING_ACTIVATIONS_H
#define TRAINING_ACTIVATIONS_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Training
{

void launch_sigmoid_forward(
  std::uint32_t count,
  const float* logits,
  float* out,
  cudaStream_t stream);

void launch_exp_forward(
  std::uint32_t count,
  const float* log_vals,
  float* out,
  cudaStream_t stream);

void launch_sigmoid_backward_chain(
  std::uint32_t count,
  const float* actual,
  const float* grad_actual,
  float* grad_logit,
  cudaStream_t stream);

void launch_exp_backward_chain(
  std::uint32_t count,
  const float* actual,
  const float* grad_actual,
  float* grad_log,
  cudaStream_t stream);

void launch_normalize_quaternions(
  std::uint32_t N,
  float* quats,
  cudaStream_t stream);

void launch_compute_view_directions(
  std::uint32_t N,
  const float* means,
  float cam_x,
  float cam_y,
  float cam_z,
  float* dirs,
  cudaStream_t stream);

} // namespace Training

#endif // TRAINING_ACTIVATIONS_H
