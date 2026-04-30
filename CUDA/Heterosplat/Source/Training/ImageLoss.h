#ifndef TRAINING_IMAGE_LOSS_H
#define TRAINING_IMAGE_LOSS_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Training
{

/// Per-pixel L1 loss between rendered and target images.
///
/// Forward: loss = (1 / (H*W*3)) * sum |rendered - target|
///
/// Backward: grad_rendered[i] = sign(rendered[i] - target[i]) / (H*W*3)
///
/// \param num_pixels   H * W (total pixels per image).
/// \param rendered     [num_pixels * 3] — rendered RGB.
/// \param target       [num_pixels * 3] — ground truth RGB.
/// \param loss         Scalar output (device pointer to a single float).
/// \param grad_rendered [num_pixels * 3] — output gradient w.r.t. rendered.
///                      Pass nullptr to skip gradient computation.
/// \param stream       CUDA stream.
void launch_l1_loss(
  std::uint32_t num_pixels,
  const float* rendered,
  const float* target,
  float* loss,
  float* grad_rendered,
  cudaStream_t stream);

} // namespace Training

#endif // TRAINING_IMAGE_LOSS_H
