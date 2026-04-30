#include "Training/ImageLoss.h"
#include "DeviceBuffer.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace GoogleUnitTests::Training
{

using ::GoogleUnitTests::DeviceBuffer;

TEST(ImageLoss, L1ForwardSinglePixel)
{
  const std::uint32_t num_pixels {1};
  const std::vector<float> rendered {0.8f, 0.2f, 0.5f};
  const std::vector<float> target   {0.3f, 0.7f, 0.5f};

  DeviceBuffer<float> d_rendered(3);
  DeviceBuffer<float> d_target(3);
  DeviceBuffer<float> d_loss(1);

  d_rendered.copy_from_host(rendered);
  d_target.copy_from_host(target);

  ::Training::launch_l1_loss(
    num_pixels, d_rendered.data(), d_target.data(),
    d_loss.data(), nullptr, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_loss.copy_to_host()};

  // L1 = (|0.5| + |0.5| + |0.0|) / 3 = 1.0 / 3
  EXPECT_NEAR(result[0], 1.0f / 3.0f, 1e-5f);
}

TEST(ImageLoss, L1ForwardMultiplePixels)
{
  const std::uint32_t num_pixels {2};
  const std::vector<float> rendered {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  const std::vector<float> target   {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

  DeviceBuffer<float> d_rendered(6);
  DeviceBuffer<float> d_target(6);
  DeviceBuffer<float> d_loss(1);

  d_rendered.copy_from_host(rendered);
  d_target.copy_from_host(target);

  ::Training::launch_l1_loss(
    num_pixels, d_rendered.data(), d_target.data(),
    d_loss.data(), nullptr, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_loss.copy_to_host()};

  // 4 elements differ by 1.0 each, 2 are zero → sum = 4.0, count = 6
  EXPECT_NEAR(result[0], 4.0f / 6.0f, 1e-5f);
}

TEST(ImageLoss, L1IdenticalImagesGiveZeroLoss)
{
  const std::uint32_t num_pixels {4};
  const std::vector<float> image(num_pixels * 3, 0.5f);

  DeviceBuffer<float> d_rendered(num_pixels * 3);
  DeviceBuffer<float> d_target(num_pixels * 3);
  DeviceBuffer<float> d_loss(1);

  d_rendered.copy_from_host(image);
  d_target.copy_from_host(image);

  ::Training::launch_l1_loss(
    num_pixels, d_rendered.data(), d_target.data(),
    d_loss.data(), nullptr, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_loss.copy_to_host()};
  EXPECT_NEAR(result[0], 0.0f, 1e-7f);
}

TEST(ImageLoss, L1BackwardSignGradient)
{
  const std::uint32_t num_pixels {2};
  const std::vector<float> rendered {0.8f, 0.2f, 0.5f, 0.1f, 0.9f, 0.5f};
  const std::vector<float> target   {0.3f, 0.7f, 0.5f, 0.6f, 0.4f, 0.5f};

  DeviceBuffer<float> d_rendered(6);
  DeviceBuffer<float> d_target(6);
  DeviceBuffer<float> d_loss(1);
  DeviceBuffer<float> d_grad(6);

  d_rendered.copy_from_host(rendered);
  d_target.copy_from_host(target);

  ::Training::launch_l1_loss(
    num_pixels, d_rendered.data(), d_target.data(),
    d_loss.data(), d_grad.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto grad {d_grad.copy_to_host()};
  const float inv_count {1.0f / 6.0f};

  for (std::uint32_t i = 0; i < 6; ++i)
  {
    const float diff {rendered[i] - target[i]};
    const float expected_sign {
      diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f)};
    EXPECT_NEAR(grad[i], expected_sign * inv_count, 1e-7f)
      << "index " << i;
  }
}

TEST(ImageLoss, L1BackwardNullGradSkipsComputation)
{
  const std::uint32_t num_pixels {2};
  const std::vector<float> rendered(6, 0.5f);
  const std::vector<float> target(6, 0.3f);

  DeviceBuffer<float> d_rendered(6);
  DeviceBuffer<float> d_target(6);
  DeviceBuffer<float> d_loss(1);

  d_rendered.copy_from_host(rendered);
  d_target.copy_from_host(target);

  ::Training::launch_l1_loss(
    num_pixels, d_rendered.data(), d_target.data(),
    d_loss.data(), nullptr, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_loss.copy_to_host()};
  EXPECT_NEAR(result[0], 0.2f, 1e-5f);
}

} // namespace GoogleUnitTests::Training
