#include "Training/Activations.h"
#include "DeviceBuffer.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace GoogleUnitTests::Training
{

using ::GoogleUnitTests::DeviceBuffer;

TEST(Activations, SigmoidForwardMatchesCPU)
{
  const std::vector<float> logits {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 10.0f};
  const std::uint32_t n {static_cast<std::uint32_t>(logits.size())};

  DeviceBuffer<float> d_logits(n);
  DeviceBuffer<float> d_out(n);
  d_logits.copy_from_host(logits);

  ::Training::launch_sigmoid_forward(n, d_logits.data(), d_out.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_out.copy_to_host()};
  for (std::uint32_t i = 0; i < n; ++i)
  {
    const float expected {1.0f / (1.0f + std::exp(-logits[i]))};
    EXPECT_NEAR(result[i], expected, 1e-6f) << "index " << i;
  }
}

TEST(Activations, ExpForwardMatchesCPU)
{
  const std::vector<float> log_vals {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
  const std::uint32_t n {static_cast<std::uint32_t>(log_vals.size())};

  DeviceBuffer<float> d_log(n);
  DeviceBuffer<float> d_out(n);
  d_log.copy_from_host(log_vals);

  ::Training::launch_exp_forward(n, d_log.data(), d_out.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_out.copy_to_host()};
  for (std::uint32_t i = 0; i < n; ++i)
  {
    const float expected {std::exp(log_vals[i])};
    EXPECT_NEAR(result[i], expected, 1e-5f) << "index " << i;
  }
}

TEST(Activations, SigmoidBackwardChainRule)
{
  const std::vector<float> actual {0.1f, 0.5f, 0.9f};
  const std::vector<float> grad_actual {1.0f, 2.0f, -1.0f};
  const std::uint32_t n {3};

  DeviceBuffer<float> d_actual(n);
  DeviceBuffer<float> d_grad_actual(n);
  DeviceBuffer<float> d_grad_logit(n);
  d_actual.copy_from_host(actual);
  d_grad_actual.copy_from_host(grad_actual);

  ::Training::launch_sigmoid_backward_chain(
    n, d_actual.data(), d_grad_actual.data(), d_grad_logit.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_grad_logit.copy_to_host()};
  for (std::uint32_t i = 0; i < n; ++i)
  {
    const float expected {grad_actual[i] * actual[i] * (1.0f - actual[i])};
    EXPECT_NEAR(result[i], expected, 1e-7f) << "index " << i;
  }
}

TEST(Activations, ExpBackwardChainRule)
{
  const std::vector<float> actual {0.05f, 1.0f, 20.0f};
  const std::vector<float> grad_actual {1.0f, -0.5f, 0.1f};
  const std::uint32_t n {3};

  DeviceBuffer<float> d_actual(n);
  DeviceBuffer<float> d_grad_actual(n);
  DeviceBuffer<float> d_grad_log(n);
  d_actual.copy_from_host(actual);
  d_grad_actual.copy_from_host(grad_actual);

  ::Training::launch_exp_backward_chain(
    n, d_actual.data(), d_grad_actual.data(), d_grad_log.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_grad_log.copy_to_host()};
  for (std::uint32_t i = 0; i < n; ++i)
  {
    const float expected {grad_actual[i] * actual[i]};
    EXPECT_NEAR(result[i], expected, 1e-7f) << "index " << i;
  }
}

TEST(Activations, NormalizeQuaternionsProducesUnitNorm)
{
  const std::vector<float> quats {
    2.0f, 0.0f, 0.0f, 0.0f,  // norm 2
    1.0f, 1.0f, 1.0f, 1.0f,  // norm 2
    0.0f, 3.0f, 4.0f, 0.0f,  // norm 5
  };
  const std::uint32_t N {3};

  DeviceBuffer<float> d_quats(N * 4);
  d_quats.copy_from_host(quats);

  ::Training::launch_normalize_quaternions(N, d_quats.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_quats.copy_to_host()};
  for (std::uint32_t n = 0; n < N; ++n)
  {
    const float w {result[n * 4 + 0]};
    const float x {result[n * 4 + 1]};
    const float y {result[n * 4 + 2]};
    const float z {result[n * 4 + 3]};
    const float norm {std::sqrt(w*w + x*x + y*y + z*z)};
    EXPECT_NEAR(norm, 1.0f, 1e-6f) << "quaternion " << n;
  }

  EXPECT_NEAR(result[0], 1.0f, 1e-6f);
  EXPECT_NEAR(result[1], 0.0f, 1e-6f);
}

TEST(Activations, ComputeViewDirectionsAreNormalized)
{
  const std::vector<float> means {
    1.0f, 0.0f, 0.0f,
    0.0f, 2.0f, 0.0f,
    0.0f, 0.0f, 5.0f,
  };
  const std::uint32_t N {3};
  const float cam_x {0.0f}, cam_y {0.0f}, cam_z {0.0f};

  DeviceBuffer<float> d_means(N * 3);
  DeviceBuffer<float> d_dirs(N * 3);
  d_means.copy_from_host(means);

  ::Training::launch_compute_view_directions(
    N, d_means.data(), cam_x, cam_y, cam_z, d_dirs.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto dirs {d_dirs.copy_to_host()};
  for (std::uint32_t n = 0; n < N; ++n)
  {
    const float dx {dirs[n * 3 + 0]};
    const float dy {dirs[n * 3 + 1]};
    const float dz {dirs[n * 3 + 2]};
    const float norm {std::sqrt(dx*dx + dy*dy + dz*dz)};
    EXPECT_NEAR(norm, 1.0f, 1e-5f) << "direction " << n;
  }

  // Direction from (1,0,0) to camera at origin = (-1,0,0)
  EXPECT_NEAR(dirs[0], -1.0f, 1e-6f);
  EXPECT_NEAR(dirs[1],  0.0f, 1e-6f);
  EXPECT_NEAR(dirs[2],  0.0f, 1e-6f);
}

} // namespace GoogleUnitTests::Training
