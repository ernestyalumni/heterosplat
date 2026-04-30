#include "Training/AdamOptimizer.h"
#include "DeviceBuffer.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace GoogleUnitTests::Training
{

using ::GoogleUnitTests::DeviceBuffer;

TEST(AdamOptimizer, SingleStepMatchesCPU)
{
  const std::uint32_t count {4};
  const float lr {0.001f};
  const float beta1 {0.9f};
  const float beta2 {0.999f};
  const float epsilon {1e-8f};
  const std::uint32_t step {1};

  std::vector<float> params {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> grads {0.1f, -0.2f, 0.3f, -0.4f};
  std::vector<float> m1(count, 0.0f);
  std::vector<float> m2(count, 0.0f);

  DeviceBuffer<float> d_params(count);
  DeviceBuffer<float> d_grads(count);
  DeviceBuffer<float> d_m1(count);
  DeviceBuffer<float> d_m2(count);

  d_params.copy_from_host(params);
  d_grads.copy_from_host(grads);
  d_m1.copy_from_host(m1);
  d_m2.copy_from_host(m2);

  ::Training::launch_adam_update(
    count, d_params.data(), d_grads.data(),
    d_m1.data(), d_m2.data(),
    lr, beta1, beta2, epsilon, step, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result_params {d_params.copy_to_host()};
  const auto result_m1 {d_m1.copy_to_host()};
  const auto result_m2 {d_m2.copy_to_host()};

  const float bc1 {1.0f - std::pow(beta1, 1.0f)};
  const float bc2 {1.0f - std::pow(beta2, 1.0f)};

  for (std::uint32_t i = 0; i < count; ++i)
  {
    const float expected_m {(1.0f - beta1) * grads[i]};
    const float expected_v {(1.0f - beta2) * grads[i] * grads[i]};
    const float m_hat {expected_m / bc1};
    const float v_hat {expected_v / bc2};
    const float expected_param {
      params[i] - lr * m_hat / (std::sqrt(v_hat) + epsilon)};

    EXPECT_NEAR(result_m1[i], expected_m, 1e-7f) << "m1[" << i << "]";
    EXPECT_NEAR(result_m2[i], expected_v, 1e-7f) << "m2[" << i << "]";
    EXPECT_NEAR(result_params[i], expected_param, 1e-6f)
      << "params[" << i << "]";
  }
}

TEST(AdamOptimizer, MultiStepMomentumAccumulates)
{
  const std::uint32_t count {2};
  const float lr {0.01f};
  const float beta1 {0.9f};
  const float beta2 {0.999f};
  const float epsilon {1e-8f};

  std::vector<float> h_params {5.0f, -3.0f};
  std::vector<float> h_m1(count, 0.0f);
  std::vector<float> h_m2(count, 0.0f);

  DeviceBuffer<float> d_params(count);
  DeviceBuffer<float> d_grads(count);
  DeviceBuffer<float> d_m1(count);
  DeviceBuffer<float> d_m2(count);

  d_params.copy_from_host(h_params);
  d_m1.copy_from_host(h_m1);
  d_m2.copy_from_host(h_m2);

  const std::vector<float> grads_step1 {0.5f, -0.5f};
  const std::vector<float> grads_step2 {0.3f, -0.1f};

  d_grads.copy_from_host(grads_step1);
  ::Training::launch_adam_update(
    count, d_params.data(), d_grads.data(),
    d_m1.data(), d_m2.data(), lr, beta1, beta2, epsilon, 1, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  d_grads.copy_from_host(grads_step2);
  ::Training::launch_adam_update(
    count, d_params.data(), d_grads.data(),
    d_m1.data(), d_m2.data(), lr, beta1, beta2, epsilon, 2, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result_m1 {d_m1.copy_to_host()};

  // After 2 steps: m = beta1 * (beta1*0 + (1-beta1)*g1) + (1-beta1)*g2
  //              = beta1*(1-beta1)*g1 + (1-beta1)*g2
  for (std::uint32_t i = 0; i < count; ++i)
  {
    const float expected_m {
      beta1 * (1.0f - beta1) * grads_step1[i]
      + (1.0f - beta1) * grads_step2[i]};
    EXPECT_NEAR(result_m1[i], expected_m, 1e-6f) << "m1[" << i << "]";
  }
}

TEST(AdamOptimizer, ZeroGradNoParamChange)
{
  const std::uint32_t count {3};
  const std::vector<float> params {1.0f, 2.0f, 3.0f};
  const std::vector<float> grads(count, 0.0f);
  std::vector<float> m1(count, 0.0f);
  std::vector<float> m2(count, 0.0f);

  DeviceBuffer<float> d_params(count);
  DeviceBuffer<float> d_grads(count);
  DeviceBuffer<float> d_m1(count);
  DeviceBuffer<float> d_m2(count);

  d_params.copy_from_host(params);
  d_grads.copy_from_host(grads);
  d_m1.copy_from_host(m1);
  d_m2.copy_from_host(m2);

  ::Training::launch_adam_update(
    count, d_params.data(), d_grads.data(),
    d_m1.data(), d_m2.data(),
    0.001f, 0.9f, 0.999f, 1e-8f, 1, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_params.copy_to_host()};
  for (std::uint32_t i = 0; i < count; ++i)
  {
    EXPECT_FLOAT_EQ(result[i], params[i]);
  }
}

} // namespace GoogleUnitTests::Training
