#include "DeviceBuffer.h"
#include "Viewer/ColormapDebug.h"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace GoogleUnitTests::Viewer
{

using ::GoogleUnitTests::DeviceBuffer;

constexpr float kC0 {0.28209479177387814f};

TEST(ColormapDebug, MinBoundProducesLowEndColor)
{
  const std::uint32_t N {1};
  const std::uint32_t K {1};
  const std::vector<float> means {0.0f, 0.0f, 0.0f};
  std::vector<float> sh(N * K * 3, 0.0f);

  DeviceBuffer<float> d_means(3);
  DeviceBuffer<float> d_sh(3);

  d_means.copy_from_host(means);
  d_sh.copy_from_host(sh);

  ::Viewer::launch_colormap_per_axis(
    N, d_means.data(),
    0.0f, 10.0f, 0.0f, 10.0f, 0.0f, 10.0f,
    0, K, d_sh.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_sh.copy_to_host()};
  // t=0.0 → first segment: r=0.18, g=0.05, b=0.53
  EXPECT_NEAR(result[0] * kC0, 0.18f, 0.01f);
  EXPECT_NEAR(result[1] * kC0, 0.05f, 0.01f);
  EXPECT_NEAR(result[2] * kC0, 0.53f, 0.01f);
}

TEST(ColormapDebug, MaxBoundProducesHighEndColor)
{
  const std::uint32_t N {1};
  const std::uint32_t K {1};
  const std::vector<float> means {10.0f, 5.0f, 5.0f};
  std::vector<float> sh(N * K * 3, 0.0f);

  DeviceBuffer<float> d_means(3);
  DeviceBuffer<float> d_sh(3);

  d_means.copy_from_host(means);
  d_sh.copy_from_host(sh);

  ::Viewer::launch_colormap_per_axis(
    N, d_means.data(),
    0.0f, 10.0f, 0.0f, 10.0f, 0.0f, 10.0f,
    0, K, d_sh.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_sh.copy_to_host()};
  // t=1.0 → last segment end: r=0.60, g=0.15, b=0.05
  EXPECT_NEAR(result[0] * kC0, 0.60f, 0.01f);
  EXPECT_NEAR(result[1] * kC0, 0.15f, 0.01f);
  EXPECT_NEAR(result[2] * kC0, 0.05f, 0.01f);
}

TEST(ColormapDebug, MidpointColor)
{
  const std::uint32_t N {1};
  const std::uint32_t K {1};
  const std::vector<float> means {5.0f, 5.0f, 5.0f};
  std::vector<float> sh(N * K * 3, 0.0f);

  DeviceBuffer<float> d_means(3);
  DeviceBuffer<float> d_sh(3);

  d_means.copy_from_host(means);
  d_sh.copy_from_host(sh);

  ::Viewer::launch_colormap_per_axis(
    N, d_means.data(),
    0.0f, 10.0f, 0.0f, 10.0f, 0.0f, 10.0f,
    0, K, d_sh.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_sh.copy_to_host()};
  // t=0.5 → boundary of second/third segment: r=0.93, g=0.90, b=0.35
  EXPECT_NEAR(result[0] * kC0, 0.93f, 0.01f);
  EXPECT_NEAR(result[1] * kC0, 0.90f, 0.01f);
  EXPECT_NEAR(result[2] * kC0, 0.35f, 0.01f);
}

TEST(ColormapDebug, AxisSelectionYZ)
{
  const std::uint32_t N {1};
  const std::uint32_t K {1};
  // Point at (5, 0, 10) — axis 1 (Y) gives t=0, axis 2 (Z) gives t=1
  const std::vector<float> means {5.0f, 0.0f, 10.0f};

  DeviceBuffer<float> d_means(3);
  DeviceBuffer<float> d_sh_y(3);
  DeviceBuffer<float> d_sh_z(3);
  std::vector<float> sh(3, 0.0f);

  d_means.copy_from_host(means);
  d_sh_y.copy_from_host(sh);
  d_sh_z.copy_from_host(sh);

  ::Viewer::launch_colormap_per_axis(
    N, d_means.data(),
    0.0f, 10.0f, 0.0f, 10.0f, 0.0f, 10.0f,
    1, K, d_sh_y.data(), nullptr);

  ::Viewer::launch_colormap_per_axis(
    N, d_means.data(),
    0.0f, 10.0f, 0.0f, 10.0f, 0.0f, 10.0f,
    2, K, d_sh_z.data(), nullptr);

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result_y {d_sh_y.copy_to_host()};
  const auto result_z {d_sh_z.copy_to_host()};

  // Y axis → t=0 (low end), Z axis → t=1 (high end)
  // These should be different colors
  EXPECT_NEAR(result_y[0] * kC0, 0.18f, 0.01f); // low-end R
  EXPECT_NEAR(result_z[0] * kC0, 0.60f, 0.01f); // high-end R
}

TEST(ColormapDebug, HigherOrderSHZeroed)
{
  const std::uint32_t N {1};
  const std::uint32_t K {4}; // SH degree 1
  std::vector<float> sh(N * K * 3, 1.0f); // Fill with 1s

  const std::vector<float> means {5.0f, 5.0f, 5.0f};

  DeviceBuffer<float> d_means(3);
  DeviceBuffer<float> d_sh(N * K * 3);

  d_means.copy_from_host(means);
  d_sh.copy_from_host(sh);

  ::Viewer::launch_colormap_per_axis(
    N, d_means.data(),
    0.0f, 10.0f, 0.0f, 10.0f, 0.0f, 10.0f,
    2, K, d_sh.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_sh.copy_to_host()};

  // DC should be non-zero (colormap value)
  EXPECT_GT(std::abs(result[0]), 0.1f);

  // Higher-order SH (indices 3..11) should be zeroed
  for (std::uint32_t i = 3; i < K * 3; ++i)
  {
    EXPECT_NEAR(result[i], 0.0f, 1e-7f) << "index " << i;
  }
}

TEST(ColormapDebug, MultipleGaussians)
{
  const std::uint32_t N {3};
  const std::uint32_t K {1};
  const std::vector<float> means {0, 0, 0, 5, 5, 5, 10, 10, 10};
  std::vector<float> sh(N * K * 3, 0.0f);

  DeviceBuffer<float> d_means(N * 3);
  DeviceBuffer<float> d_sh(N * K * 3);

  d_means.copy_from_host(means);
  d_sh.copy_from_host(sh);

  ::Viewer::launch_colormap_per_axis(
    N, d_means.data(),
    0.0f, 10.0f, 0.0f, 10.0f, 0.0f, 10.0f,
    0, K, d_sh.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_sh.copy_to_host()};

  // Three different t values → three different colors
  // t=0 → r≈0.18, t=0.5 → r≈0.93, t=1.0 → r≈0.60
  const float r0 {result[0] * kC0};
  const float r1 {result[3] * kC0};
  const float r2 {result[6] * kC0};

  EXPECT_NE(r0, r1);
  EXPECT_NE(r1, r2);
}

} // namespace GoogleUnitTests::Viewer
