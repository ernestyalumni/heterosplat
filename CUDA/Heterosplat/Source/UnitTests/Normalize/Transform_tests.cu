#include "DeviceBuffer.h"
#include "Normalize/Convention.h"
#include "Normalize/Transform.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace GoogleUnitTests::Normalize
{

using ::GoogleUnitTests::DeviceBuffer;

TEST(Transform, IdentitySimilarityIsNoOp)
{
  const std::uint32_t N {4};
  const std::vector<float> means {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<float> quats {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
  const std::vector<float> log_scales(N * 3, 0.5f);
  const std::vector<float> rotation {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::vector<float> translation {0, 0, 0};
  const float scale {1.0f};

  DeviceBuffer<float> d_means(N * 3);
  DeviceBuffer<float> d_quats(N * 4);
  DeviceBuffer<float> d_log_scales(N * 3);
  DeviceBuffer<float> d_rotation(9);
  DeviceBuffer<float> d_translation(3);

  d_means.copy_from_host(means);
  d_quats.copy_from_host(quats);
  d_log_scales.copy_from_host(log_scales);
  d_rotation.copy_from_host(rotation);
  d_translation.copy_from_host(translation);

  ::Normalize::launch_apply_similarity_transform(
    N, d_rotation.data(), scale, d_translation.data(),
    d_means.data(), d_quats.data(), d_log_scales.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result_means {d_means.copy_to_host()};
  const auto result_quats {d_quats.copy_to_host()};
  const auto result_scales {d_log_scales.copy_to_host()};

  for (std::uint32_t i = 0; i < N * 3; ++i)
    EXPECT_NEAR(result_means[i], means[i], 1e-5f);
  for (std::uint32_t i = 0; i < N * 4; ++i)
    EXPECT_NEAR(result_quats[i], quats[i], 1e-5f);
  for (std::uint32_t i = 0; i < N * 3; ++i)
    EXPECT_NEAR(result_scales[i], log_scales[i], 1e-5f);
}

TEST(Transform, TranslationOnly)
{
  const std::uint32_t N {2};
  const std::vector<float> means {0, 0, 0, 1, 1, 1};
  const std::vector<float> quats {1, 0, 0, 0, 1, 0, 0, 0};
  const std::vector<float> log_scales(N * 3, 0.0f);
  const std::vector<float> rotation {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::vector<float> translation {5, -3, 2};
  const float scale {1.0f};

  DeviceBuffer<float> d_means(N * 3);
  DeviceBuffer<float> d_quats(N * 4);
  DeviceBuffer<float> d_log_scales(N * 3);
  DeviceBuffer<float> d_rotation(9);
  DeviceBuffer<float> d_translation(3);

  d_means.copy_from_host(means);
  d_quats.copy_from_host(quats);
  d_log_scales.copy_from_host(log_scales);
  d_rotation.copy_from_host(rotation);
  d_translation.copy_from_host(translation);

  ::Normalize::launch_apply_similarity_transform(
    N, d_rotation.data(), scale, d_translation.data(),
    d_means.data(), d_quats.data(), d_log_scales.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result_means {d_means.copy_to_host()};
  EXPECT_NEAR(result_means[0], 5.0f, 1e-5f);
  EXPECT_NEAR(result_means[1], -3.0f, 1e-5f);
  EXPECT_NEAR(result_means[2], 2.0f, 1e-5f);
  EXPECT_NEAR(result_means[3], 6.0f, 1e-5f);
  EXPECT_NEAR(result_means[4], -2.0f, 1e-5f);
  EXPECT_NEAR(result_means[5], 3.0f, 1e-5f);
}

TEST(Transform, UniformScaleAffectsMeansAndLogScales)
{
  const std::uint32_t N {1};
  const std::vector<float> means {2, 4, 6};
  const std::vector<float> quats {1, 0, 0, 0};
  const std::vector<float> log_scales {-1.0f, 0.0f, 1.0f};
  const std::vector<float> rotation {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::vector<float> translation {0, 0, 0};
  const float scale {3.0f};

  DeviceBuffer<float> d_means(3);
  DeviceBuffer<float> d_quats(4);
  DeviceBuffer<float> d_log_scales(3);
  DeviceBuffer<float> d_rotation(9);
  DeviceBuffer<float> d_translation(3);

  d_means.copy_from_host(means);
  d_quats.copy_from_host(quats);
  d_log_scales.copy_from_host(log_scales);
  d_rotation.copy_from_host(rotation);
  d_translation.copy_from_host(translation);

  ::Normalize::launch_apply_similarity_transform(
    N, d_rotation.data(), scale, d_translation.data(),
    d_means.data(), d_quats.data(), d_log_scales.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result_means {d_means.copy_to_host()};
  const auto result_scales {d_log_scales.copy_to_host()};

  EXPECT_NEAR(result_means[0], 6.0f, 1e-5f);
  EXPECT_NEAR(result_means[1], 12.0f, 1e-5f);
  EXPECT_NEAR(result_means[2], 18.0f, 1e-5f);

  const float log3 {std::log(3.0f)};
  EXPECT_NEAR(result_scales[0], -1.0f + log3, 1e-5f);
  EXPECT_NEAR(result_scales[1], 0.0f + log3, 1e-5f);
  EXPECT_NEAR(result_scales[2], 1.0f + log3, 1e-5f);
}

TEST(Transform, Rotation90AroundZ)
{
  const std::uint32_t N {1};
  const std::vector<float> means {1, 0, 0};
  // Identity quaternion
  const std::vector<float> quats {1, 0, 0, 0};
  const std::vector<float> log_scales {0, 0, 0};
  // 90° around Z: [[0,-1,0],[1,0,0],[0,0,1]]
  const std::vector<float> rotation {0, -1, 0, 1, 0, 0, 0, 0, 1};
  const std::vector<float> translation {0, 0, 0};
  const float scale {1.0f};

  DeviceBuffer<float> d_means(3);
  DeviceBuffer<float> d_quats(4);
  DeviceBuffer<float> d_log_scales(3);
  DeviceBuffer<float> d_rotation(9);
  DeviceBuffer<float> d_translation(3);

  d_means.copy_from_host(means);
  d_quats.copy_from_host(quats);
  d_log_scales.copy_from_host(log_scales);
  d_rotation.copy_from_host(rotation);
  d_translation.copy_from_host(translation);

  ::Normalize::launch_apply_similarity_transform(
    N, d_rotation.data(), scale, d_translation.data(),
    d_means.data(), d_quats.data(), d_log_scales.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result_means {d_means.copy_to_host()};
  // (1,0,0) rotated 90° around Z = (0,1,0)
  EXPECT_NEAR(result_means[0], 0.0f, 1e-5f);
  EXPECT_NEAR(result_means[1], 1.0f, 1e-5f);
  EXPECT_NEAR(result_means[2], 0.0f, 1e-5f);

  const auto result_quats {d_quats.copy_to_host()};
  // 90° around Z quaternion: (cos(45°), 0, 0, sin(45°))
  const float c {static_cast<float>(std::cos(M_PI / 4.0))};
  const float s {static_cast<float>(std::sin(M_PI / 4.0))};
  // Result should be R_quat * identity = R_quat
  EXPECT_NEAR(result_quats[0], c, 1e-4f);
  EXPECT_NEAR(result_quats[1], 0.0f, 1e-4f);
  EXPECT_NEAR(result_quats[2], 0.0f, 1e-4f);
  EXPECT_NEAR(result_quats[3], s, 1e-4f);
}

TEST(Transform, HomogeneousIdentityIsNoOp)
{
  const std::uint32_t N {3};
  const std::vector<float> means {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<float> identity {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1};

  DeviceBuffer<float> d_means(N * 3);
  DeviceBuffer<float> d_xform(16);

  d_means.copy_from_host(means);
  d_xform.copy_from_host(identity);

  ::Normalize::launch_apply_homogeneous_transform_means(
    N, d_xform.data(), d_means.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_means.copy_to_host()};
  for (std::uint32_t i = 0; i < N * 3; ++i)
    EXPECT_NEAR(result[i], means[i], 1e-5f);
}

TEST(Transform, HomogeneousTranslation)
{
  const std::uint32_t N {1};
  const std::vector<float> means {1, 2, 3};
  const std::vector<float> xform {
    1, 0, 0, 10,
    0, 1, 0, 20,
    0, 0, 1, 30,
    0, 0, 0, 1};

  DeviceBuffer<float> d_means(3);
  DeviceBuffer<float> d_xform(16);

  d_means.copy_from_host(means);
  d_xform.copy_from_host(xform);

  ::Normalize::launch_apply_homogeneous_transform_means(
    N, d_xform.data(), d_means.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto result {d_means.copy_to_host()};
  EXPECT_NEAR(result[0], 11.0f, 1e-5f);
  EXPECT_NEAR(result[1], 22.0f, 1e-5f);
  EXPECT_NEAR(result[2], 33.0f, 1e-5f);
}

} // namespace GoogleUnitTests::Normalize
