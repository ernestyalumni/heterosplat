#include "Normalize/Convention.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace GoogleUnitTests::Normalize
{

TEST(Convention, DetectZUpFromFlatTerrain)
{
  // Flat terrain: large spread in X/Y, small spread in Z, positive Z mean
  const std::uint32_t N {100};
  std::vector<float> means(N * 3);
  for (std::uint32_t i = 0; i < N; ++i)
  {
    means[i * 3 + 0] = static_cast<float>(i % 10) - 5.0f;
    means[i * 3 + 1] = static_cast<float>(i / 10) - 5.0f;
    means[i * 3 + 2] = 1.0f + 0.1f * (i % 3);
  }

  const auto axis {::Normalize::detect_up_axis(means.data(), N)};
  EXPECT_EQ(axis, ::Normalize::UpAxis::z_up);
}

TEST(Convention, DetectYUpFromVerticalScene)
{
  // Vertical scene: large spread in X/Z, small spread in Y, positive Y mean
  const std::uint32_t N {100};
  std::vector<float> means(N * 3);
  for (std::uint32_t i = 0; i < N; ++i)
  {
    means[i * 3 + 0] = static_cast<float>(i % 10) - 5.0f;
    means[i * 3 + 1] = 2.0f + 0.05f * (i % 5);
    means[i * 3 + 2] = static_cast<float>(i / 10) - 5.0f;
  }

  const auto axis {::Normalize::detect_up_axis(means.data(), N)};
  EXPECT_EQ(axis, ::Normalize::UpAxis::y_up);
}

TEST(Convention, SceneExtentComputation)
{
  const std::uint32_t N {4};
  const std::vector<float> means {
    -1, -2, -3,
     1,  2,  3,
     0,  0,  0,
     0.5f, 0.5f, 0.5f};

  const auto extent {::Normalize::compute_scene_extent(means.data(), N)};

  EXPECT_NEAR(extent.extent[0], 2.0f, 1e-5f);
  EXPECT_NEAR(extent.extent[1], 4.0f, 1e-5f);
  EXPECT_NEAR(extent.extent[2], 6.0f, 1e-5f);
  EXPECT_NEAR(extent.max_extent, 6.0f, 1e-5f);
  EXPECT_NEAR(extent.centroid[0], 0.125f, 1e-5f);
  EXPECT_NEAR(extent.centroid[1], 0.125f, 1e-5f);
  EXPECT_NEAR(extent.centroid[2], 0.125f, 1e-5f);
}

TEST(Convention, RotationToZUpFromYUp)
{
  const auto R {::Normalize::rotation_to_z_up(::Normalize::UpAxis::y_up)};

  // Apply to (0, 1, 0) which is "up" in Y-up → should become (0, 0, 1) in Z-up
  const float x {R[0] * 0 + R[1] * 1 + R[2] * 0};
  const float y {R[3] * 0 + R[4] * 1 + R[5] * 0};
  const float z {R[6] * 0 + R[7] * 1 + R[8] * 0};

  EXPECT_NEAR(x, 0.0f, 1e-5f);
  EXPECT_NEAR(y, 0.0f, 1e-5f);
  EXPECT_NEAR(z, 1.0f, 1e-5f);
}

TEST(Convention, RotationToZUpFromZUpIsIdentity)
{
  const auto R {::Normalize::rotation_to_z_up(::Normalize::UpAxis::z_up)};

  const std::array<float, 9> identity {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int i = 0; i < 9; ++i)
    EXPECT_NEAR(R[i], identity[i], 1e-7f);
}

TEST(Convention, NormalizationCentersAndScales)
{
  const std::uint32_t N {4};
  const std::vector<float> means {
    0, 0, 0,
    10, 0, 0,
    0, 10, 0,
    10, 10, 0};

  const auto xform {::Normalize::compute_normalization_transform(
    means.data(), N, ::Normalize::UpAxis::z_up)};

  // Max extent = 10, so scale = 2/10 = 0.2
  EXPECT_NEAR(xform.scale, 0.2f, 1e-5f);

  // After transform, centroid (5,5,0) should map to origin
  const float cx {5.0f}, cy {5.0f}, cz {0.0f};
  const auto& R {xform.rotation};
  const float tx {xform.scale * (R[0]*cx + R[1]*cy + R[2]*cz)
    + xform.translation[0]};
  const float ty {xform.scale * (R[3]*cx + R[4]*cy + R[5]*cz)
    + xform.translation[1]};
  const float tz {xform.scale * (R[6]*cx + R[7]*cy + R[8]*cz)
    + xform.translation[2]};

  EXPECT_NEAR(tx, 0.0f, 1e-5f);
  EXPECT_NEAR(ty, 0.0f, 1e-5f);
  EXPECT_NEAR(tz, 0.0f, 1e-5f);
}

TEST(Convention, ParseConventionString)
{
  std::vector<float> dummy(30, 0.0f);
  auto c1 {::Normalize::parse_convention_string("y-up", dummy.data(), 10)};
  EXPECT_EQ(c1.up_axis, ::Normalize::UpAxis::y_up);

  auto c2 {::Normalize::parse_convention_string("z-up", dummy.data(), 10)};
  EXPECT_EQ(c2.up_axis, ::Normalize::UpAxis::z_up);
}

} // namespace GoogleUnitTests::Normalize
