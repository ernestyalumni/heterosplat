#include "DeviceBuffer.h"
#include "Kernels/Heterosplat/ProjectionEWA3DGSFused.h"
#include "OracleFixture.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using GoogleUnitTests::DeviceBuffer;
using Kernels::Heterosplat::launch_projection_ewa_3dgs_fused_forward;
using GoogleUnitTests::OracleFixture::fixture_path;
using GoogleUnitTests::OracleFixture::load_floats;
using GoogleUnitTests::OracleFixture::load_int32s;
using GoogleUnitTests::OracleFixture::load_uint32;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

namespace
{

void expect_close(
  const std::vector<float>& got,
  const std::vector<float>& expected,
  const float absolute_tolerance,
  const float relative_tolerance,
  const char* label)
{
  ASSERT_EQ(got.size(), expected.size()) << label;
  for (std::size_t i {0}; i < got.size(); ++i)
  {
    const float diff {std::abs(got[i] - expected[i])};
    const float tolerance {
      absolute_tolerance + relative_tolerance * std::abs(expected[i])};
    EXPECT_LE(diff, tolerance) <<
      label << "[" << i << "] got=" << got[i] << " expected=" << expected[i];
  }
}

} // namespace

//------------------------------------------------------------------------------
/// Oracle comparison against gsplat-Python's `fully_fused_projection()`.
/// Pinhole camera, quat+scale path (no covars), no opacities.
//------------------------------------------------------------------------------
TEST(ProjectionEWA3DGSFusedOracle, ForwardMatchesGsplatPython)
{
  const std::uint32_t B {
    load_uint32(fixture_path("ProjectionEWA3DGSFused", "B.bin"))};
  const std::uint32_t C {
    load_uint32(fixture_path("ProjectionEWA3DGSFused", "C.bin"))};
  const std::uint32_t N {
    load_uint32(fixture_path("ProjectionEWA3DGSFused", "N.bin"))};
  const std::uint32_t image_width {
    load_uint32(fixture_path("ProjectionEWA3DGSFused", "image_width.bin"))};
  const std::uint32_t image_height {
    load_uint32(fixture_path("ProjectionEWA3DGSFused", "image_height.bin"))};

  const auto h_means {
    load_floats(fixture_path("ProjectionEWA3DGSFused", "means.bin"))};
  const auto h_quats {
    load_floats(fixture_path("ProjectionEWA3DGSFused", "quats.bin"))};
  const auto h_scales {
    load_floats(fixture_path("ProjectionEWA3DGSFused", "scales.bin"))};
  const auto h_viewmats {
    load_floats(fixture_path("ProjectionEWA3DGSFused", "viewmats.bin"))};
  const auto h_Ks {
    load_floats(fixture_path("ProjectionEWA3DGSFused", "Ks.bin"))};
  const auto h_expected_radii {
    load_int32s(fixture_path("ProjectionEWA3DGSFused", "radii.bin"))};
  const auto h_expected_means2d {
    load_floats(fixture_path("ProjectionEWA3DGSFused", "means2d.bin"))};
  const auto h_expected_depths {
    load_floats(fixture_path("ProjectionEWA3DGSFused", "depths.bin"))};
  const auto h_expected_conics {
    load_floats(fixture_path("ProjectionEWA3DGSFused", "conics.bin"))};

  ASSERT_EQ(h_means.size(), B * N * 3u);
  ASSERT_EQ(h_quats.size(), B * N * 4u);
  ASSERT_EQ(h_scales.size(), B * N * 3u);
  ASSERT_EQ(h_viewmats.size(), B * C * 16u);
  ASSERT_EQ(h_Ks.size(), B * C * 9u);
  ASSERT_EQ(h_expected_radii.size(), B * C * N * 2u);
  ASSERT_EQ(h_expected_means2d.size(), B * C * N * 2u);
  ASSERT_EQ(h_expected_depths.size(), B * C * N);
  ASSERT_EQ(h_expected_conics.size(), B * C * N * 3u);

  DeviceBuffer<float> d_means{h_means.size()};
  DeviceBuffer<float> d_quats{h_quats.size()};
  DeviceBuffer<float> d_scales{h_scales.size()};
  DeviceBuffer<float> d_viewmats{h_viewmats.size()};
  DeviceBuffer<float> d_Ks{h_Ks.size()};

  d_means.copy_from_host(h_means);
  d_quats.copy_from_host(h_quats);
  d_scales.copy_from_host(h_scales);
  d_viewmats.copy_from_host(h_viewmats);
  d_Ks.copy_from_host(h_Ks);

  DeviceBuffer<std::int32_t> d_radii{B * C * N * 2};
  DeviceBuffer<float> d_means2d{B * C * N * 2};
  DeviceBuffer<float> d_depths{B * C * N};
  DeviceBuffer<float> d_conics{B * C * N * 3};

  launch_projection_ewa_3dgs_fused_forward(
    B, C, N,
    d_means.data(),
    /*covars=*/nullptr,
    d_quats.data(),
    d_scales.data(),
    /*opacities=*/nullptr,
    d_viewmats.data(),
    d_Ks.data(),
    image_width,
    image_height,
    /*eps2d=*/0.3f,
    /*near_plane=*/0.01f,
    /*far_plane=*/1e10f,
    /*radius_clip=*/0.0f,
    /*camera_model=*/0, // PINHOLE
    d_radii.data(),
    d_means2d.data(),
    d_depths.data(),
    d_conics.data(),
    /*compensations=*/nullptr,
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_radii.copy_to_host(), h_expected_radii);

  // For float outputs, mask culled Gaussians and compare only visible ones.
  const auto h_means2d {d_means2d.copy_to_host()};
  const auto h_depths {d_depths.copy_to_host()};
  const auto h_conics {d_conics.copy_to_host()};

  // Collect only visible-Gaussian elements for comparison
  std::vector<float> got_means2d, exp_means2d;
  std::vector<float> got_depths, exp_depths;
  std::vector<float> got_conics, exp_conics;

  for (std::size_t i {0}; i < B * C * N; ++i)
  {
    if (h_expected_radii[i * 2] == 0 && h_expected_radii[i * 2 + 1] == 0)
    {
      continue;
    }
    got_means2d.push_back(h_means2d[i * 2]);
    got_means2d.push_back(h_means2d[i * 2 + 1]);
    exp_means2d.push_back(h_expected_means2d[i * 2]);
    exp_means2d.push_back(h_expected_means2d[i * 2 + 1]);

    got_depths.push_back(h_depths[i]);
    exp_depths.push_back(h_expected_depths[i]);

    got_conics.push_back(h_conics[i * 3]);
    got_conics.push_back(h_conics[i * 3 + 1]);
    got_conics.push_back(h_conics[i * 3 + 2]);
    exp_conics.push_back(h_expected_conics[i * 3]);
    exp_conics.push_back(h_expected_conics[i * 3 + 1]);
    exp_conics.push_back(h_expected_conics[i * 3 + 2]);
  }

  constexpr float kAtol {1e-3f};
  constexpr float kRtol {5e-3f};
  expect_close(got_means2d, exp_means2d, kAtol, kRtol, "means2d");
  expect_close(got_depths, exp_depths, 1e-4f, 1e-4f, "depths");
  expect_close(got_conics, exp_conics, kAtol, kRtol, "conics");
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
