#include "DeviceBuffer.h"
#include "Kernels/Heterosplat/RasterizeToPixels3DGS.h"
#include "OracleFixture.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using GoogleUnitTests::DeviceBuffer;
using Kernels::Heterosplat::launch_rasterize_to_pixels_3dgs_forward;
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
/// Oracle comparison against gsplat-Python's `rasterize_to_pixels()`.
/// Full pipeline: projection -> isect_tiles -> isect_offset -> rasterize.
//------------------------------------------------------------------------------
TEST(RasterizeToPixels3DGSOracle, ForwardMatchesGsplatPython)
{
  const std::string group {"RasterizeToPixels3DGS"};

  const std::uint32_t I {
    load_uint32(fixture_path(group, "I.bin"))};
  const std::uint32_t N {
    load_uint32(fixture_path(group, "N.bin"))};
  const std::uint32_t n_isects {
    load_uint32(fixture_path(group, "n_isects.bin"))};
  const std::uint32_t image_width {
    load_uint32(fixture_path(group, "image_width.bin"))};
  const std::uint32_t image_height {
    load_uint32(fixture_path(group, "image_height.bin"))};
  const std::uint32_t tile_size {
    load_uint32(fixture_path(group, "tile_size.bin"))};

  const auto h_means2d {
    load_floats(fixture_path(group, "means2d.bin"))};
  const auto h_conics {
    load_floats(fixture_path(group, "conics.bin"))};
  const auto h_colors {
    load_floats(fixture_path(group, "colors.bin"))};
  const auto h_opacities {
    load_floats(fixture_path(group, "opacities.bin"))};
  const auto h_tile_offsets {
    load_int32s(fixture_path(group, "tile_offsets.bin"))};
  const auto h_flatten_ids {
    load_int32s(fixture_path(group, "flatten_ids.bin"))};
  const auto h_expected_colors {
    load_floats(fixture_path(group, "render_colors.bin"))};
  const auto h_expected_alphas {
    load_floats(fixture_path(group, "render_alphas.bin"))};

  constexpr std::uint32_t CDIM {3};
  const std::uint32_t n_pixels {I * image_height * image_width};

  ASSERT_EQ(h_means2d.size(), I * N * 2u);
  ASSERT_EQ(h_conics.size(), I * N * 3u);
  ASSERT_EQ(h_colors.size(), I * N * CDIM);
  ASSERT_EQ(h_opacities.size(), I * N);
  ASSERT_EQ(h_flatten_ids.size(), n_isects);
  ASSERT_EQ(h_expected_colors.size(), n_pixels * CDIM);
  ASSERT_EQ(h_expected_alphas.size(), n_pixels);

  DeviceBuffer<float> d_means2d{h_means2d.size()};
  DeviceBuffer<float> d_conics{h_conics.size()};
  DeviceBuffer<float> d_colors{h_colors.size()};
  DeviceBuffer<float> d_opacities{h_opacities.size()};
  DeviceBuffer<std::int32_t> d_tile_offsets{h_tile_offsets.size()};
  DeviceBuffer<std::int32_t> d_flatten_ids{h_flatten_ids.size()};

  d_means2d.copy_from_host(h_means2d);
  d_conics.copy_from_host(h_conics);
  d_colors.copy_from_host(h_colors);
  d_opacities.copy_from_host(h_opacities);
  d_tile_offsets.copy_from_host(h_tile_offsets);
  d_flatten_ids.copy_from_host(h_flatten_ids);

  DeviceBuffer<float> d_render_colors{n_pixels * CDIM};
  DeviceBuffer<float> d_render_alphas{n_pixels};
  DeviceBuffer<std::int32_t> d_last_ids{n_pixels};

  std::vector<float> zeros_colors(n_pixels * CDIM, 0.0f);
  std::vector<float> zeros_alphas(n_pixels, 0.0f);
  std::vector<std::int32_t> zeros_last(n_pixels, 0);
  d_render_colors.copy_from_host(zeros_colors);
  d_render_alphas.copy_from_host(zeros_alphas);
  d_last_ids.copy_from_host(zeros_last);

  launch_rasterize_to_pixels_3dgs_forward(
    I, N, n_isects, /*packed=*/false,
    d_means2d.data(),
    d_conics.data(),
    d_colors.data(),
    d_opacities.data(),
    /*backgrounds=*/nullptr,
    /*masks=*/nullptr,
    image_width, image_height, tile_size,
    d_tile_offsets.data(),
    d_flatten_ids.data(),
    d_render_colors.data(),
    d_render_alphas.data(),
    d_last_ids.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto h_render_colors {d_render_colors.copy_to_host()};
  const auto h_render_alphas {d_render_alphas.copy_to_host()};

  constexpr float kAtol {1e-3f};
  constexpr float kRtol {5e-3f};
  expect_close(h_render_colors, h_expected_colors, kAtol, kRtol,
    "render_colors");
  expect_close(h_render_alphas, h_expected_alphas, kAtol, kRtol,
    "render_alphas");
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
