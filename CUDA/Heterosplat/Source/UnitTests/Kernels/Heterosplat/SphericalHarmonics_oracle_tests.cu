#include "Core/Tensor.h"
#include "Kernels/Heterosplat/SphericalHarmonics.h"
#include "OracleFixture.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using Core::Tensor;
using Kernels::Heterosplat::launch_spherical_harmonics_backward;
using Kernels::Heterosplat::launch_spherical_harmonics_forward;
using GoogleUnitTests::OracleFixture::fixture_path;
using GoogleUnitTests::OracleFixture::load_floats;
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
/// Forward against gsplat-Python: same kernel, same inputs -> bit-equivalent
/// (within float32 round-off) colours.
//------------------------------------------------------------------------------
TEST(SphericalHarmonicsOracle, ForwardMatchesGsplatPython)
{
  const std::uint32_t N {load_uint32(fixture_path("SphericalHarmonics", "N.bin"))};
  const std::uint32_t K {load_uint32(fixture_path("SphericalHarmonics", "K.bin"))};
  const std::uint32_t degrees_to_use {
    load_uint32(fixture_path("SphericalHarmonics", "degrees_to_use.bin"))};

  const auto h_dirs {load_floats(fixture_path("SphericalHarmonics", "dirs.bin"))};
  const auto h_coeffs {
    load_floats(fixture_path("SphericalHarmonics", "coeffs.bin"))};
  const auto h_expected_colors {
    load_floats(fixture_path("SphericalHarmonics", "colors.bin"))};

  ASSERT_EQ(h_dirs.size(), N * 3u);
  ASSERT_EQ(h_coeffs.size(), N * K * 3u);
  ASSERT_EQ(h_expected_colors.size(), N * 3u);

  Tensor d_dirs{{N, 3}};
  Tensor d_coeffs{{N, K, 3}};
  Tensor d_colors{{N, 3}};
  d_dirs.copy_from_host(h_dirs.data());
  d_coeffs.copy_from_host(h_coeffs.data());

  launch_spherical_harmonics_forward(
    N, K, degrees_to_use,
    d_dirs.data(), d_coeffs.data(), /*masks=*/nullptr,
    d_colors.data(), /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_colors(N * 3);
  d_colors.copy_to_host(h_colors.data());

  // See note in QuatScaleToCovar_oracle_tests.cu: nvcc 13.0 (host) vs nvcc
  // 13.1 (container, where gsplat was built) introduces ULP-level FMA-
  // scheduling drift. 1e-4 covers it while still catching real divergence.
  constexpr float kAbsolute {1e-4f};
  constexpr float kRelative {1e-4f};
  expect_close(h_colors, h_expected_colors, kAbsolute, kRelative, "colors");
}

//------------------------------------------------------------------------------
/// Backward: feed (dirs, coeffs, v_colors), check (v_coeffs, v_dirs).
/// Note: the backward kernel uses atomicAdd into v_dirs (three colour
/// channels accumulate per-Gaussian), so v_dirs MUST be pre-zeroed --
/// we do that explicitly here.
//------------------------------------------------------------------------------
TEST(SphericalHarmonicsOracle, BackwardMatchesGsplatPython)
{
  const std::uint32_t N {load_uint32(fixture_path("SphericalHarmonics", "N.bin"))};
  const std::uint32_t K {load_uint32(fixture_path("SphericalHarmonics", "K.bin"))};
  const std::uint32_t degrees_to_use {
    load_uint32(fixture_path("SphericalHarmonics", "degrees_to_use.bin"))};

  const auto h_dirs {load_floats(fixture_path("SphericalHarmonics", "dirs.bin"))};
  const auto h_coeffs {
    load_floats(fixture_path("SphericalHarmonics", "coeffs.bin"))};
  const auto h_v_colors {
    load_floats(fixture_path("SphericalHarmonics", "v_colors.bin"))};
  const auto h_expected_v_coeffs {
    load_floats(fixture_path("SphericalHarmonics", "v_coeffs.bin"))};
  const auto h_expected_v_dirs {
    load_floats(fixture_path("SphericalHarmonics", "v_dirs.bin"))};

  Tensor d_dirs{{N, 3}};
  Tensor d_coeffs{{N, K, 3}};
  Tensor d_v_colors{{N, 3}};
  Tensor d_v_coeffs{{N, K, 3}};
  Tensor d_v_dirs{{N, 3}};
  d_dirs.copy_from_host(h_dirs.data());
  d_coeffs.copy_from_host(h_coeffs.data());
  d_v_colors.copy_from_host(h_v_colors.data());
  // Pre-zero v_dirs because the kernel atomicAdd-accumulates into it.
  ASSERT_EQ(
    cudaMemset(d_v_dirs.data(), 0, N * 3 * sizeof(float)),
    cudaSuccess);

  launch_spherical_harmonics_backward(
    N, K, degrees_to_use,
    d_dirs.data(), d_coeffs.data(), /*masks=*/nullptr,
    d_v_colors.data(),
    d_v_coeffs.data(), d_v_dirs.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_v_coeffs(N * K * 3);
  std::vector<float> h_v_dirs(N * 3);
  d_v_coeffs.copy_to_host(h_v_coeffs.data());
  d_v_dirs.copy_to_host(h_v_dirs.data());

  // See note in QuatScaleToCovar_oracle_tests.cu: nvcc 13.0 (host) vs nvcc
  // 13.1 (container, where gsplat was built) introduces ULP-level FMA-
  // scheduling drift. 1e-4 covers it while still catching real divergence.
  constexpr float kAbsolute {1e-4f};
  constexpr float kRelative {1e-4f};
  expect_close(h_v_coeffs, h_expected_v_coeffs, kAbsolute, kRelative, "v_coeffs");
  expect_close(h_v_dirs, h_expected_v_dirs, kAbsolute, kRelative, "v_dirs");
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
