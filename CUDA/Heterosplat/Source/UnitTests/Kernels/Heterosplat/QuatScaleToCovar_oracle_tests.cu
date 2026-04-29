#include "Core/Tensor.h"
#include "Kernels/Heterosplat/QuatScaleToCovar.h"
#include "OracleFixture.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using Core::Tensor;
using Kernels::Heterosplat::launch_quat_scale_to_covar_preci_backward;
using Kernels::Heterosplat::launch_quat_scale_to_covar_preci_forward;
using GoogleUnitTests::OracleFixture::fixture_path;
using GoogleUnitTests::OracleFixture::load_floats;
using GoogleUnitTests::OracleFixture::load_uint32;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// Compares element-wise to an oracle, with combined absolute / relative
/// tolerance:  |a - b| <= atol + rtol * |b|.
//------------------------------------------------------------------------------
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
/// Forward: feed gsplat-Python's `quats` and `scales` into our launcher,
/// expect bit-close agreement with gsplat-Python's `covars` and `precis`.
/// Tolerance is essentially float32 round-off (the same vendored kernels
/// run on both sides; only the launcher boundary differs).
//------------------------------------------------------------------------------
TEST(QuatScaleToCovarOracle, ForwardMatchesGsplatPython)
{
  const std::uint32_t N {load_uint32(fixture_path("QuatScaleToCovar", "N.bin"))};

  const auto h_quats {load_floats(fixture_path("QuatScaleToCovar", "quats.bin"))};
  const auto h_scales {load_floats(fixture_path("QuatScaleToCovar", "scales.bin"))};
  const auto h_expected_covars {
    load_floats(fixture_path("QuatScaleToCovar", "covars.bin"))};
  const auto h_expected_precis {
    load_floats(fixture_path("QuatScaleToCovar", "precis.bin"))};

  ASSERT_EQ(h_quats.size(), N * 4u);
  ASSERT_EQ(h_scales.size(), N * 3u);
  ASSERT_EQ(h_expected_covars.size(), N * 9u);
  ASSERT_EQ(h_expected_precis.size(), N * 9u);

  Tensor d_quats{{N, 4}};
  Tensor d_scales{{N, 3}};
  Tensor d_covars{{N, 3, 3}};
  Tensor d_precis{{N, 3, 3}};
  d_quats.copy_from_host(h_quats.data());
  d_scales.copy_from_host(h_scales.data());

  launch_quat_scale_to_covar_preci_forward(
    N, d_quats.data(), d_scales.data(), /*triu=*/false,
    d_covars.data(), d_precis.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_covars(N * 9);
  std::vector<float> h_precis(N * 9);
  d_covars.copy_to_host(h_covars.data());
  d_precis.copy_to_host(h_precis.data());

  // Same vendored kernel runs on both sides, but nvcc-version drift between
  // the host build (CUDA 13.0) and the gsplat-Python extension built at
  // image-build time (CUDA 13.1) can produce ULP-level FMA-scheduling
  // differences. 1e-4 absolute / 1e-4 relative comfortably covers that
  // while still rejecting any algorithmic divergence.
  constexpr float kAbsolute {1e-4f};
  constexpr float kRelative {1e-4f};
  expect_close(h_covars, h_expected_covars, kAbsolute, kRelative, "covars");
  expect_close(h_precis, h_expected_precis, kAbsolute, kRelative, "precis");
}

//------------------------------------------------------------------------------
/// Backward: feed gsplat's captured (quats, scales, v_covars, v_precis) into
/// our backward launcher; expect (v_quats, v_scales) to match gsplat-Python's.
//------------------------------------------------------------------------------
TEST(QuatScaleToCovarOracle, BackwardMatchesGsplatPython)
{
  const std::uint32_t N {load_uint32(fixture_path("QuatScaleToCovar", "N.bin"))};

  const auto h_quats {load_floats(fixture_path("QuatScaleToCovar", "quats.bin"))};
  const auto h_scales {load_floats(fixture_path("QuatScaleToCovar", "scales.bin"))};
  const auto h_v_covars {
    load_floats(fixture_path("QuatScaleToCovar", "v_covars.bin"))};
  const auto h_v_precis {
    load_floats(fixture_path("QuatScaleToCovar", "v_precis.bin"))};
  const auto h_expected_v_quats {
    load_floats(fixture_path("QuatScaleToCovar", "v_quats.bin"))};
  const auto h_expected_v_scales {
    load_floats(fixture_path("QuatScaleToCovar", "v_scales.bin"))};

  Tensor d_quats{{N, 4}};
  Tensor d_scales{{N, 3}};
  Tensor d_v_covars{{N, 3, 3}};
  Tensor d_v_precis{{N, 3, 3}};
  Tensor d_v_quats{{N, 4}};
  Tensor d_v_scales{{N, 3}};
  d_quats.copy_from_host(h_quats.data());
  d_scales.copy_from_host(h_scales.data());
  d_v_covars.copy_from_host(h_v_covars.data());
  d_v_precis.copy_from_host(h_v_precis.data());

  launch_quat_scale_to_covar_preci_backward(
    N, d_quats.data(), d_scales.data(), /*triu=*/false,
    d_v_covars.data(), d_v_precis.data(),
    d_v_quats.data(), d_v_scales.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_v_quats(N * 4);
  std::vector<float> h_v_scales(N * 3);
  d_v_quats.copy_to_host(h_v_quats.data());
  d_v_scales.copy_to_host(h_v_scales.data());

  // Same nvcc-version drift caveat as the forward test (see comment above).
  constexpr float kAbsolute {1e-4f};
  constexpr float kRelative {1e-4f};
  expect_close(h_v_quats, h_expected_v_quats, kAbsolute, kRelative, "v_quats");
  expect_close(h_v_scales, h_expected_v_scales, kAbsolute, kRelative, "v_scales");
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
