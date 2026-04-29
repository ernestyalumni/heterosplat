#include "Core/Tensor.h"
#include "Kernels/Heterosplat/SphericalHarmonics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using Core::Tensor;
using Kernels::Heterosplat::launch_spherical_harmonics_backward;
using Kernels::Heterosplat::launch_spherical_harmonics_forward;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// SH normalisation constants used by the kernel (Sloan/JCGT 2013).
//------------------------------------------------------------------------------
constexpr float kSHCoefficient0 {0.2820947917738781f};   // Y_0^0     = 1 / (2 * sqrt(pi))
constexpr float kSHCoefficient1 {0.48860251190292f};     // |Y_1^m|   = sqrt(3 / (4 * pi))

//------------------------------------------------------------------------------
/// Degree 0 (DC only): the only basis function is constant Y_0^0 = 0.2820...,
/// so colors[n, c] = kSHCoefficient0 * coeffs[n, 0, c]. View direction is
/// irrelevant at degree 0.
//------------------------------------------------------------------------------
TEST(SphericalHarmonics, ForwardDegreeZeroIsDCOnly)
{
  constexpr std::uint32_t N {1};
  constexpr std::uint32_t K {1};
  constexpr std::uint32_t degree {0};

  // Direction is unused at degree 0; pass an arbitrary non-zero vector.
  const std::vector<float> h_dirs {1.f, 0.f, 0.f};
  // Three channels, one coefficient each.
  const std::vector<float> h_coeffs {5.f, 7.f, 11.f};

  Tensor d_dirs{{N, 3}};
  Tensor d_coeffs{{N, K, 3}};
  Tensor d_colors{{N, 3}};
  d_dirs.copy_from_host(h_dirs.data());
  d_coeffs.copy_from_host(h_coeffs.data());

  launch_spherical_harmonics_forward(
    N, K, degree,
    d_dirs.data(), d_coeffs.data(), /*masks=*/nullptr,
    d_colors.data(), /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_colors(3, std::nanf(""));
  d_colors.copy_to_host(h_colors.data());

  EXPECT_NEAR(h_colors[0], kSHCoefficient0 *  5.f, 1e-6f);
  EXPECT_NEAR(h_colors[1], kSHCoefficient0 *  7.f, 1e-6f);
  EXPECT_NEAR(h_colors[2], kSHCoefficient0 * 11.f, 1e-6f);
}

//------------------------------------------------------------------------------
/// Degree 1 with a known direction picks out a single l=1 basis. With
/// dir = (0, 1, 0) the kernel's expansion (gsplat / Sloan convention)
///   result += 0.488602 * (-y * c[1] + z * c[2] - x * c[3])
/// reduces to `-0.488602 * c[1, channel]`. Other coefficients are zeroed
/// to make the contribution from each isolated basis testable in isolation.
//------------------------------------------------------------------------------
TEST(SphericalHarmonics, ForwardDegreeOneIsolatesSingleBasisYNeg1)
{
  constexpr std::uint32_t N {1};
  constexpr std::uint32_t K {4};   // (1+1)^2 coefficients for degree 1
  constexpr std::uint32_t degree {1};

  const std::vector<float> h_dirs {0.f, 1.f, 0.f};

  // K coefficients per channel; layout is [coeff, channel] flattened.
  // Index 1 maps to Y_1^{-1} (the y-component basis).
  std::vector<float> h_coeffs(K * 3, 0.f);
  h_coeffs[1 * 3 + 0] = 2.f;   // R coefficient on Y_1^{-1}
  h_coeffs[1 * 3 + 1] = 3.f;   // G coefficient on Y_1^{-1}
  h_coeffs[1 * 3 + 2] = 5.f;   // B coefficient on Y_1^{-1}

  Tensor d_dirs{{N, 3}};
  Tensor d_coeffs{{N, K, 3}};
  Tensor d_colors{{N, 3}};
  d_dirs.copy_from_host(h_dirs.data());
  d_coeffs.copy_from_host(h_coeffs.data());

  launch_spherical_harmonics_forward(
    N, K, degree,
    d_dirs.data(), d_coeffs.data(), nullptr,
    d_colors.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_colors(3, std::nanf(""));
  d_colors.copy_to_host(h_colors.data());

  // Expected: kSHCoefficient0 * c[0, ch] + kSHCoefficient1 * (-1 * c[1, ch] + 0 + 0)
  //         = -kSHCoefficient1 * c[1, ch]   (since c[0] = 0)
  EXPECT_NEAR(h_colors[0], -kSHCoefficient1 *  2.f, 1e-6f);
  EXPECT_NEAR(h_colors[1], -kSHCoefficient1 *  3.f, 1e-6f);
  EXPECT_NEAR(h_colors[2], -kSHCoefficient1 *  5.f, 1e-6f);
}

//------------------------------------------------------------------------------
/// Mask=false short-circuits the per-Gaussian computation. We pre-fill the
/// output with a sentinel and verify it is left untouched for the masked
/// Gaussian, while the unmasked one gets written.
//------------------------------------------------------------------------------
TEST(SphericalHarmonics, ForwardMaskSkipsMaskedGaussians)
{
  constexpr std::uint32_t N {2};
  constexpr std::uint32_t K {1};
  constexpr std::uint32_t degree {0};

  const std::vector<float> h_dirs {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
  const std::vector<float> h_coeffs {1.f, 2.f, 3.f,   4.f, 5.f, 6.f};
  const std::vector<bool> h_masks_bool {true, false};
  // std::vector<bool> is bit-packed -- copy out to a uint8 buffer for cudaMemcpy.
  std::vector<unsigned char> h_masks(N);
  for (std::size_t i {0}; i < N; ++i) h_masks[i] = h_masks_bool[i] ? 1u : 0u;

  // Pre-fill the colors buffer with a sentinel so we can detect untouched.
  const float kSentinel {-12345.f};
  std::vector<float> h_colors_initial(N * 3, kSentinel);

  Tensor d_dirs{{N, 3}};
  Tensor d_coeffs{{N, K, 3}};
  Tensor d_colors{{N, 3}};
  d_dirs.copy_from_host(h_dirs.data());
  d_coeffs.copy_from_host(h_coeffs.data());
  d_colors.copy_from_host(h_colors_initial.data());

  bool* d_masks {nullptr};
  ASSERT_EQ(cudaMalloc(&d_masks, N), cudaSuccess);
  ASSERT_EQ(
    cudaMemcpy(d_masks, h_masks.data(), N, cudaMemcpyHostToDevice),
    cudaSuccess);

  launch_spherical_harmonics_forward(
    N, K, degree,
    d_dirs.data(), d_coeffs.data(), d_masks,
    d_colors.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  cudaFree(d_masks);

  std::vector<float> h_colors(N * 3);
  d_colors.copy_to_host(h_colors.data());

  // Unmasked Gaussian (n=0): expected = kSHCoefficient0 * coeffs.
  EXPECT_NEAR(h_colors[0], kSHCoefficient0 * 1.f, 1e-6f);
  EXPECT_NEAR(h_colors[1], kSHCoefficient0 * 2.f, 1e-6f);
  EXPECT_NEAR(h_colors[2], kSHCoefficient0 * 3.f, 1e-6f);
  // Masked Gaussian (n=1): output left at sentinel.
  EXPECT_FLOAT_EQ(h_colors[3], kSentinel);
  EXPECT_FLOAT_EQ(h_colors[4], kSentinel);
  EXPECT_FLOAT_EQ(h_colors[5], kSentinel);
}

//------------------------------------------------------------------------------
/// Backward gradcheck: analytic VJP from kernel vs centered finite-difference
/// through the forward. We probe gradients w.r.t. coefficients only (v_dirs
/// would also be valid but adds atomicAdd interleaving complexity; coeffs is
/// the dominant gradient path in training).
//------------------------------------------------------------------------------
TEST(SphericalHarmonics, BackwardGradCheckCoefficientsMatchNumericalDifference)
{
  constexpr std::uint32_t N {2};
  constexpr std::uint32_t K {4};   // degree 1
  constexpr std::uint32_t degree {1};

  // Two non-axis-aligned unit-ish directions.
  const std::vector<float> h_dirs {
    0.6f, 0.8f, 0.0f,
    0.0f, 0.6f, 0.8f,
  };
  // Random-ish non-zero coefficients.
  std::vector<float> h_coeffs(N * K * 3);
  for (std::size_t i {0}; i < h_coeffs.size(); ++i)
  {
    h_coeffs[i] = 0.1f + 0.05f * static_cast<float>(i);
  }
  // Upstream gradient.
  std::vector<float> h_v_colors(N * 3);
  for (std::size_t i {0}; i < h_v_colors.size(); ++i)
  {
    h_v_colors[i] = 0.5f - 0.1f * static_cast<float>(i);
  }

  Tensor d_dirs{{N, 3}};
  Tensor d_coeffs{{N, K, 3}};
  Tensor d_v_colors{{N, 3}};
  Tensor d_v_coeffs{{N, K, 3}};
  d_dirs.copy_from_host(h_dirs.data());
  d_coeffs.copy_from_host(h_coeffs.data());
  d_v_colors.copy_from_host(h_v_colors.data());

  launch_spherical_harmonics_backward(
    N, K, degree,
    d_dirs.data(), d_coeffs.data(), nullptr,
    d_v_colors.data(),
    d_v_coeffs.data(), /*v_dirs=*/nullptr,
    nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_analytic_v_coeffs(N * K * 3);
  d_v_coeffs.copy_to_host(h_analytic_v_coeffs.data());

  // L = sum_{n, c} v_colors[n, c] * colors[n, c]
  auto compute_loss =
    [&](const std::vector<float>& coeffs) -> double
    {
      Tensor d_co{{N, K, 3}};
      Tensor d_clr{{N, 3}};
      d_co.copy_from_host(coeffs.data());
      launch_spherical_harmonics_forward(
        N, K, degree,
        d_dirs.data(), d_co.data(), nullptr,
        d_clr.data(), nullptr);
      cudaDeviceSynchronize();
      std::vector<float> h_clr(N * 3);
      d_clr.copy_to_host(h_clr.data());
      double loss {0.0};
      for (std::size_t i {0}; i < h_clr.size(); ++i)
      {
        loss += static_cast<double>(h_v_colors[i]) *
                static_cast<double>(h_clr[i]);
      }
      return loss;
    };

  const float epsilon {1e-3f};
  std::vector<float> h_numerical_v_coeffs(N * K * 3, 0.f);
  for (std::size_t i {0}; i < h_coeffs.size(); ++i)
  {
    auto plus {h_coeffs};  plus[i]  += epsilon;
    auto minus {h_coeffs}; minus[i] -= epsilon;
    h_numerical_v_coeffs[i] = static_cast<float>(
      (compute_loss(plus) - compute_loss(minus)) / (2.0 * epsilon));
  }

  constexpr float kAbsoluteTolerance {1e-3f};
  constexpr float kRelativeTolerance {5e-3f};
  for (std::size_t i {0}; i < h_coeffs.size(); ++i)
  {
    const float diff {std::abs(
      h_analytic_v_coeffs[i] - h_numerical_v_coeffs[i])};
    const float tolerance {
      kAbsoluteTolerance +
      kRelativeTolerance * std::abs(h_numerical_v_coeffs[i])};
    EXPECT_LE(diff, tolerance) <<
      "v_coeffs[" << i << "] analytic=" << h_analytic_v_coeffs[i] <<
      " numerical=" << h_numerical_v_coeffs[i];
  }
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
