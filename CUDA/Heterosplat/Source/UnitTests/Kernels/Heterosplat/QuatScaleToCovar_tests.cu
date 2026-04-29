#include "Core/Tensor.h"
#include "Kernels/Heterosplat/QuatScaleToCovar.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using Core::Tensor;
using Kernels::Heterosplat::launch_quat_scale_to_covar_preci_backward;
using Kernels::Heterosplat::launch_quat_scale_to_covar_preci_forward;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// Identity quaternion (w=1, x=y=z=0) + unit scale → identity covariance.
/// This is the smallest-possible numerical sanity check: R = I, S = I,
/// covar = R S S R^T = I.
//------------------------------------------------------------------------------
TEST(QuatScaleToCovar, IdentityQuatUnitScaleProducesIdentityCovar)
{
  constexpr std::uint32_t N {3};

  // gsplat quat layout is (w, x, y, z).
  const std::vector<float> h_quats {
    1.f, 0.f, 0.f, 0.f,
    1.f, 0.f, 0.f, 0.f,
    1.f, 0.f, 0.f, 0.f,
  };
  const std::vector<float> h_scales {
    1.f, 1.f, 1.f,
    1.f, 1.f, 1.f,
    1.f, 1.f, 1.f,
  };

  Tensor d_quats{{N, 4}};
  Tensor d_scales{{N, 3}};
  Tensor d_covars{{N, 3, 3}};

  d_quats.copy_from_host(h_quats.data());
  d_scales.copy_from_host(h_scales.data());

  launch_quat_scale_to_covar_preci_forward(
    N,
    d_quats.data(),
    d_scales.data(),
    /*triu=*/false,
    d_covars.data(),
    /*precis=*/nullptr,
    /*stream=*/nullptr);

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) <<
    "quat_scale_to_covar fwd kernel reported a launch error";

  std::vector<float> h_covars(N * 9, 0.f);
  d_covars.copy_to_host(h_covars.data());

  for (std::uint32_t n {0}; n < N; ++n)
  {
    for (std::uint32_t i {0}; i < 3; ++i)
    {
      for (std::uint32_t j {0}; j < 3; ++j)
      {
        const float expected {(i == j) ? 1.f : 0.f};
        const float got {h_covars[n * 9 + i * 3 + j]};
        EXPECT_NEAR(got, expected, 1e-6f) <<
          "n=" << n << " i=" << i << " j=" << j;
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Identity quaternion + scale (sx, sy, sz) → diagonal covariance with
/// (sx^2, sy^2, sz^2) on the diagonal. Anisotropic: catches any silent
/// row/col-major flip in the row-major copy-out.
//------------------------------------------------------------------------------
TEST(QuatScaleToCovar, IdentityQuatAnisotropicScaleDiagonalCovar)
{
  constexpr std::uint32_t N {1};
  const std::vector<float> h_quats {1.f, 0.f, 0.f, 0.f};
  const std::vector<float> h_scales {2.f, 3.f, 5.f};

  Tensor d_quats{{N, 4}};
  Tensor d_scales{{N, 3}};
  Tensor d_covars{{N, 3, 3}};
  d_quats.copy_from_host(h_quats.data());
  d_scales.copy_from_host(h_scales.data());

  launch_quat_scale_to_covar_preci_forward(
    N, d_quats.data(), d_scales.data(), /*triu=*/false,
    d_covars.data(), /*precis=*/nullptr, /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_covars(9, 0.f);
  d_covars.copy_to_host(h_covars.data());

  EXPECT_NEAR(h_covars[0], 4.f, 1e-5f);   // sx^2
  EXPECT_NEAR(h_covars[4], 9.f, 1e-5f);   // sy^2
  EXPECT_NEAR(h_covars[8], 25.f, 1e-5f);  // sz^2
  // Off-diagonals should be zero
  for (const std::uint32_t off : {1, 2, 3, 5, 6, 7})
  {
    EXPECT_NEAR(h_covars[off], 0.f, 1e-5f) << "off=" << off;
  }
}

//------------------------------------------------------------------------------
/// triu layout produces 6 floats per Gaussian in (xx, xy, xz, yy, yz, zz).
//------------------------------------------------------------------------------
TEST(QuatScaleToCovar, TriuLayoutMatchesFullLayout)
{
  constexpr std::uint32_t N {1};
  const std::vector<float> h_quats {1.f, 0.f, 0.f, 0.f};
  const std::vector<float> h_scales {2.f, 3.f, 5.f};

  Tensor d_quats{{N, 4}};
  Tensor d_scales{{N, 3}};
  Tensor d_triu{{N, 6}};
  d_quats.copy_from_host(h_quats.data());
  d_scales.copy_from_host(h_scales.data());

  launch_quat_scale_to_covar_preci_forward(
    N, d_quats.data(), d_scales.data(), /*triu=*/true,
    d_triu.data(), /*precis=*/nullptr, /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_triu(6, 0.f);
  d_triu.copy_to_host(h_triu.data());

  EXPECT_NEAR(h_triu[0], 4.f, 1e-5f);  // xx
  EXPECT_NEAR(h_triu[1], 0.f, 1e-5f);  // xy
  EXPECT_NEAR(h_triu[2], 0.f, 1e-5f);  // xz
  EXPECT_NEAR(h_triu[3], 9.f, 1e-5f);  // yy
  EXPECT_NEAR(h_triu[4], 0.f, 1e-5f);  // yz
  EXPECT_NEAR(h_triu[5], 25.f, 1e-5f); // zz
}

//------------------------------------------------------------------------------
/// Backward (VJP) closed-form sanity check.
///
/// Inputs: q = (1,0,0,0), s = (1,1,1), upstream G = I (identity 3x3).
///
/// Derivation (KernelMathematics.tex §3.5, equations (eq:vjp-mmt) -- (eq:vjp-s)):
///   tilde_G = G + G^T = 2 I
///   dL/dM   = tilde_G * M = 2 I (since M = R*S = I*I = I)
///   dL/dR   = (dL/dM) * S = 2 I
///   dL/ds_k = sum_i R_{ik} * (dL/dM)_{ik} = (dL/dM)_{kk} = 2 for k = 1,2,3
///   dL/dq   = quat_to_rotmat_vjp(q=(1,0,0,0), 2 I) = 0 because the
///             antisymmetric part of dL/dR vanishes (it is symmetric).
//------------------------------------------------------------------------------
TEST(QuatScaleToCovar, BackwardIdentityQuatUnitScaleClosedForm)
{
  constexpr std::uint32_t N {1};
  const std::vector<float> h_quats {1.f, 0.f, 0.f, 0.f};
  const std::vector<float> h_scales {1.f, 1.f, 1.f};
  // Upstream gradient G = I_3, row-major full layout.
  const std::vector<float> h_v_covars {
    1.f, 0.f, 0.f,
    0.f, 1.f, 0.f,
    0.f, 0.f, 1.f,
  };

  Tensor d_quats{{N, 4}};
  Tensor d_scales{{N, 3}};
  Tensor d_v_covars{{N, 3, 3}};
  Tensor d_v_quats{{N, 4}};
  Tensor d_v_scales{{N, 3}};

  d_quats.copy_from_host(h_quats.data());
  d_scales.copy_from_host(h_scales.data());
  d_v_covars.copy_from_host(h_v_covars.data());

  launch_quat_scale_to_covar_preci_backward(
    N, d_quats.data(), d_scales.data(), /*triu=*/false,
    d_v_covars.data(), /*v_precis=*/nullptr,
    d_v_quats.data(), d_v_scales.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_v_quats(4, std::nanf(""));
  std::vector<float> h_v_scales(3, std::nanf(""));
  d_v_quats.copy_to_host(h_v_quats.data());
  d_v_scales.copy_to_host(h_v_scales.data());

  for (int k {0}; k < 4; ++k)
  {
    EXPECT_NEAR(h_v_quats[k], 0.f, 1e-5f) << "k=" << k;
  }
  for (int k {0}; k < 3; ++k)
  {
    EXPECT_NEAR(h_v_scales[k], 2.f, 1e-5f) << "k=" << k;
  }
}

//------------------------------------------------------------------------------
/// Backward gradcheck: analytic VJP from kernel vs centered finite-difference
/// through the forward kernel.
///
/// Picks non-degenerate inputs (no axis-aligned quaternion, scales away from
/// zero) and a non-trivial symmetric upstream G to exercise the full
/// quat-to-rotmat Jacobian (the closed-form test only probes the symmetric
/// kernel of dL/dR -> dL/dq).
///
/// Tolerance accounts for centered-difference O(eps^2) truncation plus
/// float32 round-off — combined: |analytic - numerical| <= atol + rtol * |numerical|.
//------------------------------------------------------------------------------
TEST(QuatScaleToCovar, BackwardGradCheckMatchesNumericalDifference)
{
  constexpr std::uint32_t N {2};

  // Two non-axis-aligned quaternions, two non-uniform positive scales.
  const std::vector<float> h_quats {
    1.0f, 0.5f, -0.3f,  0.2f,    // Gaussian 0
    0.8f, 0.1f,  0.6f, -0.4f,    // Gaussian 1
  };
  const std::vector<float> h_scales {
    0.5f, 0.7f, 1.2f,
    1.5f, 0.3f, 0.9f,
  };

  // Symmetric upstream G replicated per Gaussian (full row-major layout).
  const std::vector<float> h_v_covars_one {
    1.f, 2.f, 3.f,
    2.f, 4.f, 5.f,
    3.f, 5.f, 6.f,
  };
  std::vector<float> h_v_covars(N * 9);
  for (std::size_t n {0}; n < N; ++n)
  {
    std::copy(
      h_v_covars_one.begin(),
      h_v_covars_one.end(),
      h_v_covars.begin() + n * 9);
  }

  // ---- Analytic backward via our kernel.
  Tensor d_quats{{N, 4}};
  Tensor d_scales{{N, 3}};
  Tensor d_v_covars{{N, 3, 3}};
  Tensor d_v_quats{{N, 4}};
  Tensor d_v_scales{{N, 3}};
  d_quats.copy_from_host(h_quats.data());
  d_scales.copy_from_host(h_scales.data());
  d_v_covars.copy_from_host(h_v_covars.data());

  launch_quat_scale_to_covar_preci_backward(
    N, d_quats.data(), d_scales.data(), /*triu=*/false,
    d_v_covars.data(), /*v_precis=*/nullptr,
    d_v_quats.data(), d_v_scales.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<float> h_analytic_v_quats(N * 4);
  std::vector<float> h_analytic_v_scales(N * 3);
  d_v_quats.copy_to_host(h_analytic_v_quats.data());
  d_v_scales.copy_to_host(h_analytic_v_scales.data());

  // ---- Numerical gradient via centered finite difference on the forward.
  // Loss L = sum_{n, i, j} G_{ij} * Sigma_n[i, j] (inner product of upstream
  // grad with forward output).
  auto compute_loss =
    [&](const std::vector<float>& q, const std::vector<float>& s) -> double
    {
      Tensor d_q{{N, 4}};
      Tensor d_s{{N, 3}};
      Tensor d_sigma{{N, 3, 3}};
      d_q.copy_from_host(q.data());
      d_s.copy_from_host(s.data());
      launch_quat_scale_to_covar_preci_forward(
        N, d_q.data(), d_s.data(), /*triu=*/false,
        d_sigma.data(), /*precis=*/nullptr,
        /*stream=*/nullptr);
      cudaDeviceSynchronize();
      std::vector<float> h_sigma(N * 9);
      d_sigma.copy_to_host(h_sigma.data());
      double loss {0.0};
      for (std::size_t n {0}; n < N; ++n)
      {
        for (std::size_t i {0}; i < 9; ++i)
        {
          loss += static_cast<double>(h_v_covars[n * 9 + i]) *
                  static_cast<double>(h_sigma[n * 9 + i]);
        }
      }
      return loss;
    };

  const float epsilon {1e-3f};
  std::vector<float> h_numerical_v_quats(N * 4, 0.f);
  std::vector<float> h_numerical_v_scales(N * 3, 0.f);

  for (std::size_t i {0}; i < N * 4; ++i)
  {
    auto q_plus {h_quats};  q_plus[i]  += epsilon;
    auto q_minus {h_quats}; q_minus[i] -= epsilon;
    h_numerical_v_quats[i] = static_cast<float>(
      (compute_loss(q_plus, h_scales) - compute_loss(q_minus, h_scales)) /
      (2.0 * epsilon));
  }
  for (std::size_t i {0}; i < N * 3; ++i)
  {
    auto s_plus {h_scales};  s_plus[i]  += epsilon;
    auto s_minus {h_scales}; s_minus[i] -= epsilon;
    h_numerical_v_scales[i] = static_cast<float>(
      (compute_loss(h_quats, s_plus) - compute_loss(h_quats, s_minus)) /
      (2.0 * epsilon));
  }

  // Combined absolute / relative tolerance.
  constexpr float kAbsoluteTolerance {1e-2f};
  constexpr float kRelativeTolerance {5e-3f};
  for (std::size_t i {0}; i < N * 4; ++i)
  {
    const float diff {std::abs(h_analytic_v_quats[i] - h_numerical_v_quats[i])};
    const float tolerance {
      kAbsoluteTolerance +
      kRelativeTolerance * std::abs(h_numerical_v_quats[i])};
    EXPECT_LE(diff, tolerance) <<
      "v_quats[" << i << "] analytic=" << h_analytic_v_quats[i] <<
      " numerical=" << h_numerical_v_quats[i];
  }
  for (std::size_t i {0}; i < N * 3; ++i)
  {
    const float diff {std::abs(h_analytic_v_scales[i] - h_numerical_v_scales[i])};
    const float tolerance {
      kAbsoluteTolerance +
      kRelativeTolerance * std::abs(h_numerical_v_scales[i])};
    EXPECT_LE(diff, tolerance) <<
      "v_scales[" << i << "] analytic=" << h_analytic_v_scales[i] <<
      " numerical=" << h_numerical_v_scales[i];
  }
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
