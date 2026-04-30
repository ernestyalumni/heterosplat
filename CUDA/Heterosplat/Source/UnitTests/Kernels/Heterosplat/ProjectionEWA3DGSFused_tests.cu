#include "DeviceBuffer.h"
#include "Kernels/Heterosplat/ProjectionEWA3DGSFused.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using GoogleUnitTests::DeviceBuffer;
using Kernels::Heterosplat::launch_projection_ewa_3dgs_fused_forward;
using Kernels::Heterosplat::launch_projection_ewa_3dgs_fused_backward;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// Single Gaussian on the optical axis, identity viewmat, pinhole camera.
/// mean_world = (0, 0, 5), identity quaternion, uniform scale = 0.1.
/// With fx=fy=100 and cx=cy=64, the projected mean should be (64, 64)
/// at depth 5, with non-zero radii.
//------------------------------------------------------------------------------
TEST(ProjectionEWA3DGSFused, SingleGaussianOnAxisProjectsToCenter)
{
  constexpr std::uint32_t B {1};
  constexpr std::uint32_t C {1};
  constexpr std::uint32_t N {1};
  constexpr std::uint32_t image_width {128};
  constexpr std::uint32_t image_height {128};

  // mean at (0, 0, 5)
  DeviceBuffer<float> means{B * N * 3};
  means.copy_from_host({0.0f, 0.0f, 5.0f});

  // identity quaternion (w, x, y, z) = (1, 0, 0, 0)
  DeviceBuffer<float> quats{B * N * 4};
  quats.copy_from_host({1.0f, 0.0f, 0.0f, 0.0f});

  // isotropic scale
  DeviceBuffer<float> scales{B * N * 3};
  scales.copy_from_host({0.1f, 0.1f, 0.1f});

  // identity viewmat (row-major 4x4)
  DeviceBuffer<float> viewmats{B * C * 16};
  viewmats.copy_from_host({
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1});

  // intrinsics: fx=100, fy=100, cx=64, cy=64
  DeviceBuffer<float> Ks{B * C * 9};
  Ks.copy_from_host({
    100, 0, 64,
    0, 100, 64,
    0, 0, 1});

  DeviceBuffer<std::int32_t> radii{B * C * N * 2};
  DeviceBuffer<float> means2d{B * C * N * 2};
  DeviceBuffer<float> depths{B * C * N};
  DeviceBuffer<float> conics{B * C * N * 3};

  launch_projection_ewa_3dgs_fused_forward(
    B, C, N,
    means.data(),
    /*covars=*/nullptr,
    quats.data(),
    scales.data(),
    /*opacities=*/nullptr,
    viewmats.data(),
    Ks.data(),
    image_width,
    image_height,
    /*eps2d=*/0.3f,
    /*near_plane=*/0.01f,
    /*far_plane=*/1e10f,
    /*radius_clip=*/0.0f,
    /*camera_model=*/0, // PINHOLE
    radii.data(),
    means2d.data(),
    depths.data(),
    conics.data(),
    /*compensations=*/nullptr,
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto h_radii {radii.copy_to_host()};
  const auto h_means2d {means2d.copy_to_host()};
  const auto h_depths {depths.copy_to_host()};
  const auto h_conics {conics.copy_to_host()};

  // Gaussian should not be culled
  EXPECT_GT(h_radii[0], 0);
  EXPECT_GT(h_radii[1], 0);

  // Projected center should be at the principal point
  EXPECT_NEAR(h_means2d[0], 64.0f, 0.01f);
  EXPECT_NEAR(h_means2d[1], 64.0f, 0.01f);

  // Depth should be 5
  EXPECT_NEAR(h_depths[0], 5.0f, 1e-5f);

  // Conics should be finite and symmetric (upper triangle: a, b, c)
  EXPECT_TRUE(std::isfinite(h_conics[0]));
  EXPECT_TRUE(std::isfinite(h_conics[1]));
  EXPECT_TRUE(std::isfinite(h_conics[2]));
}

//------------------------------------------------------------------------------
/// Gaussian behind the camera (depth < near_plane) should be culled.
//------------------------------------------------------------------------------
TEST(ProjectionEWA3DGSFused, BehindCameraIsCulled)
{
  constexpr std::uint32_t B {1};
  constexpr std::uint32_t C {1};
  constexpr std::uint32_t N {1};

  DeviceBuffer<float> means{3};
  means.copy_from_host({0.0f, 0.0f, -1.0f}); // behind camera

  DeviceBuffer<float> quats{4};
  quats.copy_from_host({1.0f, 0.0f, 0.0f, 0.0f});

  DeviceBuffer<float> scales{3};
  scales.copy_from_host({0.1f, 0.1f, 0.1f});

  DeviceBuffer<float> viewmats{16};
  viewmats.copy_from_host({1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1});

  DeviceBuffer<float> Ks{9};
  Ks.copy_from_host({100,0,64, 0,100,64, 0,0,1});

  DeviceBuffer<std::int32_t> radii{2};
  DeviceBuffer<float> means2d{2};
  DeviceBuffer<float> depths{1};
  DeviceBuffer<float> conics{3};

  launch_projection_ewa_3dgs_fused_forward(
    B, C, N,
    means.data(), nullptr, quats.data(), scales.data(), nullptr,
    viewmats.data(), Ks.data(),
    128, 128, 0.3f, 0.01f, 1e10f, 0.0f, 0,
    radii.data(), means2d.data(), depths.data(), conics.data(),
    nullptr, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto h_radii {radii.copy_to_host()};
  EXPECT_EQ(h_radii[0], 0);
  EXPECT_EQ(h_radii[1], 0);
}

//------------------------------------------------------------------------------
/// Backward produces finite, non-zero gradients for means when the Gaussian
/// is visible (basic smoke test, not a full gradcheck).
//------------------------------------------------------------------------------
TEST(ProjectionEWA3DGSFused, BackwardProducesFiniteGradients)
{
  constexpr std::uint32_t B {1};
  constexpr std::uint32_t C {1};
  constexpr std::uint32_t N {1};

  DeviceBuffer<float> means{3};
  means.copy_from_host({0.0f, 0.0f, 5.0f});

  DeviceBuffer<float> quats{4};
  quats.copy_from_host({1.0f, 0.0f, 0.0f, 0.0f});

  DeviceBuffer<float> scales{3};
  scales.copy_from_host({0.1f, 0.1f, 0.1f});

  DeviceBuffer<float> viewmats{16};
  viewmats.copy_from_host({1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1});

  DeviceBuffer<float> Ks{9};
  Ks.copy_from_host({100,0,64, 0,100,64, 0,0,1});

  // Forward pass
  DeviceBuffer<std::int32_t> radii{2};
  DeviceBuffer<float> means2d{2};
  DeviceBuffer<float> depths{1};
  DeviceBuffer<float> conics{3};

  launch_projection_ewa_3dgs_fused_forward(
    B, C, N,
    means.data(), nullptr, quats.data(), scales.data(), nullptr,
    viewmats.data(), Ks.data(),
    128, 128, 0.3f, 0.01f, 1e10f, 0.0f, 0,
    radii.data(), means2d.data(), depths.data(), conics.data(),
    nullptr, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Synthetic upstream gradients: ones
  DeviceBuffer<float> v_means2d{2};
  DeviceBuffer<float> v_depths{1};
  DeviceBuffer<float> v_conics{3};
  v_means2d.copy_from_host({1.0f, 1.0f});
  v_depths.copy_from_host({1.0f});
  v_conics.copy_from_host({1.0f, 1.0f, 1.0f});

  // Gradient outputs (must be zeroed before backward)
  DeviceBuffer<float> v_means{3};
  DeviceBuffer<float> v_quats{4};
  DeviceBuffer<float> v_scales{3};
  v_means.copy_from_host({0.0f, 0.0f, 0.0f});
  v_quats.copy_from_host({0.0f, 0.0f, 0.0f, 0.0f});
  v_scales.copy_from_host({0.0f, 0.0f, 0.0f});

  launch_projection_ewa_3dgs_fused_backward(
    B, C, N,
    means.data(), nullptr, quats.data(), scales.data(),
    viewmats.data(), Ks.data(),
    128, 128, 0.3f, 0,
    radii.data(), conics.data(), nullptr,
    v_means2d.data(), v_depths.data(), v_conics.data(), nullptr,
    v_means.data(), nullptr, v_quats.data(), v_scales.data(),
    nullptr, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto h_v_means {v_means.copy_to_host()};
  bool any_nonzero {false};
  for (int i = 0; i < 3; ++i)
  {
    EXPECT_TRUE(std::isfinite(h_v_means[i]));
    if (h_v_means[i] != 0.0f) any_nonzero = true;
  }
  EXPECT_TRUE(any_nonzero);
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
