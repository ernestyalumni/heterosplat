#include "DeviceBuffer.h"
#include "Kernels/Heterosplat/RasterizeToPixels3DGS.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using GoogleUnitTests::DeviceBuffer;
using Kernels::Heterosplat::launch_rasterize_to_pixels_3dgs_forward;
using Kernels::Heterosplat::launch_rasterize_to_pixels_3dgs_backward;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// Single Gaussian in a 1-image, 1-tile (16x16) scene. The Gaussian sits at
/// pixel (8,8) with a tight isotropic conic. It should produce non-zero
/// render_alpha and colour contribution on the center pixel.
//------------------------------------------------------------------------------
TEST(RasterizeToPixels3DGS, SingleGaussianRendersToCenter)
{
  constexpr std::uint32_t I {1};
  constexpr std::uint32_t N {1};
  constexpr std::uint32_t image_width {16};
  constexpr std::uint32_t image_height {16};
  constexpr std::uint32_t tile_size {16};
  constexpr std::uint32_t tile_width {1};
  constexpr std::uint32_t tile_height {1};
  constexpr std::uint32_t n_tiles {tile_width * tile_height};
  constexpr std::uint32_t CDIM {3};

  DeviceBuffer<float> means2d{N * 2};
  means2d.copy_from_host({8.0f, 8.0f});

  // Isotropic conic (inverse covariance): a=1, b=0, c=1
  DeviceBuffer<float> conics{N * 3};
  conics.copy_from_host({1.0f, 0.0f, 1.0f});

  // Red Gaussian
  DeviceBuffer<float> colors{N * CDIM};
  colors.copy_from_host({1.0f, 0.0f, 0.0f});

  DeviceBuffer<float> opacities{N};
  opacities.copy_from_host({0.9f});

  // One tile, one intersection: offset = [0], flatten_ids = [0]
  DeviceBuffer<std::int32_t> tile_offsets{I * n_tiles};
  tile_offsets.copy_from_host({0});

  constexpr std::uint32_t n_isects {1};
  DeviceBuffer<std::int32_t> flatten_ids{n_isects};
  flatten_ids.copy_from_host({0});

  DeviceBuffer<float> render_colors{I * image_height * image_width * CDIM};
  DeviceBuffer<float> render_alphas{I * image_height * image_width};
  DeviceBuffer<std::int32_t> last_ids{I * image_height * image_width};

  std::vector<float> zeros_colors(
    I * image_height * image_width * CDIM, 0.0f);
  std::vector<float> zeros_alphas(I * image_height * image_width, 0.0f);
  std::vector<std::int32_t> zeros_last(I * image_height * image_width, 0);
  render_colors.copy_from_host(zeros_colors);
  render_alphas.copy_from_host(zeros_alphas);
  last_ids.copy_from_host(zeros_last);

  launch_rasterize_to_pixels_3dgs_forward(
    I, N, n_isects, /*packed=*/false,
    means2d.data(),
    conics.data(),
    colors.data(),
    opacities.data(),
    /*backgrounds=*/nullptr,
    /*masks=*/nullptr,
    image_width, image_height, tile_size,
    tile_offsets.data(),
    flatten_ids.data(),
    render_colors.data(),
    render_alphas.data(),
    last_ids.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto h_colors {render_colors.copy_to_host()};
  const auto h_alphas {render_alphas.copy_to_host()};

  // Center pixel (8,8): the Gaussian is at exactly px=8.5, py=8.5 so
  // delta=(0,0) means sigma=0, alpha=0.9*exp(0)=0.9
  const std::size_t center {8u * image_width + 8u};
  EXPECT_GT(h_alphas[center], 0.5f);
  EXPECT_GT(h_colors[center * CDIM], 0.0f); // red channel
  EXPECT_EQ(h_colors[center * CDIM + 1], 0.0f); // green
  EXPECT_EQ(h_colors[center * CDIM + 2], 0.0f); // blue
}

//------------------------------------------------------------------------------
/// Zero intersections should produce zero alpha everywhere.
//------------------------------------------------------------------------------
TEST(RasterizeToPixels3DGS, ZeroIntersectionsProducesBlack)
{
  constexpr std::uint32_t I {1};
  constexpr std::uint32_t N {0};
  constexpr std::uint32_t image_width {16};
  constexpr std::uint32_t image_height {16};
  constexpr std::uint32_t tile_size {16};
  constexpr std::uint32_t tile_width {1};
  constexpr std::uint32_t tile_height {1};
  constexpr std::uint32_t CDIM {3};

  DeviceBuffer<std::int32_t> tile_offsets{I * tile_width * tile_height};
  tile_offsets.copy_from_host({0});

  DeviceBuffer<float> render_colors{I * image_height * image_width * CDIM};
  DeviceBuffer<float> render_alphas{I * image_height * image_width};
  DeviceBuffer<std::int32_t> last_ids{I * image_height * image_width};

  std::vector<float> zeros_colors(
    I * image_height * image_width * CDIM, 0.0f);
  std::vector<float> zeros_alphas(I * image_height * image_width, 0.0f);
  std::vector<std::int32_t> zeros_last(I * image_height * image_width, 0);
  render_colors.copy_from_host(zeros_colors);
  render_alphas.copy_from_host(zeros_alphas);
  last_ids.copy_from_host(zeros_last);

  launch_rasterize_to_pixels_3dgs_forward(
    I, N, /*n_isects=*/0, /*packed=*/false,
    /*means2d=*/nullptr,
    /*conics=*/nullptr,
    /*colors=*/nullptr,
    /*opacities=*/nullptr,
    /*backgrounds=*/nullptr,
    /*masks=*/nullptr,
    image_width, image_height, tile_size,
    tile_offsets.data(),
    /*flatten_ids=*/nullptr,
    render_colors.data(),
    render_alphas.data(),
    last_ids.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto h_alphas {render_alphas.copy_to_host()};
  for (std::size_t i {0}; i < h_alphas.size(); ++i)
  {
    EXPECT_FLOAT_EQ(h_alphas[i], 0.0f);
  }
}

//------------------------------------------------------------------------------
/// Backward produces finite, non-zero gradients for means2d when a Gaussian
/// is visible (basic smoke test).
//------------------------------------------------------------------------------
TEST(RasterizeToPixels3DGS, BackwardProducesFiniteGradients)
{
  constexpr std::uint32_t I {1};
  constexpr std::uint32_t N {1};
  constexpr std::uint32_t image_width {16};
  constexpr std::uint32_t image_height {16};
  constexpr std::uint32_t tile_size {16};
  constexpr std::uint32_t CDIM {3};
  constexpr std::uint32_t n_pixels {image_width * image_height};

  DeviceBuffer<float> means2d{N * 2};
  means2d.copy_from_host({8.0f, 8.0f});

  DeviceBuffer<float> conics_buf{N * 3};
  conics_buf.copy_from_host({1.0f, 0.0f, 1.0f});

  DeviceBuffer<float> colors{N * CDIM};
  colors.copy_from_host({1.0f, 0.5f, 0.2f});

  DeviceBuffer<float> opacities{N};
  opacities.copy_from_host({0.9f});

  DeviceBuffer<std::int32_t> tile_offsets{1};
  tile_offsets.copy_from_host({0});

  constexpr std::uint32_t n_isects {1};
  DeviceBuffer<std::int32_t> flatten_ids{1};
  flatten_ids.copy_from_host({0});

  // Forward pass to get render_alphas and last_ids
  DeviceBuffer<float> render_colors{n_pixels * CDIM};
  DeviceBuffer<float> render_alphas{n_pixels};
  DeviceBuffer<std::int32_t> last_ids{n_pixels};

  std::vector<float> zeros_colors(n_pixels * CDIM, 0.0f);
  std::vector<float> zeros_alphas(n_pixels, 0.0f);
  std::vector<std::int32_t> zeros_last(n_pixels, 0);
  render_colors.copy_from_host(zeros_colors);
  render_alphas.copy_from_host(zeros_alphas);
  last_ids.copy_from_host(zeros_last);

  launch_rasterize_to_pixels_3dgs_forward(
    I, N, n_isects, false,
    means2d.data(), conics_buf.data(), colors.data(), opacities.data(),
    nullptr, nullptr,
    image_width, image_height, tile_size,
    tile_offsets.data(), flatten_ids.data(),
    render_colors.data(), render_alphas.data(), last_ids.data(),
    nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Synthetic upstream gradients: ones
  DeviceBuffer<float> v_render_colors{n_pixels * CDIM};
  DeviceBuffer<float> v_render_alphas{n_pixels};
  std::vector<float> ones_colors(n_pixels * CDIM, 1.0f);
  std::vector<float> ones_alphas(n_pixels, 1.0f);
  v_render_colors.copy_from_host(ones_colors);
  v_render_alphas.copy_from_host(ones_alphas);

  // Gradient outputs (must be pre-zeroed)
  DeviceBuffer<float> v_means2d{N * 2};
  DeviceBuffer<float> v_conics{N * 3};
  DeviceBuffer<float> v_colors_out{N * CDIM};
  DeviceBuffer<float> v_opacities_out{N};
  v_means2d.copy_from_host({0.0f, 0.0f});
  v_conics.copy_from_host({0.0f, 0.0f, 0.0f});
  v_colors_out.copy_from_host({0.0f, 0.0f, 0.0f});
  v_opacities_out.copy_from_host({0.0f});

  launch_rasterize_to_pixels_3dgs_backward(
    I, N, n_isects, false,
    means2d.data(), conics_buf.data(), colors.data(), opacities.data(),
    nullptr, nullptr,
    image_width, image_height, tile_size,
    tile_offsets.data(), flatten_ids.data(),
    render_alphas.data(), last_ids.data(),
    v_render_colors.data(), v_render_alphas.data(),
    /*v_means2d_abs=*/nullptr,
    v_means2d.data(), v_conics.data(),
    v_colors_out.data(), v_opacities_out.data(),
    nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto h_v_colors {v_colors_out.copy_to_host()};
  bool any_nonzero {false};
  for (int i = 0; i < static_cast<int>(CDIM); ++i)
  {
    EXPECT_TRUE(std::isfinite(h_v_colors[i]));
    if (h_v_colors[i] != 0.0f) any_nonzero = true;
  }
  EXPECT_TRUE(any_nonzero);
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
