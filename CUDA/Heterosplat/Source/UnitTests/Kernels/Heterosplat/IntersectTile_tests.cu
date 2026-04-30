#include "DeviceBuffer.h"
#include "Kernels/Heterosplat/IntersectTile.h"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using GoogleUnitTests::DeviceBuffer;
using Kernels::Heterosplat::launch_intersect_tile_forward;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

namespace
{

std::int64_t expected_intersection_id(
  const std::int64_t image_id,
  const std::int64_t tile_id,
  const float depth,
  const std::uint32_t tile_n_bits)
{
  std::uint32_t depth_bits {0};
  std::memcpy(&depth_bits, &depth, sizeof(depth_bits));
  return (image_id << (32u + tile_n_bits)) |
         (tile_id << 32u) |
         static_cast<std::int64_t>(depth_bits);
}

} // namespace

//------------------------------------------------------------------------------
/// Dense AABB path: one Gaussian touches a 2x2 block of tiles and one Gaussian
/// has an invalid radius. The second pass should emit four unsorted
/// intersections in row-major tile order with flatten id 0.
//------------------------------------------------------------------------------
TEST(IntersectTile, DenseAabbTwoPassWritesExpectedIntersections)
{
  constexpr std::uint32_t I {1};
  constexpr std::uint32_t N {2};
  constexpr std::uint32_t tile_size {8};
  constexpr std::uint32_t tile_width {4};
  constexpr std::uint32_t tile_height {4};
  constexpr std::uint32_t tile_n_bits {5}; // floor(log2(16)) + 1

  DeviceBuffer<float> means2d{I * N * 2};
  DeviceBuffer<std::int32_t> radii{I * N * 2};
  DeviceBuffer<float> depths{I * N};
  DeviceBuffer<std::int32_t> tiles_per_gauss{I * N};

  means2d.copy_from_host({8.0f, 8.0f, 20.0f, 20.0f});
  radii.copy_from_host({8, 8, 0, 4});
  depths.copy_from_host({1.0f, 2.0f});

  launch_intersect_tile_forward(
    /*packed=*/false,
    I,
    N,
    /*nnz=*/0,
    /*image_ids=*/nullptr,
    /*gaussian_ids=*/nullptr,
    means2d.data(),
    radii.data(),
    depths.data(),
    /*conics=*/nullptr,
    /*opacities=*/nullptr,
    /*cum_tiles_per_gauss=*/nullptr,
    tile_size,
    tile_width,
    tile_height,
    tiles_per_gauss.data(),
    /*isect_ids=*/nullptr,
    /*flatten_ids=*/nullptr,
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(tiles_per_gauss.copy_to_host(), (std::vector<std::int32_t>{4, 0}));

  DeviceBuffer<std::int64_t> cum_tiles_per_gauss{I * N};
  DeviceBuffer<std::int64_t> isect_ids{4};
  DeviceBuffer<std::int32_t> flatten_ids{4};
  cum_tiles_per_gauss.copy_from_host({4, 4});

  launch_intersect_tile_forward(
    /*packed=*/false,
    I,
    N,
    /*nnz=*/0,
    /*image_ids=*/nullptr,
    /*gaussian_ids=*/nullptr,
    means2d.data(),
    radii.data(),
    depths.data(),
    /*conics=*/nullptr,
    /*opacities=*/nullptr,
    cum_tiles_per_gauss.data(),
    tile_size,
    tile_width,
    tile_height,
    /*tiles_per_gauss=*/nullptr,
    isect_ids.data(),
    flatten_ids.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(
    isect_ids.copy_to_host(),
    (std::vector<std::int64_t>{
      expected_intersection_id(0, 0, 1.0f, tile_n_bits),
      expected_intersection_id(0, 1, 1.0f, tile_n_bits),
      expected_intersection_id(0, 4, 1.0f, tile_n_bits),
      expected_intersection_id(0, 5, 1.0f, tile_n_bits)}));
  EXPECT_EQ(flatten_ids.copy_to_host(), (std::vector<std::int32_t>{0, 0, 0, 0}));
}

//------------------------------------------------------------------------------
/// Packed mode reads image ids from the `image_ids` array and still reports the
/// packed row index as `flatten_ids`.
//------------------------------------------------------------------------------
TEST(IntersectTile, PackedAabbEncodesImageIds)
{
  constexpr std::uint32_t I {3};
  constexpr std::uint32_t nnz {2};
  constexpr std::uint32_t tile_size {8};
  constexpr std::uint32_t tile_width {4};
  constexpr std::uint32_t tile_height {4};
  constexpr std::uint32_t tile_n_bits {5};

  DeviceBuffer<std::int64_t> image_ids{nnz};
  DeviceBuffer<std::int64_t> gaussian_ids{nnz};
  DeviceBuffer<float> means2d{nnz * 2};
  DeviceBuffer<std::int32_t> radii{nnz * 2};
  DeviceBuffer<float> depths{nnz};
  DeviceBuffer<std::int32_t> tiles_per_gauss{nnz};

  image_ids.copy_from_host({2, 1});
  gaussian_ids.copy_from_host({7, 8});
  means2d.copy_from_host({4.0f, 4.0f, 12.0f, 4.0f});
  radii.copy_from_host({4, 4, 4, 4});
  depths.copy_from_host({1.0f, 3.0f});

  launch_intersect_tile_forward(
    /*packed=*/true,
    I,
    /*N=*/0,
    nnz,
    image_ids.data(),
    gaussian_ids.data(),
    means2d.data(),
    radii.data(),
    depths.data(),
    /*conics=*/nullptr,
    /*opacities=*/nullptr,
    /*cum_tiles_per_gauss=*/nullptr,
    tile_size,
    tile_width,
    tile_height,
    tiles_per_gauss.data(),
    /*isect_ids=*/nullptr,
    /*flatten_ids=*/nullptr,
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(tiles_per_gauss.copy_to_host(), (std::vector<std::int32_t>{1, 1}));

  DeviceBuffer<std::int64_t> cum_tiles_per_gauss{nnz};
  DeviceBuffer<std::int64_t> isect_ids{2};
  DeviceBuffer<std::int32_t> flatten_ids{2};
  cum_tiles_per_gauss.copy_from_host({1, 2});

  launch_intersect_tile_forward(
    /*packed=*/true,
    I,
    /*N=*/0,
    nnz,
    image_ids.data(),
    gaussian_ids.data(),
    means2d.data(),
    radii.data(),
    depths.data(),
    /*conics=*/nullptr,
    /*opacities=*/nullptr,
    cum_tiles_per_gauss.data(),
    tile_size,
    tile_width,
    tile_height,
    /*tiles_per_gauss=*/nullptr,
    isect_ids.data(),
    flatten_ids.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(
    isect_ids.copy_to_host(),
    (std::vector<std::int64_t>{
      expected_intersection_id(2, 0, 1.0f, tile_n_bits),
      expected_intersection_id(1, 1, 3.0f, tile_n_bits)}));
  EXPECT_EQ(flatten_ids.copy_to_host(), (std::vector<std::int32_t>{0, 1}));
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
