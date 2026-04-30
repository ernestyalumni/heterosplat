#include "DeviceBuffer.h"
#include "Kernels/Heterosplat/IntersectOffset.h"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using GoogleUnitTests::DeviceBuffer;
using Kernels::Heterosplat::launch_intersect_offset_forward;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

namespace
{

std::int64_t make_isect_id(
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
/// Single image, 2x2 tile grid, 3 intersections touching tiles 0, 1, 1.
/// Expected offsets: [0, 1, 3, 3].
//------------------------------------------------------------------------------
TEST(IntersectOffset, SingleImageProducesCorrectOffsets)
{
  constexpr std::uint32_t I {1};
  constexpr std::uint32_t tile_width {2};
  constexpr std::uint32_t tile_height {2};
  constexpr std::uint32_t n_tiles {tile_width * tile_height};
  constexpr std::uint32_t tile_n_bits {3}; // floor(log2(4)) + 1 = 3

  const std::vector<std::int64_t> h_isect_ids {
    make_isect_id(0, 0, 1.0f, tile_n_bits),
    make_isect_id(0, 1, 2.0f, tile_n_bits),
    make_isect_id(0, 1, 3.0f, tile_n_bits),
  };
  const std::uint32_t n_isects {static_cast<std::uint32_t>(h_isect_ids.size())};

  DeviceBuffer<std::int64_t> d_isect_ids{n_isects};
  DeviceBuffer<std::int32_t> d_offsets{I * n_tiles};
  d_isect_ids.copy_from_host(h_isect_ids);

  launch_intersect_offset_forward(
    n_isects,
    d_isect_ids.data(),
    I,
    tile_width,
    tile_height,
    d_offsets.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // tile 0: starts at 0, tile 1: starts at 1, tiles 2,3: start at 3 (end)
  EXPECT_EQ(
    d_offsets.copy_to_host(),
    (std::vector<std::int32_t>{0, 1, 3, 3}));
}

//------------------------------------------------------------------------------
/// Two images, 2x1 tile grid. Image 0 has one intersection in tile 0;
/// image 1 has two intersections in tile 1. Offsets shape is [I * n_tiles] = 4.
//------------------------------------------------------------------------------
TEST(IntersectOffset, MultiImageEncodesImageBoundaries)
{
  constexpr std::uint32_t I {2};
  constexpr std::uint32_t tile_width {2};
  constexpr std::uint32_t tile_height {1};
  constexpr std::uint32_t n_tiles {tile_width * tile_height};
  constexpr std::uint32_t tile_n_bits {2}; // floor(log2(2)) + 1 = 2

  const std::vector<std::int64_t> h_isect_ids {
    make_isect_id(0, 0, 1.0f, tile_n_bits),
    make_isect_id(1, 1, 2.0f, tile_n_bits),
    make_isect_id(1, 1, 3.0f, tile_n_bits),
  };
  const std::uint32_t n_isects {static_cast<std::uint32_t>(h_isect_ids.size())};

  DeviceBuffer<std::int64_t> d_isect_ids{n_isects};
  DeviceBuffer<std::int32_t> d_offsets{I * n_tiles};
  d_isect_ids.copy_from_host(h_isect_ids);

  launch_intersect_offset_forward(
    n_isects,
    d_isect_ids.data(),
    I,
    tile_width,
    tile_height,
    d_offsets.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Flat layout [img0_tile0, img0_tile1, img1_tile0, img1_tile1]:
  //   img0_tile0: starts at 0
  //   img0_tile1: starts at 1 (no intersections, same as next)
  //   img1_tile0: starts at 1 (no intersections, same as next)
  //   img1_tile1: starts at 1
  EXPECT_EQ(
    d_offsets.copy_to_host(),
    (std::vector<std::int32_t>{0, 1, 1, 1}));
}

//------------------------------------------------------------------------------
/// Zero intersections: all offsets should be zero.
//------------------------------------------------------------------------------
TEST(IntersectOffset, ZeroIntersectionsProducesAllZeros)
{
  constexpr std::uint32_t I {1};
  constexpr std::uint32_t tile_width {3};
  constexpr std::uint32_t tile_height {2};
  constexpr std::uint32_t n_tiles {tile_width * tile_height};

  DeviceBuffer<std::int32_t> d_offsets{I * n_tiles};

  // Fill with non-zero to confirm the launcher zeros them
  const std::vector<std::int32_t> garbage(I * n_tiles, 99);
  d_offsets.copy_from_host(garbage);

  launch_intersect_offset_forward(
    /*n_isects=*/0,
    /*isect_ids=*/nullptr,
    I,
    tile_width,
    tile_height,
    d_offsets.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::vector<std::int32_t> expected(I * n_tiles, 0);
  EXPECT_EQ(d_offsets.copy_to_host(), expected);
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
