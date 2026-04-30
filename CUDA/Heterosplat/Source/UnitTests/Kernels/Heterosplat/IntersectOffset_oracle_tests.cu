#include "DeviceBuffer.h"
#include "Kernels/Heterosplat/IntersectOffset.h"
#include "OracleFixture.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using GoogleUnitTests::DeviceBuffer;
using Kernels::Heterosplat::launch_intersect_offset_forward;
using GoogleUnitTests::OracleFixture::fixture_path;
using GoogleUnitTests::OracleFixture::load_int32s;
using GoogleUnitTests::OracleFixture::load_int64s;
using GoogleUnitTests::OracleFixture::load_uint32;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// Oracle comparison against gsplat-Python's `isect_offset_encode()`.
/// Uses sorted isect_ids from `isect_tiles(..., sort=True)`.
//------------------------------------------------------------------------------
TEST(IntersectOffsetOracle, ForwardMatchesGsplatPython)
{
  const std::uint32_t I {load_uint32(fixture_path("IntersectOffset", "I.bin"))};
  const std::uint32_t n_tiles {
    load_uint32(fixture_path("IntersectOffset", "n_tiles.bin"))};
  const std::uint32_t tile_width {
    load_uint32(fixture_path("IntersectOffset", "tile_width.bin"))};
  const std::uint32_t tile_height {
    load_uint32(fixture_path("IntersectOffset", "tile_height.bin"))};
  const std::uint32_t n_isects {
    load_uint32(fixture_path("IntersectOffset", "n_isects.bin"))};

  ASSERT_EQ(n_tiles, tile_width * tile_height);

  const auto h_isect_ids_sorted {
    load_int64s(fixture_path("IntersectOffset", "isect_ids_sorted.bin"))};
  const auto h_expected_offsets {
    load_int32s(fixture_path("IntersectOffset", "offsets.bin"))};

  ASSERT_EQ(h_isect_ids_sorted.size(), n_isects);
  ASSERT_EQ(h_expected_offsets.size(), I * n_tiles);

  DeviceBuffer<std::int64_t> d_isect_ids_sorted{n_isects};
  DeviceBuffer<std::int32_t> d_offsets{I * n_tiles};

  d_isect_ids_sorted.copy_from_host(h_isect_ids_sorted);

  launch_intersect_offset_forward(
    n_isects,
    d_isect_ids_sorted.data(),
    I,
    tile_width,
    tile_height,
    d_offsets.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_offsets.copy_to_host(), h_expected_offsets);
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
