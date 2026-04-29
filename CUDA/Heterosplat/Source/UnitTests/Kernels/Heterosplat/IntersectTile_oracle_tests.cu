#include "Kernels/Heterosplat/IntersectTile.h"
#include "OracleFixture.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using Kernels::Heterosplat::launch_intersect_tile_forward;
using GoogleUnitTests::OracleFixture::fixture_path;
using GoogleUnitTests::OracleFixture::load_floats;
using GoogleUnitTests::OracleFixture::load_int32s;
using GoogleUnitTests::OracleFixture::load_int64s;
using GoogleUnitTests::OracleFixture::load_uint32;

namespace GoogleUnitTests
{
namespace Kernels
{
namespace Heterosplat
{

namespace
{

template <typename T>
class DeviceBuffer
{
  public:
    explicit DeviceBuffer(const std::size_t count):
      count_{count}
    {
      const cudaError_t status {
        cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T))};
      if (status != cudaSuccess)
      {
        throw std::runtime_error{cudaGetErrorString(status)};
      }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
      cudaFree(data_);
    }

    void copy_from_host(const std::vector<T>& host)
    {
      ASSERT_EQ(host.size(), count_);
      ASSERT_EQ(
        cudaMemcpy(data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
        cudaSuccess);
    }

    std::vector<T> copy_to_host() const
    {
      std::vector<T> host(count_);
      EXPECT_EQ(
        cudaMemcpy(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
        cudaSuccess);
      return host;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

  private:
    T* data_ {nullptr};
    std::size_t count_ {0};
};

} // namespace

//------------------------------------------------------------------------------
/// Oracle comparison against gsplat-Python's `isect_tiles(..., sort=False)`.
/// This exercises the same two-pass raw kernel path without CUB sorting.
//------------------------------------------------------------------------------
TEST(IntersectTileOracle, DenseAabbForwardMatchesGsplatPython)
{
  const std::uint32_t I {load_uint32(fixture_path("IntersectTile", "I.bin"))};
  const std::uint32_t N {load_uint32(fixture_path("IntersectTile", "N.bin"))};
  const std::uint32_t tile_size {
    load_uint32(fixture_path("IntersectTile", "tile_size.bin"))};
  const std::uint32_t tile_width {
    load_uint32(fixture_path("IntersectTile", "tile_width.bin"))};
  const std::uint32_t tile_height {
    load_uint32(fixture_path("IntersectTile", "tile_height.bin"))};
  const std::uint32_t n_isects {
    load_uint32(fixture_path("IntersectTile", "n_isects.bin"))};

  const auto h_means2d {load_floats(fixture_path("IntersectTile", "means2d.bin"))};
  const auto h_radii {load_int32s(fixture_path("IntersectTile", "radii.bin"))};
  const auto h_depths {load_floats(fixture_path("IntersectTile", "depths.bin"))};
  const auto h_expected_tiles_per_gauss {
    load_int32s(fixture_path("IntersectTile", "tiles_per_gauss.bin"))};
  const auto h_cum_tiles_per_gauss {
    load_int64s(fixture_path("IntersectTile", "cum_tiles_per_gauss.bin"))};
  const auto h_expected_isect_ids {
    load_int64s(fixture_path("IntersectTile", "isect_ids.bin"))};
  const auto h_expected_flatten_ids {
    load_int32s(fixture_path("IntersectTile", "flatten_ids.bin"))};

  ASSERT_EQ(h_means2d.size(), I * N * 2u);
  ASSERT_EQ(h_radii.size(), I * N * 2u);
  ASSERT_EQ(h_depths.size(), I * N);
  ASSERT_EQ(h_expected_tiles_per_gauss.size(), I * N);
  ASSERT_EQ(h_cum_tiles_per_gauss.size(), I * N);
  ASSERT_EQ(h_expected_isect_ids.size(), n_isects);
  ASSERT_EQ(h_expected_flatten_ids.size(), n_isects);

  DeviceBuffer<float> d_means2d{h_means2d.size()};
  DeviceBuffer<std::int32_t> d_radii{h_radii.size()};
  DeviceBuffer<float> d_depths{h_depths.size()};
  DeviceBuffer<std::int32_t> d_tiles_per_gauss{h_expected_tiles_per_gauss.size()};

  d_means2d.copy_from_host(h_means2d);
  d_radii.copy_from_host(h_radii);
  d_depths.copy_from_host(h_depths);

  launch_intersect_tile_forward(
    /*packed=*/false,
    I,
    N,
    /*nnz=*/0,
    /*image_ids=*/nullptr,
    /*gaussian_ids=*/nullptr,
    d_means2d.data(),
    d_radii.data(),
    d_depths.data(),
    /*conics=*/nullptr,
    /*opacities=*/nullptr,
    /*cum_tiles_per_gauss=*/nullptr,
    tile_size,
    tile_width,
    tile_height,
    d_tiles_per_gauss.data(),
    /*isect_ids=*/nullptr,
    /*flatten_ids=*/nullptr,
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_tiles_per_gauss.copy_to_host(), h_expected_tiles_per_gauss);

  DeviceBuffer<std::int64_t> d_cum_tiles_per_gauss{h_cum_tiles_per_gauss.size()};
  DeviceBuffer<std::int64_t> d_isect_ids{h_expected_isect_ids.size()};
  DeviceBuffer<std::int32_t> d_flatten_ids{h_expected_flatten_ids.size()};
  d_cum_tiles_per_gauss.copy_from_host(h_cum_tiles_per_gauss);

  launch_intersect_tile_forward(
    /*packed=*/false,
    I,
    N,
    /*nnz=*/0,
    /*image_ids=*/nullptr,
    /*gaussian_ids=*/nullptr,
    d_means2d.data(),
    d_radii.data(),
    d_depths.data(),
    /*conics=*/nullptr,
    /*opacities=*/nullptr,
    d_cum_tiles_per_gauss.data(),
    tile_size,
    tile_width,
    tile_height,
    /*tiles_per_gauss=*/nullptr,
    d_isect_ids.data(),
    d_flatten_ids.data(),
    /*stream=*/nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_isect_ids.copy_to_host(), h_expected_isect_ids);
  EXPECT_EQ(d_flatten_ids.copy_to_host(), h_expected_flatten_ids);
}

} // namespace Heterosplat
} // namespace Kernels
} // namespace GoogleUnitTests
