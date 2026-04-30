#include "Core/CubOperations.h"

#include <cassert>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace Core
{

void cub_inclusive_sum_int32_to_int64(
  const std::uint32_t count,
  const std::int32_t* d_input,
  std::int64_t* d_output,
  cudaStream_t stream)
{
  assert(d_input != nullptr);
  assert(d_output != nullptr);

  if (count == 0)
  {
    return;
  }

  void* d_temp_storage {nullptr};
  std::size_t temp_storage_bytes {0};

  cub::DeviceScan::InclusiveSum(
    d_temp_storage,
    temp_storage_bytes,
    d_input,
    d_output,
    static_cast<int>(count),
    stream);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceScan::InclusiveSum(
    d_temp_storage,
    temp_storage_bytes,
    d_input,
    d_output,
    static_cast<int>(count),
    stream);

  cudaFree(d_temp_storage);
}

void cub_radix_sort_pairs_int64_int32(
  const std::uint32_t count,
  const std::int64_t* d_keys_in,
  std::int64_t* d_keys_out,
  const std::int32_t* d_values_in,
  std::int32_t* d_values_out,
  const std::uint32_t begin_bit,
  const std::uint32_t end_bit,
  cudaStream_t stream)
{
  assert(d_keys_in != nullptr);
  assert(d_keys_out != nullptr);
  assert(d_values_in != nullptr);
  assert(d_values_out != nullptr);

  if (count == 0)
  {
    return;
  }

  void* d_temp_storage {nullptr};
  std::size_t temp_storage_bytes {0};

  cub::DeviceRadixSort::SortPairs(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    static_cast<int>(count),
    static_cast<int>(begin_bit),
    static_cast<int>(end_bit),
    stream);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceRadixSort::SortPairs(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    static_cast<int>(count),
    static_cast<int>(begin_bit),
    static_cast<int>(end_bit),
    stream);

  cudaFree(d_temp_storage);
}

} // namespace Core
