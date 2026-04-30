#ifndef CORE_CUB_OPERATIONS_H
#define CORE_CUB_OPERATIONS_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Core
{

/// Inclusive prefix sum from int32 input to int64 output.
/// Accumulation happens in int64 to avoid overflow when tile counts are large.
void cub_inclusive_sum_int32_to_int64(
  std::uint32_t count,
  const std::int32_t* d_input,
  std::int64_t* d_output,
  cudaStream_t stream);

/// Radix sort key-value pairs: int64 keys with int32 values.
/// Used to sort intersection ids while keeping flatten_ids in sync.
void cub_radix_sort_pairs_int64_int32(
  std::uint32_t count,
  const std::int64_t* d_keys_in,
  std::int64_t* d_keys_out,
  const std::int32_t* d_values_in,
  std::int32_t* d_values_out,
  std::uint32_t begin_bit,
  std::uint32_t end_bit,
  cudaStream_t stream);

} // namespace Core

#endif // CORE_CUB_OPERATIONS_H
