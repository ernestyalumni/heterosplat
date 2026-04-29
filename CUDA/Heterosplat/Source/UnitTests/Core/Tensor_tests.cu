#include "Core/Tensor.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

using Core::Tensor;

namespace GoogleUnitTests
{
namespace Core
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Tensor, DefaultConstructsWithNullDataAndEmptyShape)
{
  Tensor t;
  EXPECT_EQ(t.data(), nullptr);
  EXPECT_TRUE(t.shape().empty());
  EXPECT_EQ(t.stream(), nullptr);
  // number_of_elements() over an empty shape is the empty product, i.e. 1
  // (consistent with a 0-d / scalar interpretation). The default-constructed
  // tensor still has no allocation though — confirmed by data() == nullptr
  // above.
  EXPECT_EQ(t.number_of_elements(), 1u);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Tensor, ConstructsWithShapeAndAllocatesDeviceMemory)
{
  Tensor t{{4, 3}};
  EXPECT_NE(t.data(), nullptr);
  EXPECT_EQ(t.shape().size(), 2u);
  EXPECT_EQ(t.shape()[0], 4);
  EXPECT_EQ(t.shape()[1], 3);
  EXPECT_EQ(t.number_of_elements(), 12u);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Tensor, NumelMultipliesAllShapeDimensions)
{
  EXPECT_EQ(Tensor({5}).number_of_elements(), 5u);
  EXPECT_EQ(Tensor({2, 3}).number_of_elements(), 6u);
  EXPECT_EQ(Tensor({2, 3, 4}).number_of_elements(), 24u);
  EXPECT_EQ(Tensor({1, 1, 1, 1}).number_of_elements(), 1u);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Tensor, StoresStreamHandle)
{
  cudaStream_t stream {nullptr};
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  Tensor t{{2, 2}, stream};
  EXPECT_EQ(t.stream(), stream);

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

//------------------------------------------------------------------------------
/// h2d→d2h roundtrip preserves bit-identical float values.
//------------------------------------------------------------------------------
TEST(Tensor, CopyFromHostThenCopyToHostRoundtripsValues)
{
  const std::vector<float> source {
    -3.5f, 0.f, 1.0f, 2.5f,
    7.125f, -0.125f, 99.f, 1e-3f
  };
  Tensor t{{2, 4}};
  ASSERT_EQ(t.number_of_elements(), source.size());

  t.copy_from_host(source.data());

  std::vector<float> readback(source.size(), -1.f);
  t.copy_to_host(readback.data());

  for (std::size_t i {0}; i < source.size(); ++i)
  {
    EXPECT_EQ(readback[i], source[i]) << "i=" << i;
  }
}

//------------------------------------------------------------------------------
/// Larger buffer to catch off-by-one byte-count errors that pass on small
/// fixtures. Pattern is i + 0.5 so each element is unique and float-exact.
//------------------------------------------------------------------------------
TEST(Tensor, RoundtripsLargeBufferByteExact)
{
  constexpr std::size_t kN {10000};
  std::vector<float> source(kN);
  for (std::size_t i {0}; i < kN; ++i)
  {
    source[i] = static_cast<float>(i) + 0.5f;
  }

  Tensor t{{static_cast<int64_t>(kN)}};
  t.copy_from_host(source.data());

  std::vector<float> readback(kN, 0.f);
  t.copy_to_host(readback.data());

  for (std::size_t i {0}; i < kN; ++i)
  {
    ASSERT_EQ(readback[i], source[i]) << "i=" << i;
  }
}

//------------------------------------------------------------------------------
/// Move ctor transfers the device pointer; source becomes default-state so
/// its destructor must not double-free.
//------------------------------------------------------------------------------
TEST(Tensor, MoveConstructorTransfersOwnership)
{
  Tensor source{{3, 3}};
  float* original_data {source.data()};
  ASSERT_NE(original_data, nullptr);

  Tensor moved{std::move(source)};

  EXPECT_EQ(moved.data(), original_data);
  EXPECT_EQ(moved.number_of_elements(), 9u);
  EXPECT_EQ(source.data(), nullptr);
}

//------------------------------------------------------------------------------
/// Move-assigning over a live tensor must free the existing allocation
/// before taking the new one (no leak), and source must be reset to null.
//------------------------------------------------------------------------------
TEST(Tensor, MoveAssignmentReplacesExistingAllocation)
{
  Tensor target{{2, 2}};
  Tensor source{{4, 3}};
  float* source_data {source.data()};
  ASSERT_NE(target.data(), nullptr);
  ASSERT_NE(source_data, nullptr);

  target = std::move(source);

  EXPECT_EQ(target.data(), source_data);
  EXPECT_EQ(target.number_of_elements(), 12u);
  EXPECT_EQ(target.shape()[0], 4);
  EXPECT_EQ(target.shape()[1], 3);
  EXPECT_EQ(source.data(), nullptr);
}

//------------------------------------------------------------------------------
/// Self-move-assignment must not free the underlying allocation.
//------------------------------------------------------------------------------
TEST(Tensor, SelfMoveAssignmentIsSafe)
{
  Tensor t{{5}};
  float* original {t.data()};
  ASSERT_NE(original, nullptr);

  t = std::move(t);

  EXPECT_EQ(t.data(), original);
  EXPECT_EQ(t.number_of_elements(), 5u);

  // Confirm the allocation is still usable after the no-op self-move.
  std::vector<float> source(5, 42.f);
  t.copy_from_host(source.data());
  std::vector<float> readback(5, 0.f);
  t.copy_to_host(readback.data());
  for (float v : readback)
  {
    EXPECT_EQ(v, 42.f);
  }
}

//------------------------------------------------------------------------------
/// Const accessor on a const Tensor returns const float* and exposes the
/// same allocation. Compile-test more than runtime — but checks linkage.
//------------------------------------------------------------------------------
TEST(Tensor, ConstDataAccessorReturnsSamePointer)
{
  Tensor t{{2}};
  const Tensor& const_ref {t};
  EXPECT_EQ(const_ref.data(), t.data());
}

} // namespace Core
} // namespace GoogleUnitTests
