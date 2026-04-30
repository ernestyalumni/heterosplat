#ifndef UNIT_TESTS_DEVICE_BUFFER_H
#define UNIT_TESTS_DEVICE_BUFFER_H

#include <cstddef>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace GoogleUnitTests
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

} // namespace GoogleUnitTests

#endif // UNIT_TESTS_DEVICE_BUFFER_H
