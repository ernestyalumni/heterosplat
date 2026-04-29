#ifndef CORE_TENSOR_H
#define CORE_TENSOR_H

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Core
{

//------------------------------------------------------------------------------
/// \brief Owning device-side float buffer with shape + stream metadata.
///
/// Replaces at::Tensor in the gsplat launcher boundary. Float-only for
/// Phase 0a/0b — raw `float*` is what PLAN.md commits to. A future Phase will
/// likely templatise on scalar_t when half/bfloat16 paths matter.
///
/// Owns its device allocation. Move-only (no accidental deep-copy of GPU
/// memory). Stream is a non-owning handle — caller is responsible for stream
/// lifetime.
//------------------------------------------------------------------------------
class Tensor
{
  public:

    Tensor() = default;

    explicit Tensor(
      std::vector<int64_t> shape,
      cudaStream_t stream = nullptr):
      shape_{std::move(shape)},
      stream_{stream}
    {
      const std::size_t bytes {number_of_elements() * sizeof(float)};
      const cudaError_t status {cudaMalloc(
        reinterpret_cast<void**>(&data_), bytes)};
      if (status != cudaSuccess)
      {
        throw std::runtime_error(
          std::string{"Tensor cudaMalloc failed: "} +
          cudaGetErrorString(status));
      }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept:
      data_{other.data_},
      shape_{std::move(other.shape_)},
      stream_{other.stream_}
    {
      other.data_ = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept
    {
      if (this != &other)
      {
        free_resources();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        stream_ = other.stream_;
        other.data_ = nullptr;
      }
      return *this;
    }

    ~Tensor()
    {
      free_resources();
    }

    //--------------------------------------------------------------------------
    /// Number of elements implied by `shape_`.
    //--------------------------------------------------------------------------
    std::size_t number_of_elements() const
    {
      std::size_t n {1};
      for (const int64_t d : shape_)
      {
        n *= static_cast<std::size_t>(d);
      }
      return n;
    }

    //--------------------------------------------------------------------------
    /// Synchronous host->device copy. Caller-supplied `host` must hold
    /// `number_of_elements()` floats.
    //--------------------------------------------------------------------------
    void copy_from_host(const float* host)
    {
      const cudaError_t status {cudaMemcpy(
        data_,
        host,
        number_of_elements() * sizeof(float), cudaMemcpyHostToDevice)};
      if (status != cudaSuccess)
      {
        throw std::runtime_error(
          std::string{"Tensor copy_from_host failed: "} +
          cudaGetErrorString(status));
      }
    }

    //--------------------------------------------------------------------------
    /// Synchronous device->host copy. Caller-supplied `host` must hold
    /// `number_of_elements()` floats.
    //--------------------------------------------------------------------------
    void copy_to_host(float* host) const
    {
      const cudaError_t status {cudaMemcpy(
        host,
        data_,
        number_of_elements() * sizeof(float),
        cudaMemcpyDeviceToHost)};
      if (status != cudaSuccess)
      {
        throw std::runtime_error(
          std::string{"Tensor copy_to_host failed: "} +
          cudaGetErrorString(status));
      }
    }

    float* data() { return data_; }
    const float* data() const { return data_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    cudaStream_t stream() const { return stream_; }

  private:

    void free_resources()
    {
      if (data_ != nullptr)
      {
        const cudaError_t status {cudaFree(data_)};
        if (status != cudaSuccess)
        {
          std::cerr << "Tensor cudaFree failed: " <<
            cudaGetErrorString(status) << '\n';
        }
        data_ = nullptr;
      }
    }

    float* data_ {nullptr};
    std::vector<int64_t> shape_;
    cudaStream_t stream_ {nullptr};
};

} // namespace Core

#endif // CORE_TENSOR_H
