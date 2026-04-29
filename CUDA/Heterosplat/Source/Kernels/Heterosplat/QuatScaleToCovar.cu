#include "Kernels/Heterosplat/QuatScaleToCovar.h"
#include "Kernels/Thirdparty/Gsplat/QuatScaleToCovarKernels.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

namespace
{
//------------------------------------------------------------------------------
/// Threads per block for the `quat_scale_to_covar_preci` kernels. gsplat
/// upstream pins this at `dim3(256)`. We mirror that exactly so our
/// numerical results agree with the gsplat-Python oracle bit-for-bit; a
/// different block size would not change the math but could subtly shift
/// floating-point accumulation order in any future reduction-flavoured
/// variant of these kernels.
//------------------------------------------------------------------------------
constexpr std::uint32_t kThreadsPerBlock {256};
} // namespace

void launch_quat_scale_to_covar_preci_forward(
  const std::uint32_t N,
  const float* quats,
  const float* scales,
  const bool triu,
  float* covars,
  float* precis,
  cudaStream_t stream)
{
  // Empty batch: no kernel issued, no work done. Mirrors the upstream
  // launcher's `if (n_elements == 0) return;` guard.
  if (N == 0)
  {
    return;
  }

  // 1D grid over Gaussians: one CUDA thread per Gaussian. The kernel itself
  // does no inter-thread communication, so this is "embarrassingly parallel"
  // — the only purpose of the block size is to fit nicely into the SM's
  // warp scheduler (256 = 8 warps, one of the canonical choices).
  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {(N + kThreadsPerBlock - 1) / kThreadsPerBlock};

  // Explicit `<float>` instantiates the templated __global__ for our
  // float-only path. The vendored kernel template lives in
  // ../Thirdparty/Gsplat/QuatScaleToCovarKernels.cuh (Apache-2.0,
  // verbatim from upstream gsplat with the launcher stripped).
  gsplat::quat_scale_to_covar_preci_fwd_kernel<float><<<grid, threads, 0, stream>>>(
    N, quats, scales, triu, covars, precis);
}

void launch_quat_scale_to_covar_preci_backward(
  const std::uint32_t N,
  const float* quats,
  const float* scales,
  const bool triu,
  const float* v_covars,
  const float* v_precis,
  float* v_quats,
  float* v_scales,
  cudaStream_t stream)
{
  // Two ways the launch is a no-op:
  //   1. Empty batch.
  //   2. Neither output (Sigma nor Pi) contributed to the loss this step,
  //      so there is nothing for the chain rule to propagate. v_quats /
  //      v_scales are LEFT UNTOUCHED in this case — the caller is
  //      responsible for any zeroing if they expect "no gradient" to mean
  //      literal zero rather than stale memory.
  if (N == 0 || (v_covars == nullptr && v_precis == nullptr))
  {
    return;
  }

  const dim3 threads {kThreadsPerBlock};
  const dim3 grid {(N + kThreadsPerBlock - 1) / kThreadsPerBlock};

  gsplat::quat_scale_to_covar_preci_bwd_kernel<float><<<grid, threads, 0, stream>>>(
    N, quats, scales, triu, v_covars, v_precis, v_quats, v_scales);
}

} // namespace Heterosplat
} // namespace Kernels
