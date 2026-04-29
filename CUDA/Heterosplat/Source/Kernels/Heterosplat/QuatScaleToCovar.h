#ifndef KERNELS_HETEROSPLAT_QUAT_SCALE_TO_COVAR_H
#define KERNELS_HETEROSPLAT_QUAT_SCALE_TO_COVAR_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `quat_scale_to_covar_preci`
///        forward pass.
///
/// Given a batch of N 3D Gaussians parameterised by a quaternion `q` and a
/// per-axis scale vector `s`, computes the covariance matrix `Sigma` and/or
/// the precision matrix `Pi = Sigma^{-1}`:
///
///     R = R(q)            (3x3 rotation built from the unit quaternion)
///     S = diag(s)         (3x3 diagonal of per-axis scales)
///     Sigma = R * S^2 * R^T
///     Pi    = R * S^{-2} * R^T
///
/// The math is documented in `Documents/LaTeX/KernelMathematics.tex`,
/// section "Quaternion + scale to covariance / precision". This launcher
/// is a thin host-side wrapper around the templated `__global__` kernel
/// `gsplat::quat_scale_to_covar_preci_fwd_kernel<float>`, vendored in
/// `../Thirdparty/Gsplat/QuatScaleToCovarKernels.cuh` (Apache-2.0). We
/// replace gsplat's torch-coupled launcher (`at::Tensor`,
/// `AT_DISPATCH_FLOATING_TYPES`, `at::cuda::getCurrentCUDAStream()`) with
/// this raw-pointer interface so heterosplat builds without libtorch.
///
/// Float-only (no half/bfloat16 path); the underlying kernel is templated
/// but we instantiate it for `float` only at the launch site.
///
/// Naming: we keep gsplat's canonical operation identifier
/// `quat_scale_to_covar_preci` for grep-ability across the vendoring
/// boundary; we spell out `_forward` / `_backward` (instead of
/// `_fwd` / `_bwd`) for our own launcher symbols, matching heterosplat's
/// no-abbreviations style.
///
/// \param N         Number of Gaussians in the batch (one per `(q, s)` pair).
///                  When `N == 0` the launch is skipped (no-op).
/// \param quats     Device pointer, row-major shape `[N, 4]`. Each row is one
///                  quaternion stored real-part first as `(w, x, y, z)`. Need
///                  not be unit length: the kernel normalises in-line via
///                  `rsqrt(w^2 + x^2 + y^2 + z^2)`.
/// \param scales    Device pointer, row-major shape `[N, 3]`. Each row is a
///                  positive per-axis scale `(s1, s2, s3)`; geometrically the
///                  standard deviations of the Gaussian along its principal
///                  axes (the rotated frame defined by `R(q)`).
/// \param triu      Output layout selector for `covars` and `precis`:
///                    - `true`  -> `[N, 6]` upper-triangle order
///                                 `(xx, xy, xz, yy, yz, zz)`.
///                    - `false` -> `[N, 9]` row-major full `3x3`
///                                 (the kernel transposes on write because
///                                 GLM stores matrices column-major).
/// \param covars    Device output pointer for the covariances. Shape is
///                  `[N, 6]` or `[N, 9]` per `triu`. Pass `nullptr` to skip
///                  the covariance branch entirely (the per-Gaussian inner
///                  loop checks the pointer and does not write).
/// \param precis    Device output pointer for the precisions, same shape /
///                  `nullptr`-skip semantics as `covars`. Useful when only
///                  one of `Sigma` / `Pi` is needed downstream.
/// \param stream    CUDA stream to enqueue the launch on. `nullptr`
///                  (equivalently `0`) selects the default (legacy) stream.
///                  Use a non-default stream when overlapping this kernel
///                  with concurrent host->device copies or other GPU work
///                  in a training loop.
///
/// \pre `N > 0` AND at least one of `covars`, `precis` is non-null
///      (otherwise the launch is skipped and no work is performed).
/// \pre All `s_k > 0` whenever `precis != nullptr`. The precision branch
///      forms `1 / s_k`; a kernel-side `assert` guards this. Production
///      pipelines drop degenerate Gaussians (frustum culling, minimum-
///      radius filter) upstream so the assert never fires in practice.
///
/// Output is a "write" not an "accumulate": the kernel overwrites whatever
/// was in `covars` / `precis` previously.
//------------------------------------------------------------------------------
void launch_quat_scale_to_covar_preci_forward(
  std::uint32_t N,
  const float* quats,
  const float* scales,
  bool triu,
  float* covars,
  float* precis,
  cudaStream_t stream);

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `quat_scale_to_covar_preci`
///        backward pass — the vector-Jacobian product (VJP).
///
/// Given upstream gradients of the loss `L` with respect to the forward
/// outputs (`v_covars = dL/dSigma`, `v_precis = dL/dPi`), computes the
/// gradients with respect to the forward inputs:
///
///     v_quats  = dL/dq
///     v_scales = dL/ds
///
/// using the matrix-calculus identities derived in
/// `Documents/LaTeX/KernelMathematics.tex`, section "The backward map".
/// In short, with `M = R * S`:
///
///     dL/dM   = (G + G^T) * M           where G = dL/dSigma
///     dL/dR   = (dL/dM) * S
///     dL/ds_k = sum_i R_{ik} * (dL/dM)_{ik}        (for each k = 1..3)
///     dL/dq   = quat_to_rotmat_vjp(q, dL/dR)
///
/// The precision branch follows the same structure with `M' = R * S^{-1}`
/// and an extra `-1/s_k^2` chain-rule factor for the inversion of `s`.
///
/// \par The `v_` prefix
/// Convention (inherited from gsplat / common ML practice): `v_X` reads as
/// "gradient flowing through `X`", i.e. `dL/dX`. The `v` is short for
/// "vector-Jacobian product"; equivalently it is the cotangent / pulled-back
/// covector dual to the tangent direction `dX`.
///
/// \par Why we re-pass `quats`/`scales`
/// gsplat's PyTorch path stashes the forward inputs in autograd's saved-
/// tensor cache so they are available for backward without recomputation.
/// We have no such cache: the caller passes them again, and the kernel
/// recomputes `R(q)` and `M = R * S` to apply the chain rule. This trades
/// O(N) extra arithmetic for not having to materialise saved tensors —
/// negligible compared to the projection / rasterizer kernels later in the
/// pipeline.
///
/// \param N           Batch size. Must match the value passed to the
///                    forward call. `N == 0` skips the launch.
/// \param quats       Device, `[N, 4]` — same quaternions used in forward.
/// \param scales      Device, `[N, 3]` — same scales used in forward.
/// \param triu        Layout of `v_covars` / `v_precis`. Must match the
///                    layout used by the forward call. When `true`, the
///                    kernel applies a `0.5` factor when expanding the
///                    `[6]` triu vector to a symmetric `3x3` matrix —
///                    correcting for the double-counting of off-diagonals
///                    (each off-diagonal `g_{ij}` for `i != j` would
///                    otherwise contribute to both `Sigma_{ij}` and
///                    `Sigma_{ji}`).
/// \param v_covars    Device, INPUT gradient `dL/dSigma`. Shape matches
///                    the forward output: `[N, 6]` or `[N, 9]`. Pass
///                    `nullptr` if `Sigma` did not feed into the loss this
///                    step — the kernel skips the covariance-side chain
///                    rule.
/// \param v_precis    Device, INPUT gradient `dL/dPi`. Same shape /
///                    `nullptr`-skip semantics as `v_covars`.
/// \param v_quats     Device, OUTPUT gradient `dL/dq`, shape `[N, 4]`.
///                    The kernel WRITES (does not accumulate). If the
///                    caller wants `+=` semantics they must do the
///                    accumulate themselves.
/// \param v_scales    Device, OUTPUT gradient `dL/ds`, shape `[N, 3]`.
///                    Same write-not-accumulate semantics as `v_quats`.
/// \param stream      CUDA stream (`nullptr` -> default).
///
/// \pre At least one of `v_covars`, `v_precis` is non-null. If both are
///      `nullptr` the launch is skipped and `v_quats` / `v_scales` are
///      LEFT UNTOUCHED — caller must zero them out beforehand if "no
///      gradient" should mean a literal zero rather than stale memory.
//------------------------------------------------------------------------------
void launch_quat_scale_to_covar_preci_backward(
  std::uint32_t N,
  const float* quats,
  const float* scales,
  bool triu,
  const float* v_covars,
  const float* v_precis,
  float* v_quats,
  float* v_scales,
  cudaStream_t stream);

} // namespace Heterosplat
} // namespace Kernels

#endif // KERNELS_HETEROSPLAT_QUAT_SCALE_TO_COVAR_H
