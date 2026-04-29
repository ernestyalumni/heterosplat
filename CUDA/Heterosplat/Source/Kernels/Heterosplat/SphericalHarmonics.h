#ifndef KERNELS_HETEROSPLAT_SPHERICAL_HARMONICS_H
#define KERNELS_HETEROSPLAT_SPHERICAL_HARMONICS_H

#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels
{
namespace Heterosplat
{

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `spherical_harmonics` forward.
///
/// Evaluates a 3-channel (RGB) view-direction-dependent colour for each of N
/// Gaussians from real spherical-harmonic (SH) coefficients of degree up to
/// `degrees_to_use`:
///
///     colors[n, c] = sum_{l=0..degrees_to_use, m=-l..l} c_{l m, n, c} * Y_l^m(dir_n)
///
/// The exact constants and recursion used are Sloan's "Efficient Spherical
/// Harmonic Evaluation" (JCGT 2013); the math is documented in
/// `Documents/LaTeX/KernelMathematics.tex` section "Spherical harmonics".
///
/// `degrees_to_use` selects how many degrees actually contribute; passing a
/// value smaller than implied by `K` lets the caller turn coarse-to-fine SH
/// training on/off without re-allocating coefficient buffers. Supported up
/// to degree 4 (K = 25 coefficients).
///
/// \param N                 Number of Gaussians.
/// \param K                 Coefficients per Gaussian; must be at least
///                          `(degrees_to_use + 1)^2` (1, 4, 9, 16, 25 for
///                          degrees 0..4). The kernel uses `K` as the stride
///                          between adjacent Gaussians' coefficient blocks.
/// \param degrees_to_use    SH degree to evaluate (0..4). Must be `<=`
///                          implied by `K`.
/// \param dirs              Device, `[N, 3]` row-major. View directions
///                          (need not be unit; the kernel normalises).
/// \param coeffs            Device, `[N, K, 3]` row-major. SH coefficients
///                          per Gaussian per channel.
/// \param masks             Device, `[N]` `bool` array; `nullptr` to evaluate
///                          all Gaussians. When non-null, threads with
///                          `masks[n] == false` short-circuit (output is
///                          left unchanged for that Gaussian).
/// \param colors            Device output, `[N, 3]` row-major.
/// \param stream            CUDA stream; `nullptr` -> default stream.
//------------------------------------------------------------------------------
void launch_spherical_harmonics_forward(
  std::uint32_t N,
  std::uint32_t K,
  std::uint32_t degrees_to_use,
  const float* dirs,
  const float* coeffs,
  const bool* masks,
  float* colors,
  cudaStream_t stream);

//------------------------------------------------------------------------------
/// \brief Raw-pointer launcher for gsplat's `spherical_harmonics` backward
///        (the VJP).
///
/// Given an upstream colour gradient `v_colors = dL/dcolors`, propagates to
/// the SH coefficients (`v_coeffs = dL/dcoeffs`) and, optionally, to the
/// view directions (`v_dirs = dL/ddirs`). The direction gradient uses
/// `atomicAdd` because all three colour channels share the same direction
/// (each thread handles one (Gaussian, channel) pair, and the three
/// per-channel contributions to the same `v_dir` accumulate).
///
/// \param N                 Number of Gaussians.
/// \param K                 Coefficients per Gaussian (stride).
/// \param degrees_to_use    SH degree (0..4). Must match the forward call.
/// \param dirs              Device, `[N, 3]`.
/// \param coeffs            Device, `[N, K, 3]`. Read-only forward input.
/// \param masks             Device, `[N]` `bool`; `nullptr` for "all".
/// \param v_colors          Device, INPUT gradient `dL/dcolors`, `[N, 3]`.
/// \param v_coeffs          Device, OUTPUT gradient `dL/dcoeffs`, `[N, K, 3]`.
///                          Kernel writes per (Gaussian, coefficient,
///                          channel); single-writer per slot, no atomics.
/// \param v_dirs            Device, OUTPUT gradient `dL/ddirs`, `[N, 3]`.
///                          Pass `nullptr` to skip the direction-gradient
///                          path (saves work and avoids the atomic-add).
///                          When non-null, the buffer must be PRE-ZEROED
///                          by the caller -- the kernel uses `atomicAdd`
///                          to accumulate per-channel contributions.
/// \param stream            CUDA stream.
//------------------------------------------------------------------------------
void launch_spherical_harmonics_backward(
  std::uint32_t N,
  std::uint32_t K,
  std::uint32_t degrees_to_use,
  const float* dirs,
  const float* coeffs,
  const bool* masks,
  const float* v_colors,
  float* v_coeffs,
  float* v_dirs,
  cudaStream_t stream);

} // namespace Heterosplat
} // namespace Kernels

#endif // KERNELS_HETEROSPLAT_SPHERICAL_HARMONICS_H
