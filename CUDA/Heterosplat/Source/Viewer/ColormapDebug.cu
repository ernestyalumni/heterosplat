#include "Viewer/ColormapDebug.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace Viewer
{

namespace
{

constexpr std::uint32_t kThreadsPerBlock {256};

__global__ void colormap_per_axis_kernel(
  const std::uint32_t N,
  const float* __restrict__ means,
  const float min_x, const float max_x,
  const float min_y, const float max_y,
  const float min_z, const float max_z,
  const std::uint32_t axis,
  const std::uint32_t K,
  float* __restrict__ sh_coeffs)
{
  const std::uint32_t n {blockIdx.x * blockDim.x + threadIdx.x};
  if (n >= N) return;

  float t;
  if (axis == 0)
    t = (means[n * 3 + 0] - min_x) / fmaxf(max_x - min_x, 1e-6f);
  else if (axis == 1)
    t = (means[n * 3 + 1] - min_y) / fmaxf(max_y - min_y, 1e-6f);
  else
    t = (means[n * 3 + 2] - min_z) / fmaxf(max_z - min_z, 1e-6f);

  t = fmaxf(0.0f, fminf(1.0f, t));

  // Turbo-ish colormap (simplified piecewise linear)
  float r, g, b;
  if (t < 0.25f)
  {
    const float s {t / 0.25f};
    r = 0.18f + 0.50f * s;
    g = 0.05f + 0.50f * s;
    b = 0.53f + 0.47f * s;
  }
  else if (t < 0.5f)
  {
    const float s {(t - 0.25f) / 0.25f};
    r = 0.68f + 0.25f * s;
    g = 0.55f + 0.35f * s;
    b = 1.0f - 0.65f * s;
  }
  else if (t < 0.75f)
  {
    const float s {(t - 0.5f) / 0.25f};
    r = 0.93f + 0.07f * s;
    g = 0.90f - 0.15f * s;
    b = 0.35f - 0.25f * s;
  }
  else
  {
    const float s {(t - 0.75f) / 0.25f};
    r = 1.0f - 0.40f * s;
    g = 0.75f - 0.60f * s;
    b = 0.10f - 0.05f * s;
  }

  // SH DC coefficient: color = C0 * sh_dc
  // To display color c, set sh_dc = c / C0
  constexpr float inv_C0 {1.0f / 0.28209479177387814f};

  // DC is the first SH coefficient per Gaussian: sh_coeffs[n * K * 3 + {0,1,2}]
  const std::uint32_t base {n * K * 3};
  sh_coeffs[base + 0] = r * inv_C0;
  sh_coeffs[base + 1] = g * inv_C0;
  sh_coeffs[base + 2] = b * inv_C0;

  // Zero out higher-order SH terms so colormap is view-independent
  for (std::uint32_t k = 1; k < K; ++k)
  {
    sh_coeffs[base + k * 3 + 0] = 0.0f;
    sh_coeffs[base + k * 3 + 1] = 0.0f;
    sh_coeffs[base + k * 3 + 2] = 0.0f;
  }
}

} // namespace

void launch_colormap_per_axis(
  const std::uint32_t N,
  const float* means,
  const float min_x, const float max_x,
  const float min_y, const float max_y,
  const float min_z, const float max_z,
  const std::uint32_t axis,
  const std::uint32_t K,
  float* sh_coeffs,
  cudaStream_t stream)
{
  if (N == 0) return;
  const dim3 grid {(N + kThreadsPerBlock - 1u) / kThreadsPerBlock};
  colormap_per_axis_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
    N, means, min_x, max_x, min_y, max_y, min_z, max_z, axis, K, sh_coeffs);
}

} // namespace Viewer
