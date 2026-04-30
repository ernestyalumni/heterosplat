#include "Core/CubOperations.h"
#include "Kernels/Heterosplat/IntersectOffset.h"
#include "Kernels/Heterosplat/IntersectTile.h"
#include "Kernels/Heterosplat/ProjectionEWA3DGSFused.h"
#include "Kernels/Heterosplat/QuatScaleToCovar.h"
#include "Kernels/Heterosplat/RasterizeToPixels3DGS.h"
#include "Kernels/Heterosplat/SphericalHarmonics.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

namespace
{

template <typename T>
struct GpuBuffer
{
  T* ptr {nullptr};
  std::size_t count {0};

  explicit GpuBuffer(const std::size_t n) : count{n}
  {
    if (count > 0)
    {
      if (cudaMalloc(&ptr, count * sizeof(T)) != cudaSuccess)
      {
        std::cerr << "cudaMalloc failed for " << count << " elements\n";
        std::exit(1);
      }
    }
  }

  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;
  ~GpuBuffer() { if (ptr) cudaFree(ptr); }

  void upload(const std::vector<T>& host)
  {
    cudaMemcpy(ptr, host.data(), count * sizeof(T), cudaMemcpyHostToDevice);
  }

  void zero()
  {
    cudaMemset(ptr, 0, count * sizeof(T));
  }

  std::vector<T> download() const
  {
    std::vector<T> host(count);
    cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
  }
};

bool check_cuda(const char* label)
{
  const cudaError_t err {cudaDeviceSynchronize()};
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA ERROR after " << label << ": "
              << cudaGetErrorString(err) << "\n";
    return false;
  }
  return true;
}

template <typename T>
bool all_finite(const std::vector<T>& v, const char* name)
{
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    if (!std::isfinite(static_cast<double>(v[i])))
    {
      std::cerr << "FAIL: " << name << "[" << i << "] is not finite ("
                << v[i] << ")\n";
      return false;
    }
  }
  return true;
}

template <typename T>
bool has_nonzero(const std::vector<T>& v, const char* name)
{
  for (const auto& x : v)
  {
    if (x != T{0}) return true;
  }
  std::cerr << "FAIL: " << name << " is all zeros\n";
  return false;
}

template <typename T>
bool check_finite_nonzero(const std::vector<T>& v, const char* name)
{
  return all_finite(v, name) && has_nonzero(v, name);
}

} // namespace

int main()
{
  constexpr std::uint32_t N {1024};
  constexpr std::uint32_t B {1};
  constexpr std::uint32_t C {1};
  constexpr std::uint32_t I {B * C};
  constexpr std::uint32_t image_width {64};
  constexpr std::uint32_t image_height {64};
  constexpr std::uint32_t tile_size {16};
  constexpr std::uint32_t tile_width {
    (image_width + tile_size - 1) / tile_size};
  constexpr std::uint32_t tile_height {
    (image_height + tile_size - 1) / tile_size};
  constexpr std::uint32_t n_pixels {I * image_height * image_width};
  constexpr std::uint32_t K {1};
  constexpr std::uint32_t degrees_to_use {0};

  std::cout << "Forward-backward smoke test: " << N << " Gaussians, "
            << image_width << "x" << image_height << " image, "
            << tile_width << "x" << tile_height << " tiles\n\n";

  // ====================================================================
  // Synthetic data generation
  // ====================================================================

  std::mt19937 rng{42};
  std::uniform_real_distribution<float> dist_pos{-2.0f, 2.0f};
  std::uniform_real_distribution<float> dist_scale{0.01f, 0.1f};
  std::uniform_real_distribution<float> dist_opacity{0.5f, 1.0f};
  std::uniform_real_distribution<float> dist_sh{-0.5f, 0.5f};
  std::uniform_real_distribution<float> dist_unit{-1.0f, 1.0f};

  std::vector<float> h_means(N * 3);
  for (std::uint32_t n = 0; n < N; ++n)
  {
    h_means[n * 3 + 0] = dist_pos(rng);
    h_means[n * 3 + 1] = dist_pos(rng);
    h_means[n * 3 + 2] = 5.0f + dist_unit(rng);
  }

  std::vector<float> h_quats(N * 4);
  for (std::uint32_t n = 0; n < N; ++n)
  {
    float w {dist_unit(rng)};
    float x {dist_unit(rng)};
    float y {dist_unit(rng)};
    float z {dist_unit(rng)};
    const float inv_norm {1.0f / std::sqrt(w*w + x*x + y*y + z*z)};
    h_quats[n * 4 + 0] = w * inv_norm;
    h_quats[n * 4 + 1] = x * inv_norm;
    h_quats[n * 4 + 2] = y * inv_norm;
    h_quats[n * 4 + 3] = z * inv_norm;
  }

  std::vector<float> h_scales(N * 3);
  for (auto& s : h_scales) s = dist_scale(rng);

  std::vector<float> h_opacities(N);
  for (auto& o : h_opacities) o = dist_opacity(rng);

  std::vector<float> h_sh_coeffs(N * K * 3);
  for (auto& c : h_sh_coeffs) c = dist_sh(rng);

  std::vector<float> h_dirs(N * 3);
  for (std::uint32_t n = 0; n < N; ++n)
  {
    const float dx {h_means[n * 3 + 0]};
    const float dy {h_means[n * 3 + 1]};
    const float dz {h_means[n * 3 + 2]};
    const float inv_norm {1.0f / std::sqrt(dx*dx + dy*dy + dz*dz)};
    h_dirs[n * 3 + 0] = dx * inv_norm;
    h_dirs[n * 3 + 1] = dy * inv_norm;
    h_dirs[n * 3 + 2] = dz * inv_norm;
  }

  // Identity viewmat, pinhole camera with fx=fy=50, cx=cy=32
  const std::vector<float> h_viewmats {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1};

  const std::vector<float> h_Ks {
    50, 0, 32,
    0, 50, 32,
    0, 0, 1};

  // ====================================================================
  // Upload to device
  // ====================================================================

  GpuBuffer<float> d_means(N * 3);
  GpuBuffer<float> d_quats(N * 4);
  GpuBuffer<float> d_scales(N * 3);
  GpuBuffer<float> d_opacities(N);
  GpuBuffer<float> d_sh_coeffs(N * K * 3);
  GpuBuffer<float> d_dirs(N * 3);
  GpuBuffer<float> d_viewmats(B * C * 16);
  GpuBuffer<float> d_Ks(B * C * 9);

  d_means.upload(h_means);
  d_quats.upload(h_quats);
  d_scales.upload(h_scales);
  d_opacities.upload(h_opacities);
  d_sh_coeffs.upload(h_sh_coeffs);
  d_dirs.upload(h_dirs);
  d_viewmats.upload(h_viewmats);
  d_Ks.upload(h_Ks);

  // ====================================================================
  // FORWARD PIPELINE
  // ====================================================================

  // --- 1. Projection ---
  std::cout << "  [1/9] projection_ewa_3dgs_fused forward... " << std::flush;

  GpuBuffer<std::int32_t> d_radii(B * C * N * 2);
  GpuBuffer<float> d_means2d(B * C * N * 2);
  GpuBuffer<float> d_depths(B * C * N);
  GpuBuffer<float> d_conics(B * C * N * 3);

  Kernels::Heterosplat::launch_projection_ewa_3dgs_fused_forward(
    B, C, N,
    d_means.ptr, nullptr, d_quats.ptr, d_scales.ptr, nullptr,
    d_viewmats.ptr, d_Ks.ptr,
    image_width, image_height,
    0.3f, 0.01f, 1e10f, 0.0f, 0,
    d_radii.ptr, d_means2d.ptr, d_depths.ptr, d_conics.ptr,
    nullptr, nullptr);
  if (!check_cuda("projection fwd")) return 1;
  std::cout << "ok\n";

  // --- 2. Spherical harmonics ---
  std::cout << "  [2/9] spherical_harmonics forward... " << std::flush;

  GpuBuffer<float> d_colors(N * 3);

  Kernels::Heterosplat::launch_spherical_harmonics_forward(
    N, K, degrees_to_use,
    d_dirs.ptr, d_sh_coeffs.ptr, nullptr,
    d_colors.ptr, nullptr);
  if (!check_cuda("SH fwd")) return 1;
  std::cout << "ok\n";

  // --- 3. Intersect tile (pass 1: count tiles per Gaussian) ---
  std::cout << "  [3/9] intersect_tile pass 1 (count)... " << std::flush;

  GpuBuffer<std::int32_t> d_tiles_per_gauss(I * N);

  Kernels::Heterosplat::launch_intersect_tile_forward(
    false, I, N, 0,
    nullptr, nullptr,
    d_means2d.ptr, d_radii.ptr, d_depths.ptr,
    nullptr, nullptr, nullptr,
    tile_size, tile_width, tile_height,
    d_tiles_per_gauss.ptr, nullptr, nullptr, nullptr);
  if (!check_cuda("intersect tile pass 1")) return 1;
  std::cout << "ok\n";

  // --- 4. CUB inclusive prefix sum ---
  std::cout << "  [4/9] CUB inclusive sum... " << std::flush;

  GpuBuffer<std::int64_t> d_cum_tiles(I * N);

  Core::cub_inclusive_sum_int32_to_int64(
    I * N, d_tiles_per_gauss.ptr, d_cum_tiles.ptr, nullptr);
  if (!check_cuda("CUB inclusive sum")) return 1;

  std::int64_t n_isects_64 {0};
  cudaMemcpy(
    &n_isects_64,
    d_cum_tiles.ptr + (I * N - 1),
    sizeof(std::int64_t),
    cudaMemcpyDeviceToHost);
  const auto n_isects {static_cast<std::uint32_t>(n_isects_64)};
  std::cout << "ok (" << n_isects << " intersections)\n";

  if (n_isects == 0)
  {
    std::cerr << "FAIL: zero intersections (all Gaussians culled)\n";
    return 1;
  }

  // --- 5. Intersect tile (pass 2: write isect_ids + flatten_ids) ---
  std::cout << "  [5/9] intersect_tile pass 2 (write)... " << std::flush;

  GpuBuffer<std::int64_t> d_isect_ids(n_isects);
  GpuBuffer<std::int32_t> d_flatten_ids(n_isects);

  Kernels::Heterosplat::launch_intersect_tile_forward(
    false, I, N, 0,
    nullptr, nullptr,
    d_means2d.ptr, d_radii.ptr, d_depths.ptr,
    nullptr, nullptr, d_cum_tiles.ptr,
    tile_size, tile_width, tile_height,
    nullptr, d_isect_ids.ptr, d_flatten_ids.ptr, nullptr);
  if (!check_cuda("intersect tile pass 2")) return 1;
  std::cout << "ok\n";

  // --- 6. CUB radix sort ---
  std::cout << "  [6/9] CUB radix sort... " << std::flush;

  GpuBuffer<std::int64_t> d_isect_ids_sorted(n_isects);
  GpuBuffer<std::int32_t> d_flatten_ids_sorted(n_isects);

  Core::cub_radix_sort_pairs_int64_int32(
    n_isects,
    d_isect_ids.ptr, d_isect_ids_sorted.ptr,
    d_flatten_ids.ptr, d_flatten_ids_sorted.ptr,
    0, 64, nullptr);
  if (!check_cuda("CUB radix sort")) return 1;
  std::cout << "ok\n";

  // --- 7. Intersect offset ---
  std::cout << "  [7/9] intersect_offset forward... " << std::flush;

  GpuBuffer<std::int32_t> d_tile_offsets(I * tile_height * tile_width);

  Kernels::Heterosplat::launch_intersect_offset_forward(
    n_isects, d_isect_ids_sorted.ptr,
    I, tile_width, tile_height,
    d_tile_offsets.ptr, nullptr);
  if (!check_cuda("intersect offset")) return 1;
  std::cout << "ok\n";

  // --- 8. Rasterize forward ---
  std::cout << "  [8/9] rasterize_to_pixels_3dgs forward... " << std::flush;

  GpuBuffer<float> d_render_colors(n_pixels * 3);
  GpuBuffer<float> d_render_alphas(n_pixels);
  GpuBuffer<std::int32_t> d_last_ids(n_pixels);

  Kernels::Heterosplat::launch_rasterize_to_pixels_3dgs_forward(
    I, N, n_isects, false,
    d_means2d.ptr, d_conics.ptr, d_colors.ptr, d_opacities.ptr,
    nullptr, nullptr,
    image_width, image_height, tile_size,
    d_tile_offsets.ptr, d_flatten_ids_sorted.ptr,
    d_render_colors.ptr, d_render_alphas.ptr, d_last_ids.ptr,
    nullptr);
  if (!check_cuda("rasterize fwd")) return 1;

  {
    const auto h_rc {d_render_colors.download()};
    const auto h_ra {d_render_alphas.download()};
    if (!check_finite_nonzero(h_rc, "render_colors")) return 1;
    if (!check_finite_nonzero(h_ra, "render_alphas")) return 1;
  }
  std::cout << "ok\n";

  // ====================================================================
  // BACKWARD PIPELINE
  // ====================================================================

  // --- 9a. Rasterize backward ---
  std::cout << "  [9/9] backward (rasterize + SH + projection)... "
            << std::flush;

  GpuBuffer<float> d_v_render_colors(n_pixels * 3);
  GpuBuffer<float> d_v_render_alphas(n_pixels);
  {
    std::vector<float> ones_c(n_pixels * 3, 1.0f);
    std::vector<float> ones_a(n_pixels, 1.0f);
    d_v_render_colors.upload(ones_c);
    d_v_render_alphas.upload(ones_a);
  }

  GpuBuffer<float> d_v_means2d(I * N * 2);
  GpuBuffer<float> d_v_conics(I * N * 3);
  GpuBuffer<float> d_v_colors(I * N * 3);
  GpuBuffer<float> d_v_opacities(I * N);
  d_v_means2d.zero();
  d_v_conics.zero();
  d_v_colors.zero();
  d_v_opacities.zero();

  Kernels::Heterosplat::launch_rasterize_to_pixels_3dgs_backward(
    I, N, n_isects, false,
    d_means2d.ptr, d_conics.ptr, d_colors.ptr, d_opacities.ptr,
    nullptr, nullptr,
    image_width, image_height, tile_size,
    d_tile_offsets.ptr, d_flatten_ids_sorted.ptr,
    d_render_alphas.ptr, d_last_ids.ptr,
    d_v_render_colors.ptr, d_v_render_alphas.ptr,
    nullptr, d_v_means2d.ptr, d_v_conics.ptr,
    d_v_colors.ptr, d_v_opacities.ptr,
    nullptr);
  if (!check_cuda("rasterize bwd")) return 1;

  // --- 9b. SH backward ---
  GpuBuffer<float> d_v_sh_coeffs(N * K * 3);
  d_v_sh_coeffs.zero();

  Kernels::Heterosplat::launch_spherical_harmonics_backward(
    N, K, degrees_to_use,
    d_dirs.ptr, d_sh_coeffs.ptr, nullptr,
    d_v_colors.ptr,
    d_v_sh_coeffs.ptr, nullptr, nullptr);
  if (!check_cuda("SH bwd")) return 1;

  // --- 9c. Projection backward ---
  GpuBuffer<float> d_v_means(B * N * 3);
  GpuBuffer<float> d_v_quats(B * N * 4);
  GpuBuffer<float> d_v_scales(B * N * 3);
  GpuBuffer<float> d_v_depths(B * C * N);
  d_v_means.zero();
  d_v_quats.zero();
  d_v_scales.zero();
  d_v_depths.zero();

  Kernels::Heterosplat::launch_projection_ewa_3dgs_fused_backward(
    B, C, N,
    d_means.ptr, nullptr, d_quats.ptr, d_scales.ptr,
    d_viewmats.ptr, d_Ks.ptr,
    image_width, image_height, 0.3f, 0,
    d_radii.ptr, d_conics.ptr, nullptr,
    d_v_means2d.ptr, d_v_depths.ptr, d_v_conics.ptr, nullptr,
    d_v_means.ptr, nullptr, d_v_quats.ptr, d_v_scales.ptr,
    nullptr, nullptr);
  if (!check_cuda("projection bwd")) return 1;

  {
    const auto h_vm {d_v_means.download()};
    const auto h_vq {d_v_quats.download()};
    const auto h_vs {d_v_scales.download()};
    const auto h_vc {d_v_sh_coeffs.download()};
    if (!check_finite_nonzero(h_vm, "v_means")) return 1;
    if (!check_finite_nonzero(h_vq, "v_quats")) return 1;
    if (!check_finite_nonzero(h_vs, "v_scales")) return 1;
    if (!check_finite_nonzero(h_vc, "v_sh_coeffs")) return 1;
  }
  std::cout << "ok\n";

  // ====================================================================
  // Bonus: standalone quat_scale_to_covar exercise
  // ====================================================================

  std::cout << "  [+]   quat_scale_to_covar fwd+bwd... " << std::flush;

  GpuBuffer<float> d_covars(N * 6);

  Kernels::Heterosplat::launch_quat_scale_to_covar_preci_forward(
    N, d_quats.ptr, d_scales.ptr, true,
    d_covars.ptr, nullptr, nullptr);
  if (!check_cuda("quat_scale_to_covar fwd")) return 1;

  GpuBuffer<float> d_v_covars(N * 6);
  {
    std::vector<float> ones(N * 6, 1.0f);
    d_v_covars.upload(ones);
  }

  GpuBuffer<float> d_v_quats_qsc(N * 4);
  GpuBuffer<float> d_v_scales_qsc(N * 3);

  Kernels::Heterosplat::launch_quat_scale_to_covar_preci_backward(
    N, d_quats.ptr, d_scales.ptr, true,
    d_v_covars.ptr, nullptr,
    d_v_quats_qsc.ptr, d_v_scales_qsc.ptr, nullptr);
  if (!check_cuda("quat_scale_to_covar bwd")) return 1;

  {
    const auto h_cov {d_covars.download()};
    const auto h_vq {d_v_quats_qsc.download()};
    if (!check_finite_nonzero(h_cov, "covars")) return 1;
    if (!check_finite_nonzero(h_vq, "v_quats (qsc)")) return 1;
  }
  std::cout << "ok\n";

  // ====================================================================
  // Summary
  // ====================================================================

  std::cout << "\nAll checks passed: " << N << " Gaussians, "
            << n_isects << " tile intersections, "
            << image_width << "x" << image_height << " render, "
            << "fwd+bwd through all 6 kernels.\n";

  return 0;
}
