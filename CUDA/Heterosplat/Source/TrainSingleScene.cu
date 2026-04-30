#include "Colmap/ColmapReader.h"
#include "Core/CubOperations.h"
#include "IO/ImageIO.h"
#include "IO/PlyWriter.h"
#include "Kernels/Heterosplat/IntersectOffset.h"
#include "Kernels/Heterosplat/IntersectTile.h"
#include "Kernels/Heterosplat/ProjectionEWA3DGSFused.h"
#include "Kernels/Heterosplat/RasterizeToPixels3DGS.h"
#include "Kernels/Heterosplat/SphericalHarmonics.h"
#include "Training/Activations.h"
#include "Training/AdamOptimizer.h"
#include "Training/ImageLoss.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

// ============================================================================
// GPU buffer RAII wrapper (same as ForwardBackwardSmokeTest)
// ============================================================================

template <typename T>
struct GpuBuffer
{
  T* ptr {nullptr};
  std::size_t count {0};

  GpuBuffer() = default;

  explicit GpuBuffer(const std::size_t n) : count{n}
  {
    if (count > 0)
    {
      if (cudaMalloc(&ptr, count * sizeof(T)) != cudaSuccess)
      {
        std::cerr << "cudaMalloc failed for " << count << " elements ("
                  << count * sizeof(T) << " bytes)\n";
        std::exit(1);
      }
    }
  }

  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;

  GpuBuffer(GpuBuffer&& o) noexcept : ptr{o.ptr}, count{o.count}
  {
    o.ptr = nullptr;
    o.count = 0;
  }

  GpuBuffer& operator=(GpuBuffer&& o) noexcept
  {
    if (this != &o)
    {
      if (ptr) cudaFree(ptr);
      ptr = o.ptr;
      count = o.count;
      o.ptr = nullptr;
      o.count = 0;
    }
    return *this;
  }

  ~GpuBuffer() { if (ptr) cudaFree(ptr); }

  void resize(const std::size_t n)
  {
    if (n <= count) { count = n; return; }
    if (ptr) cudaFree(ptr);
    count = n;
    if (cudaMalloc(&ptr, count * sizeof(T)) != cudaSuccess)
    {
      std::cerr << "cudaMalloc failed for " << count << " elements\n";
      std::exit(1);
    }
  }

  void upload(const T* host, std::size_t n)
  {
    assert(n <= count);
    cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice);
  }

  void upload(const std::vector<T>& host)
  {
    upload(host.data(), host.size());
  }

  void zero()
  {
    if (count > 0)
      cudaMemset(ptr, 0, count * sizeof(T));
  }

  std::vector<T> download() const
  {
    std::vector<T> host(count);
    cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
  }
};

// ============================================================================
// Training configuration
// ============================================================================

struct TrainConfig
{
  std::uint32_t num_iterations {30000};
  std::uint32_t sh_degree_max {3};

  float lr_means {1.6e-4f};
  float lr_sh {2.5e-3f};
  float lr_opacities {5e-2f};
  float lr_scales {5e-3f};
  float lr_quats {1e-3f};

  float adam_beta1 {0.9f};
  float adam_beta2 {0.999f};
  float adam_epsilon {1e-15f};

  std::uint32_t sh_degree_interval {1000};
  std::uint32_t print_interval {100};
  std::uint32_t save_interval {7000};

  std::uint32_t tile_size {16};
  float eps2d {0.3f};
  float near_plane {0.01f};
  float far_plane {1e10f};
};

// ============================================================================
// Gaussian model — all trainable parameters + Adam state on GPU
// ============================================================================

struct GaussianModel
{
  std::uint32_t N {0};
  std::uint32_t sh_degree_max {3};
  std::uint32_t K {0};

  GpuBuffer<float> means;
  GpuBuffer<float> quats;
  GpuBuffer<float> log_scales;
  GpuBuffer<float> logit_opacities;
  GpuBuffer<float> sh_coeffs;

  GpuBuffer<float> m1_means, m2_means;
  GpuBuffer<float> m1_quats, m2_quats;
  GpuBuffer<float> m1_log_scales, m2_log_scales;
  GpuBuffer<float> m1_logit_opacities, m2_logit_opacities;
  GpuBuffer<float> m1_sh_coeffs, m2_sh_coeffs;

  GpuBuffer<float> actual_scales;
  GpuBuffer<float> actual_opacities;

  void allocate(const std::uint32_t num_gaussians, const std::uint32_t max_sh_deg)
  {
    N = num_gaussians;
    sh_degree_max = max_sh_deg;
    K = (sh_degree_max + 1) * (sh_degree_max + 1);

    means = GpuBuffer<float>(N * 3);
    quats = GpuBuffer<float>(N * 4);
    log_scales = GpuBuffer<float>(N * 3);
    logit_opacities = GpuBuffer<float>(N);
    sh_coeffs = GpuBuffer<float>(N * K * 3);

    m1_means = GpuBuffer<float>(N * 3); m1_means.zero();
    m2_means = GpuBuffer<float>(N * 3); m2_means.zero();
    m1_quats = GpuBuffer<float>(N * 4); m1_quats.zero();
    m2_quats = GpuBuffer<float>(N * 4); m2_quats.zero();
    m1_log_scales = GpuBuffer<float>(N * 3); m1_log_scales.zero();
    m2_log_scales = GpuBuffer<float>(N * 3); m2_log_scales.zero();
    m1_logit_opacities = GpuBuffer<float>(N); m1_logit_opacities.zero();
    m2_logit_opacities = GpuBuffer<float>(N); m2_logit_opacities.zero();
    m1_sh_coeffs = GpuBuffer<float>(N * K * 3); m1_sh_coeffs.zero();
    m2_sh_coeffs = GpuBuffer<float>(N * K * 3); m2_sh_coeffs.zero();

    actual_scales = GpuBuffer<float>(N * 3);
    actual_opacities = GpuBuffer<float>(N);
  }
};

// ============================================================================
// Initialize Gaussians from COLMAP sparse points
// ============================================================================

void initialize_from_colmap(
  GaussianModel& model,
  const std::vector<Colmap::Point3D>& points,
  const std::uint32_t max_sh_degree)
{
  const std::uint32_t N {static_cast<std::uint32_t>(points.size())};
  model.allocate(N, max_sh_degree);

  const std::uint32_t K {model.K};

  std::vector<float> h_means(N * 3);
  std::vector<float> h_quats(N * 4);
  std::vector<float> h_log_scales(N * 3);
  std::vector<float> h_logit_opacities(N);
  std::vector<float> h_sh_coeffs(N * K * 3, 0.0f);

  constexpr float C0 {0.28209479177387814f};
  constexpr float initial_logit_opacity {-2.1972246f}; // logit(0.1)

  float min_x {1e30f}, min_y {1e30f}, min_z {1e30f};
  float max_x {-1e30f}, max_y {-1e30f}, max_z {-1e30f};

  for (std::uint32_t n = 0; n < N; ++n)
  {
    const auto& p {points[n]};
    h_means[n * 3 + 0] = static_cast<float>(p.position[0]);
    h_means[n * 3 + 1] = static_cast<float>(p.position[1]);
    h_means[n * 3 + 2] = static_cast<float>(p.position[2]);

    min_x = std::min(min_x, h_means[n * 3 + 0]);
    min_y = std::min(min_y, h_means[n * 3 + 1]);
    min_z = std::min(min_z, h_means[n * 3 + 2]);
    max_x = std::max(max_x, h_means[n * 3 + 0]);
    max_y = std::max(max_y, h_means[n * 3 + 1]);
    max_z = std::max(max_z, h_means[n * 3 + 2]);

    h_quats[n * 4 + 0] = 1.0f;
    h_quats[n * 4 + 1] = 0.0f;
    h_quats[n * 4 + 2] = 0.0f;
    h_quats[n * 4 + 3] = 0.0f;

    // SH DC from COLMAP point color
    h_sh_coeffs[n * K * 3 + 0] = (p.color[0] / 255.0f - 0.5f) / C0;
    h_sh_coeffs[n * K * 3 + 1] = (p.color[1] / 255.0f - 0.5f) / C0;
    h_sh_coeffs[n * K * 3 + 2] = (p.color[2] / 255.0f - 0.5f) / C0;

    h_logit_opacities[n] = initial_logit_opacity;
  }

  // Scale initialization: average spacing from bounding box.
  // Use max extent / cbrt(N) to handle degenerate (coplanar) point clouds.
  const float extent_x {max_x - min_x};
  const float extent_y {max_y - min_y};
  const float extent_z {max_z - min_z};
  const float max_extent {std::max({extent_x, extent_y, extent_z, 1e-6f})};
  const float avg_spacing {max_extent / std::cbrt(static_cast<float>(N))};
  const float initial_log_scale {std::log(std::max(avg_spacing * 0.5f, 1e-6f))};

  std::cout << "  Scene extents: [" << extent_x << ", " << extent_y << ", "
            << extent_z << "], avg spacing: " << avg_spacing
            << ", initial scale: " << std::exp(initial_log_scale) << "\n";

  for (std::uint32_t n = 0; n < N; ++n)
  {
    h_log_scales[n * 3 + 0] = initial_log_scale;
    h_log_scales[n * 3 + 1] = initial_log_scale;
    h_log_scales[n * 3 + 2] = initial_log_scale;
  }

  model.means.upload(h_means);
  model.quats.upload(h_quats);
  model.log_scales.upload(h_log_scales);
  model.logit_opacities.upload(h_logit_opacities);
  model.sh_coeffs.upload(h_sh_coeffs);
}

// ============================================================================
// Camera center from 4x4 viewmat (row-major)
// cam_center = -R^T * t
// ============================================================================

void camera_center_from_viewmat(
  const float* viewmat, float& cx, float& cy, float& cz)
{
  const float r00 {viewmat[0]}, r01 {viewmat[1]}, r02 {viewmat[2]};
  const float r10 {viewmat[4]}, r11 {viewmat[5]}, r12 {viewmat[6]};
  const float r20 {viewmat[8]}, r21 {viewmat[9]}, r22 {viewmat[10]};
  const float tx {viewmat[3]}, ty {viewmat[7]}, tz {viewmat[11]};
  cx = -(r00 * tx + r10 * ty + r20 * tz);
  cy = -(r01 * tx + r11 * ty + r21 * tz);
  cz = -(r02 * tx + r12 * ty + r22 * tz);
}

// ============================================================================
// Single training step
// ============================================================================

struct TrainBuffers
{
  GpuBuffer<std::int32_t> radii;
  GpuBuffer<float> means2d;
  GpuBuffer<float> depths;
  GpuBuffer<float> conics;
  GpuBuffer<float> colors;
  GpuBuffer<float> dirs;

  GpuBuffer<std::int32_t> tiles_per_gauss;
  GpuBuffer<std::int64_t> cum_tiles;

  GpuBuffer<std::int64_t> isect_ids;
  GpuBuffer<std::int32_t> flatten_ids;
  GpuBuffer<std::int64_t> isect_ids_sorted;
  GpuBuffer<std::int32_t> flatten_ids_sorted;
  GpuBuffer<std::int32_t> tile_offsets;

  GpuBuffer<float> render_colors;
  GpuBuffer<float> render_alphas;
  GpuBuffer<std::int32_t> last_ids;
  GpuBuffer<float> gt_image;

  GpuBuffer<float> loss;
  GpuBuffer<float> grad_rendered;

  // Backward buffers
  GpuBuffer<float> v_render_alphas;
  GpuBuffer<float> v_means2d;
  GpuBuffer<float> v_conics;
  GpuBuffer<float> v_colors;
  GpuBuffer<float> v_opacities;
  GpuBuffer<float> v_means;
  GpuBuffer<float> v_quats;
  GpuBuffer<float> v_scales;
  GpuBuffer<float> v_depths;
  GpuBuffer<float> v_sh_coeffs;

  std::size_t isect_capacity {0};

  void allocate(
    const std::uint32_t N,
    const std::uint32_t K,
    const std::uint32_t max_image_width,
    const std::uint32_t max_image_height,
    const std::uint32_t tile_size)
  {
    const std::uint32_t n_pixels {max_image_width * max_image_height};
    const std::uint32_t tile_w {
      (max_image_width + tile_size - 1) / tile_size};
    const std::uint32_t tile_h {
      (max_image_height + tile_size - 1) / tile_size};

    radii = GpuBuffer<std::int32_t>(N * 2);
    means2d = GpuBuffer<float>(N * 2);
    depths = GpuBuffer<float>(N);
    conics = GpuBuffer<float>(N * 3);
    colors = GpuBuffer<float>(N * 3);
    dirs = GpuBuffer<float>(N * 3);

    tiles_per_gauss = GpuBuffer<std::int32_t>(N);
    cum_tiles = GpuBuffer<std::int64_t>(N);

    tile_offsets = GpuBuffer<std::int32_t>(tile_h * tile_w);

    render_colors = GpuBuffer<float>(n_pixels * 3);
    render_alphas = GpuBuffer<float>(n_pixels);
    last_ids = GpuBuffer<std::int32_t>(n_pixels);
    gt_image = GpuBuffer<float>(n_pixels * 3);

    loss = GpuBuffer<float>(1);
    grad_rendered = GpuBuffer<float>(n_pixels * 3);

    v_render_alphas = GpuBuffer<float>(n_pixels);
    v_means2d = GpuBuffer<float>(N * 2);
    v_conics = GpuBuffer<float>(N * 3);
    v_colors = GpuBuffer<float>(N * 3);
    v_opacities = GpuBuffer<float>(N);
    v_means = GpuBuffer<float>(N * 3);
    v_quats = GpuBuffer<float>(N * 4);
    v_scales = GpuBuffer<float>(N * 3);
    v_depths = GpuBuffer<float>(N);
    v_sh_coeffs = GpuBuffer<float>(N * K * 3);

    isect_capacity = 0;
  }

  void ensure_isect_capacity(const std::size_t needed)
  {
    if (needed <= isect_capacity) return;
    const std::size_t new_cap {std::max(needed, isect_capacity * 2)};
    isect_ids = GpuBuffer<std::int64_t>(new_cap);
    flatten_ids = GpuBuffer<std::int32_t>(new_cap);
    isect_ids_sorted = GpuBuffer<std::int64_t>(new_cap);
    flatten_ids_sorted = GpuBuffer<std::int32_t>(new_cap);
    isect_capacity = new_cap;
  }
};

float train_step(
  GaussianModel& model,
  TrainBuffers& bufs,
  const float* h_viewmat,
  const float* h_K,
  const IO::Image& gt,
  const std::uint32_t degrees_to_use,
  const TrainConfig& cfg,
  const std::uint32_t adam_step,
  cudaStream_t stream)
{
  const std::uint32_t N {model.N};
  const std::uint32_t K_sh {model.K};
  const std::uint32_t I {1};
  const std::uint32_t B {1};
  const std::uint32_t C {1};
  const std::uint32_t image_w {gt.width};
  const std::uint32_t image_h {gt.height};
  const std::uint32_t tile_size {cfg.tile_size};
  const std::uint32_t tile_w {(image_w + tile_size - 1) / tile_size};
  const std::uint32_t tile_h {(image_h + tile_size - 1) / tile_size};
  const std::uint32_t n_pixels {image_w * image_h};

  // Upload viewmat + K
  GpuBuffer<float> d_viewmat(16);
  GpuBuffer<float> d_K(9);
  cudaMemcpyAsync(d_viewmat.ptr, h_viewmat, 16 * sizeof(float),
    cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_K.ptr, h_K, 9 * sizeof(float),
    cudaMemcpyHostToDevice, stream);

  // Upload GT image
  cudaMemcpyAsync(bufs.gt_image.ptr, gt.pixels.data(),
    n_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

  // Compute camera center
  float cam_x, cam_y, cam_z;
  camera_center_from_viewmat(h_viewmat, cam_x, cam_y, cam_z);

  // ---- Activations ----
  Training::launch_exp_forward(
    N * 3, model.log_scales.ptr, model.actual_scales.ptr, stream);
  Training::launch_sigmoid_forward(
    N, model.logit_opacities.ptr, model.actual_opacities.ptr, stream);

  // ---- View directions ----
  Training::launch_compute_view_directions(
    N, model.means.ptr, cam_x, cam_y, cam_z, bufs.dirs.ptr, stream);

  // ---- Forward: Projection ----
  Kernels::Heterosplat::launch_projection_ewa_3dgs_fused_forward(
    B, C, N,
    model.means.ptr, nullptr, model.quats.ptr, model.actual_scales.ptr,
    nullptr,
    d_viewmat.ptr, d_K.ptr,
    image_w, image_h,
    cfg.eps2d, cfg.near_plane, cfg.far_plane, 0.0f, 0,
    bufs.radii.ptr, bufs.means2d.ptr, bufs.depths.ptr, bufs.conics.ptr,
    nullptr, stream);

  // ---- Forward: SH ----
  Kernels::Heterosplat::launch_spherical_harmonics_forward(
    N, K_sh, degrees_to_use,
    bufs.dirs.ptr, model.sh_coeffs.ptr, nullptr,
    bufs.colors.ptr, stream);

  // ---- Forward: Intersect tile pass 1 (count) ----
  Kernels::Heterosplat::launch_intersect_tile_forward(
    false, I, N, 0,
    nullptr, nullptr,
    bufs.means2d.ptr, bufs.radii.ptr, bufs.depths.ptr,
    nullptr, nullptr, nullptr,
    tile_size, tile_w, tile_h,
    bufs.tiles_per_gauss.ptr, nullptr, nullptr, stream);

  // ---- CUB inclusive prefix sum ----
  Core::cub_inclusive_sum_int32_to_int64(
    I * N, bufs.tiles_per_gauss.ptr, bufs.cum_tiles.ptr, stream);

  // Read n_isects from device
  std::int64_t n_isects_64 {0};
  cudaMemcpyAsync(&n_isects_64, bufs.cum_tiles.ptr + (I * N - 1),
    sizeof(std::int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  const auto n_isects {static_cast<std::uint32_t>(n_isects_64)};

  if (n_isects == 0)
  {
    return 0.0f;
  }

  bufs.ensure_isect_capacity(n_isects);

  // ---- Forward: Intersect tile pass 2 (write) ----
  Kernels::Heterosplat::launch_intersect_tile_forward(
    false, I, N, 0,
    nullptr, nullptr,
    bufs.means2d.ptr, bufs.radii.ptr, bufs.depths.ptr,
    nullptr, nullptr, bufs.cum_tiles.ptr,
    tile_size, tile_w, tile_h,
    nullptr, bufs.isect_ids.ptr, bufs.flatten_ids.ptr, stream);

  // ---- CUB radix sort ----
  Core::cub_radix_sort_pairs_int64_int32(
    n_isects,
    bufs.isect_ids.ptr, bufs.isect_ids_sorted.ptr,
    bufs.flatten_ids.ptr, bufs.flatten_ids_sorted.ptr,
    0, 64, stream);

  // ---- Forward: Intersect offset ----
  bufs.tile_offsets.zero();
  Kernels::Heterosplat::launch_intersect_offset_forward(
    n_isects, bufs.isect_ids_sorted.ptr,
    I, tile_w, tile_h,
    bufs.tile_offsets.ptr, stream);

  // ---- Forward: Rasterize ----
  Kernels::Heterosplat::launch_rasterize_to_pixels_3dgs_forward(
    I, N, n_isects, false,
    bufs.means2d.ptr, bufs.conics.ptr, bufs.colors.ptr,
    model.actual_opacities.ptr,
    nullptr, nullptr,
    image_w, image_h, tile_size,
    bufs.tile_offsets.ptr, bufs.flatten_ids_sorted.ptr,
    bufs.render_colors.ptr, bufs.render_alphas.ptr, bufs.last_ids.ptr,
    stream);

  // ---- L1 Loss ----
  Training::launch_l1_loss(
    n_pixels,
    bufs.render_colors.ptr, bufs.gt_image.ptr,
    bufs.loss.ptr, bufs.grad_rendered.ptr,
    stream);

  // Read loss to host
  float h_loss {0.0f};
  cudaMemcpyAsync(&h_loss, bufs.loss.ptr, sizeof(float),
    cudaMemcpyDeviceToHost, stream);

  // ---- Backward: Rasterize ----
  bufs.v_render_alphas.zero();
  bufs.v_means2d.zero();
  bufs.v_conics.zero();
  bufs.v_colors.zero();
  bufs.v_opacities.zero();

  Kernels::Heterosplat::launch_rasterize_to_pixels_3dgs_backward(
    I, N, n_isects, false,
    bufs.means2d.ptr, bufs.conics.ptr, bufs.colors.ptr,
    model.actual_opacities.ptr,
    nullptr, nullptr,
    image_w, image_h, tile_size,
    bufs.tile_offsets.ptr, bufs.flatten_ids_sorted.ptr,
    bufs.render_alphas.ptr, bufs.last_ids.ptr,
    bufs.grad_rendered.ptr, bufs.v_render_alphas.ptr,
    nullptr, bufs.v_means2d.ptr, bufs.v_conics.ptr,
    bufs.v_colors.ptr, bufs.v_opacities.ptr,
    stream);

  // ---- Backward: SH ----
  bufs.v_sh_coeffs.zero();

  Kernels::Heterosplat::launch_spherical_harmonics_backward(
    N, K_sh, degrees_to_use,
    bufs.dirs.ptr, model.sh_coeffs.ptr, nullptr,
    bufs.v_colors.ptr,
    bufs.v_sh_coeffs.ptr, nullptr, stream);

  // ---- Backward: Projection ----
  bufs.v_means.zero();
  bufs.v_quats.zero();
  bufs.v_scales.zero();
  bufs.v_depths.zero();

  Kernels::Heterosplat::launch_projection_ewa_3dgs_fused_backward(
    B, C, N,
    model.means.ptr, nullptr, model.quats.ptr, model.actual_scales.ptr,
    d_viewmat.ptr, d_K.ptr,
    image_w, image_h, cfg.eps2d, 0,
    bufs.radii.ptr, bufs.conics.ptr, nullptr,
    bufs.v_means2d.ptr, bufs.v_depths.ptr, bufs.v_conics.ptr, nullptr,
    bufs.v_means.ptr, nullptr, bufs.v_quats.ptr, bufs.v_scales.ptr,
    nullptr, stream);

  // ---- Chain activation gradients ----
  // v_scales is dL/d(actual_scale), need dL/d(log_scale)
  // Reuse v_scales buffer for the chained gradient
  Training::launch_exp_backward_chain(
    N * 3, model.actual_scales.ptr, bufs.v_scales.ptr, bufs.v_scales.ptr,
    stream);

  // v_opacities is dL/d(actual_opacity), need dL/d(logit_opacity)
  // Reuse v_opacities buffer for the chained gradient
  Training::launch_sigmoid_backward_chain(
    N, model.actual_opacities.ptr, bufs.v_opacities.ptr,
    bufs.v_opacities.ptr, stream);

  // ---- Adam updates ----
  Training::launch_adam_update(
    N * 3, model.means.ptr, bufs.v_means.ptr,
    model.m1_means.ptr, model.m2_means.ptr,
    cfg.lr_means, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon,
    adam_step, stream);

  Training::launch_adam_update(
    N * 4, model.quats.ptr, bufs.v_quats.ptr,
    model.m1_quats.ptr, model.m2_quats.ptr,
    cfg.lr_quats, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon,
    adam_step, stream);

  Training::launch_adam_update(
    N * 3, model.log_scales.ptr, bufs.v_scales.ptr,
    model.m1_log_scales.ptr, model.m2_log_scales.ptr,
    cfg.lr_scales, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon,
    adam_step, stream);

  Training::launch_adam_update(
    N, model.logit_opacities.ptr, bufs.v_opacities.ptr,
    model.m1_logit_opacities.ptr, model.m2_logit_opacities.ptr,
    cfg.lr_opacities, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon,
    adam_step, stream);

  // SH coefficients: [N, K, 3] layout interleaves DC and rest, so a single
  // learning rate for the entire buffer. DC vs rest split requires a strided
  // Adam kernel — deferred to Phase 2.
  Training::launch_adam_update(
    N * K_sh * 3, model.sh_coeffs.ptr, bufs.v_sh_coeffs.ptr,
    model.m1_sh_coeffs.ptr, model.m2_sh_coeffs.ptr,
    cfg.lr_sh, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon,
    adam_step, stream);

  cudaStreamSynchronize(stream);
  return h_loss;
}

// ============================================================================
// Save Gaussians to PLY
// ============================================================================

void save_model(
  const GaussianModel& model,
  const std::string& path,
  const std::uint32_t sh_degree)
{
  auto h_means {model.means.download()};
  auto h_sh {model.sh_coeffs.download()};
  auto h_logit_opacities {model.logit_opacities.download()};
  auto h_log_scales {model.log_scales.download()};
  auto h_quats {model.quats.download()};

  IO::write_gaussians_ply(
    path, model.N,
    h_means.data(), sh_degree,
    h_sh.data(),
    h_logit_opacities.data(),
    h_log_scales.data(),
    h_quats.data());
}

} // namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0]
              << " <colmap_sparse_dir> <images_dir>"
              << " [output.ply] [num_iterations]\n"
              << "\n"
              << "  colmap_sparse_dir  Directory with cameras.bin, images.bin,"
              << " points3D.bin\n"
              << "  images_dir         Directory with training images\n"
              << "  output.ply         Output PLY path"
              << " (default: output.ply)\n"
              << "  num_iterations     Number of training iterations"
              << " (default: 30000)\n";
    return 1;
  }

  const std::string colmap_dir {argv[1]};
  const std::string images_dir {argv[2]};
  const std::string output_ply {argc >= 4 ? argv[3] : "output.ply"};

  TrainConfig cfg;
  if (argc >= 5)
  {
    cfg.num_iterations = static_cast<std::uint32_t>(std::atoi(argv[4]));
  }

  // ---- Load COLMAP data ----
  std::cout << "Loading COLMAP data from " << colmap_dir << "...\n";

  const auto cameras {
    Colmap::read_cameras_binary(colmap_dir + "/cameras.bin")};
  const auto colmap_images {
    Colmap::read_images_binary(colmap_dir + "/images.bin")};
  const auto points {
    Colmap::read_points3d_binary(colmap_dir + "/points3D.bin")};

  std::cout << "  Cameras: " << cameras.size()
            << ", Images: " << colmap_images.size()
            << ", Points: " << points.size() << "\n";

  if (cameras.empty() || colmap_images.empty() || points.empty())
  {
    std::cerr << "ERROR: empty COLMAP data\n";
    return 1;
  }

  // ---- Load training images ----
  std::cout << "Loading training images from " << images_dir << "...\n";

  struct TrainImage
  {
    IO::Image image;
    std::vector<float> viewmat; // 16 floats, row-major 4x4
    std::vector<float> K;       // 9 floats, row-major 3x3
  };

  std::vector<TrainImage> train_images;
  train_images.reserve(colmap_images.size());

  for (const auto& cimg : colmap_images)
  {
    const fs::path img_path {fs::path(images_dir) / cimg.name};
    if (!fs::exists(img_path))
    {
      std::cerr << "  WARNING: skipping missing image " << img_path << "\n";
      continue;
    }

    TrainImage ti;
    ti.image = IO::load_image(img_path.string());
    ti.viewmat.resize(16);
    cimg.viewmat(ti.viewmat.data());

    const auto cam_it {std::find_if(cameras.begin(), cameras.end(),
      [&](const Colmap::Camera& c) { return c.id == cimg.camera_id; })};
    if (cam_it == cameras.end())
    {
      std::cerr << "  WARNING: camera " << cimg.camera_id
                << " not found for image " << cimg.name << "\n";
      continue;
    }
    ti.K.resize(9);
    cam_it->intrinsic_matrix(ti.K.data());

    train_images.push_back(std::move(ti));
  }

  std::cout << "  Loaded " << train_images.size() << " training images\n";

  if (train_images.empty())
  {
    std::cerr << "ERROR: no training images loaded\n";
    return 1;
  }

  // ---- Initialize Gaussians ----
  std::cout << "Initializing " << points.size() << " Gaussians from sparse"
            << " points...\n";

  GaussianModel model;
  initialize_from_colmap(model, points, cfg.sh_degree_max);

  // ---- Allocate training buffers ----
  std::uint32_t max_w {0}, max_h {0};
  for (const auto& ti : train_images)
  {
    max_w = std::max(max_w, ti.image.width);
    max_h = std::max(max_h, ti.image.height);
  }

  TrainBuffers bufs;
  bufs.allocate(model.N, model.K, max_w, max_h, cfg.tile_size);

  // ---- Create CUDA stream ----
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // ---- Training loop ----
  std::cout << "\nStarting training: " << cfg.num_iterations
            << " iterations, " << model.N << " Gaussians, "
            << train_images.size() << " images\n\n";

  std::mt19937 rng{42};

  const auto t_start {std::chrono::steady_clock::now()};
  float loss_accum {0.0f};
  std::uint32_t loss_count {0};

  for (std::uint32_t iter = 1; iter <= cfg.num_iterations; ++iter)
  {
    // SH degree schedule
    const std::uint32_t degrees_to_use {std::min(
      cfg.sh_degree_max,
      (iter - 1) / cfg.sh_degree_interval)};

    // Random image selection
    std::uniform_int_distribution<std::size_t> img_dist{
      0, train_images.size() - 1};
    const auto& ti {train_images[img_dist(rng)]};

    const float loss {train_step(
      model, bufs,
      ti.viewmat.data(), ti.K.data(), ti.image,
      degrees_to_use, cfg, iter, stream)};

    loss_accum += loss;
    ++loss_count;

    if (iter % cfg.print_interval == 0)
    {
      const auto t_now {std::chrono::steady_clock::now()};
      const double elapsed {std::chrono::duration<double>(
        t_now - t_start).count()};
      const double its_per_sec {iter / elapsed};

      std::cout << "  [" << iter << "/" << cfg.num_iterations
                << "]  loss=" << (loss_accum / loss_count)
                << "  sh_deg=" << degrees_to_use
                << "  " << its_per_sec << " it/s\n";
      loss_accum = 0.0f;
      loss_count = 0;
    }

    if (iter % cfg.save_interval == 0)
    {
      const std::string ckpt_path {
        output_ply.substr(0, output_ply.rfind('.')) + "_iter"
        + std::to_string(iter) + ".ply"};
      std::cout << "  Saving checkpoint: " << ckpt_path << "\n";
      save_model(model, ckpt_path,
        std::min(cfg.sh_degree_max, iter / cfg.sh_degree_interval));
    }
  }

  // ---- Save final model ----
  std::cout << "\nSaving final model to " << output_ply << "\n";
  save_model(model, output_ply, cfg.sh_degree_max);

  const auto t_end {std::chrono::steady_clock::now()};
  const double total_seconds {
    std::chrono::duration<double>(t_end - t_start).count()};
  std::cout << "Training complete: " << cfg.num_iterations << " iterations in "
            << total_seconds << "s ("
            << (cfg.num_iterations / total_seconds) << " it/s)\n";

  cudaStreamDestroy(stream);
  return 0;
}
