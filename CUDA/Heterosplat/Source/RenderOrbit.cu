#include "Core/CubOperations.h"
#include "IO/ImageIO.h"
#include "IO/PlyReader.h"
#include "Kernels/Heterosplat/IntersectOffset.h"
#include "Kernels/Heterosplat/IntersectTile.h"
#include "Kernels/Heterosplat/ProjectionEWA3DGSFused.h"
#include "Kernels/Heterosplat/RasterizeToPixels3DGS.h"
#include "Kernels/Heterosplat/SphericalHarmonics.h"
#include "Normalize/Convention.h"
#include "Normalize/Transform.h"
#include "Training/Activations.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

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
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
      }
    }
  }

  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;

  GpuBuffer(GpuBuffer&& o) noexcept : ptr{o.ptr}, count{o.count}
  {
    o.ptr = nullptr; o.count = 0;
  }

  GpuBuffer& operator=(GpuBuffer&& o) noexcept
  {
    if (this != &o)
    {
      if (ptr) cudaFree(ptr);
      ptr = o.ptr; count = o.count;
      o.ptr = nullptr; o.count = 0;
    }
    return *this;
  }

  ~GpuBuffer() { if (ptr) cudaFree(ptr); }

  void upload(const float* host, std::size_t n)
  {
    cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice);
  }

  void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }

  void zero() { if (count > 0) cudaMemset(ptr, 0, count * sizeof(T)); }

  std::vector<T> download() const
  {
    std::vector<T> host(count);
    cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
  }
};

// Generate orbit viewmat: camera orbits around center at given radius and
// elevation angle. Returns row-major 4x4 float.
std::vector<float> orbit_viewmat(
  const float center_x, const float center_y, const float center_z,
  const float radius, const float elevation_deg, const float azimuth_deg)
{
  constexpr float pi {3.14159265358979323846f};
  const float elev {elevation_deg * pi / 180.0f};
  const float azim {azimuth_deg * pi / 180.0f};

  // Camera position in world space
  const float cam_x {center_x + radius * std::cos(elev) * std::cos(azim)};
  const float cam_y {center_y + radius * std::cos(elev) * std::sin(azim)};
  const float cam_z {center_z + radius * std::sin(elev)};

  // Look-at: camera looks at center
  float fwd_x {center_x - cam_x};
  float fwd_y {center_y - cam_y};
  float fwd_z {center_z - cam_z};
  const float fwd_norm {std::sqrt(fwd_x*fwd_x + fwd_y*fwd_y + fwd_z*fwd_z)};
  fwd_x /= fwd_norm; fwd_y /= fwd_norm; fwd_z /= fwd_norm;

  // World up = +Z (COLMAP convention)
  float up_x {0.0f}, up_y {0.0f}, up_z {1.0f};

  // Right = fwd x up
  float right_x {fwd_y * up_z - fwd_z * up_y};
  float right_y {fwd_z * up_x - fwd_x * up_z};
  float right_z {fwd_x * up_y - fwd_y * up_x};
  const float right_norm {
    std::sqrt(right_x*right_x + right_y*right_y + right_z*right_z)};
  right_x /= right_norm; right_y /= right_norm; right_z /= right_norm;

  // Recompute up = right x fwd
  up_x = right_y * fwd_z - right_z * fwd_y;
  up_y = right_z * fwd_x - right_x * fwd_z;
  up_z = right_x * fwd_y - right_y * fwd_x;

  // World-to-camera rotation: R = [right; -up; fwd]
  // (OpenGL-like: -up because camera Y points down in screen space)
  // Translation: t = -R * cam_pos
  const float r00 {right_x}, r01 {right_y}, r02 {right_z};
  const float r10 {-up_x}, r11 {-up_y}, r12 {-up_z};
  const float r20 {fwd_x}, r21 {fwd_y}, r22 {fwd_z};

  const float tx {-(r00*cam_x + r01*cam_y + r02*cam_z)};
  const float ty {-(r10*cam_x + r11*cam_y + r12*cam_z)};
  const float tz {-(r20*cam_x + r21*cam_y + r22*cam_z)};

  return {
    r00, r01, r02, tx,
    r10, r11, r12, ty,
    r20, r21, r22, tz,
    0.0f, 0.0f, 0.0f, 1.0f
  };
}

void render_frame(
  const std::uint32_t N,
  const std::uint32_t K,
  const std::uint32_t sh_degree,
  const GpuBuffer<float>& d_means,
  const GpuBuffer<float>& d_actual_scales,
  const GpuBuffer<float>& d_actual_opacities,
  const GpuBuffer<float>& d_quats,
  const GpuBuffer<float>& d_sh_coeffs,
  const float* h_viewmat,
  const float* h_K_intrinsics,
  const std::uint32_t image_w,
  const std::uint32_t image_h,
  const std::uint32_t tile_size,
  std::vector<float>& out_pixels,
  cudaStream_t stream)
{
  const std::uint32_t I {1};
  const std::uint32_t B {1};
  const std::uint32_t C {1};
  const std::uint32_t tile_w {(image_w + tile_size - 1) / tile_size};
  const std::uint32_t tile_h {(image_h + tile_size - 1) / tile_size};
  const std::uint32_t n_pixels {image_w * image_h};

  GpuBuffer<float> d_viewmat(16);
  GpuBuffer<float> d_K(9);
  cudaMemcpyAsync(d_viewmat.ptr, h_viewmat, 16 * sizeof(float),
    cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_K.ptr, h_K_intrinsics, 9 * sizeof(float),
    cudaMemcpyHostToDevice, stream);

  // Camera center for view directions
  const float r00{h_viewmat[0]}, r01{h_viewmat[1]}, r02{h_viewmat[2]};
  const float r10{h_viewmat[4]}, r11{h_viewmat[5]}, r12{h_viewmat[6]};
  const float r20{h_viewmat[8]}, r21{h_viewmat[9]}, r22{h_viewmat[10]};
  const float tx{h_viewmat[3]}, ty{h_viewmat[7]}, tz{h_viewmat[11]};
  const float cam_x{-(r00*tx + r10*ty + r20*tz)};
  const float cam_y{-(r01*tx + r11*ty + r21*tz)};
  const float cam_z{-(r02*tx + r12*ty + r22*tz)};

  // View directions
  GpuBuffer<float> d_dirs(N * 3);
  Training::launch_compute_view_directions(
    N, d_means.ptr, cam_x, cam_y, cam_z, d_dirs.ptr, stream);

  // Projection
  GpuBuffer<std::int32_t> d_radii(N * 2);
  GpuBuffer<float> d_means2d(N * 2);
  GpuBuffer<float> d_depths(N);
  GpuBuffer<float> d_conics(N * 3);

  Kernels::Heterosplat::launch_projection_ewa_3dgs_fused_forward(
    B, C, N,
    d_means.ptr, nullptr, d_quats.ptr, d_actual_scales.ptr, nullptr,
    d_viewmat.ptr, d_K.ptr,
    image_w, image_h, 0.3f, 0.01f, 1e10f, 0.0f, 0,
    d_radii.ptr, d_means2d.ptr, d_depths.ptr, d_conics.ptr,
    nullptr, stream);

  // SH
  GpuBuffer<float> d_colors(N * 3);
  Kernels::Heterosplat::launch_spherical_harmonics_forward(
    N, K, sh_degree, d_dirs.ptr, d_sh_coeffs.ptr, nullptr,
    d_colors.ptr, stream);

  // Intersect tile pass 1
  GpuBuffer<std::int32_t> d_tiles_per_gauss(I * N);
  Kernels::Heterosplat::launch_intersect_tile_forward(
    false, I, N, 0,
    nullptr, nullptr,
    d_means2d.ptr, d_radii.ptr, d_depths.ptr,
    nullptr, nullptr, nullptr,
    tile_size, tile_w, tile_h,
    d_tiles_per_gauss.ptr, nullptr, nullptr, stream);

  // CUB prefix sum
  GpuBuffer<std::int64_t> d_cum_tiles(I * N);
  Core::cub_inclusive_sum_int32_to_int64(
    I * N, d_tiles_per_gauss.ptr, d_cum_tiles.ptr, stream);

  std::int64_t n_isects_64 {0};
  cudaMemcpyAsync(&n_isects_64, d_cum_tiles.ptr + (I * N - 1),
    sizeof(std::int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  const auto n_isects {static_cast<std::uint32_t>(n_isects_64)};

  if (n_isects == 0)
  {
    out_pixels.assign(n_pixels * 3, 0.0f);
    return;
  }

  // Intersect tile pass 2
  GpuBuffer<std::int64_t> d_isect_ids(n_isects);
  GpuBuffer<std::int32_t> d_flatten_ids(n_isects);
  Kernels::Heterosplat::launch_intersect_tile_forward(
    false, I, N, 0,
    nullptr, nullptr,
    d_means2d.ptr, d_radii.ptr, d_depths.ptr,
    nullptr, nullptr, d_cum_tiles.ptr,
    tile_size, tile_w, tile_h,
    nullptr, d_isect_ids.ptr, d_flatten_ids.ptr, stream);

  // CUB radix sort
  GpuBuffer<std::int64_t> d_isect_ids_sorted(n_isects);
  GpuBuffer<std::int32_t> d_flatten_ids_sorted(n_isects);
  Core::cub_radix_sort_pairs_int64_int32(
    n_isects,
    d_isect_ids.ptr, d_isect_ids_sorted.ptr,
    d_flatten_ids.ptr, d_flatten_ids_sorted.ptr,
    0, 64, stream);

  // Intersect offset
  GpuBuffer<std::int32_t> d_tile_offsets(I * tile_h * tile_w);
  d_tile_offsets.zero();
  Kernels::Heterosplat::launch_intersect_offset_forward(
    n_isects, d_isect_ids_sorted.ptr,
    I, tile_w, tile_h, d_tile_offsets.ptr, stream);

  // Rasterize
  GpuBuffer<float> d_render_colors(n_pixels * 3);
  GpuBuffer<float> d_render_alphas(n_pixels);
  GpuBuffer<std::int32_t> d_last_ids(n_pixels);

  Kernels::Heterosplat::launch_rasterize_to_pixels_3dgs_forward(
    I, N, n_isects, false,
    d_means2d.ptr, d_conics.ptr, d_colors.ptr,
    d_actual_opacities.ptr,
    nullptr, nullptr,
    image_w, image_h, tile_size,
    d_tile_offsets.ptr, d_flatten_ids_sorted.ptr,
    d_render_colors.ptr, d_render_alphas.ptr, d_last_ids.ptr,
    stream);

  out_pixels = d_render_colors.download();
}

} // namespace

int main(int argc, char** argv)
{
  // Strip the optional `--up auto|y-up|z-up` flag out of argv before the
  // existing positional parsing runs.
  std::string up_str {"auto"};
  std::vector<char*> argv_filtered;
  argv_filtered.reserve(argc);
  for (int i = 0; i < argc; ++i)
  {
    const std::string arg {argv[i]};
    if (arg == "--up" && i + 1 < argc)
    {
      up_str = argv[++i];
      continue;
    }
    argv_filtered.push_back(argv[i]);
  }
  argc = static_cast<int>(argv_filtered.size());
  argv = argv_filtered.data();

  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0]
              << " <input.ply>"
              << " [output_dir] [num_frames] [image_width] [image_height]"
              << " [orbit_radius] [elevation_deg] [fov_deg]"
              << " [--up auto|y-up|z-up]\n";
    return 1;
  }

  const std::string ply_path {argv[1]};
  const std::string output_dir {argc >= 3 ? argv[2] : "orbit_frames"};
  const std::uint32_t num_frames {
    argc >= 4 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 120u};
  const std::uint32_t image_w {
    argc >= 5 ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 800u};
  const std::uint32_t image_h {
    argc >= 6 ? static_cast<std::uint32_t>(std::atoi(argv[5])) : 800u};
  const float fov_deg {argc >= 9 ? std::atof(argv[8]) : 60.0f};

  // Load PLY
  std::cout << "Loading " << ply_path << "...\n";
  auto gaussians {IO::read_gaussians_ply(ply_path)};
  const std::uint32_t N {gaussians.num_gaussians};
  const std::uint32_t K {
    (gaussians.sh_degree + 1) * (gaussians.sh_degree + 1)};

  std::cout << "  " << N << " Gaussians, SH degree " << gaussians.sh_degree
            << " (K=" << K << ")\n";

  // Upload Gaussians to GPU. Means / quats / log-scales are uploaded up front
  // because we may rotate them in-place via Normalize::launch_apply_similarity_
  // transform before computing centroid + radius for the orbit camera.
  GpuBuffer<float> d_means(N * 3);
  GpuBuffer<float> d_quats(N * 4);
  GpuBuffer<float> d_log_scales(N * 3);
  GpuBuffer<float> d_sh_coeffs(N * K * 3);
  GpuBuffer<float> d_logit_opacities(N);
  GpuBuffer<float> d_actual_scales(N * 3);
  GpuBuffer<float> d_actual_opacities(N);

  d_means.upload(gaussians.means);
  d_quats.upload(gaussians.quats);
  d_log_scales.upload(gaussians.scales);
  d_sh_coeffs.upload(gaussians.sh_coeffs);
  d_logit_opacities.upload(gaussians.opacities);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Detect scene up-axis (or take CLI override). The orbit camera below
  // hardcodes world up = +Z, so any Y-up scene (Inria 3DGS / OpenGL / Blender
  // exports) renders sideways unless we rotate it first. The auto detector's
  // variance-ratio test is conservative (requires the smallest-variance axis
  // to be < 50% of the next-smallest); for borderline scenes pass --up y-up
  // / --up z-up to force.
  const auto up_conv {
    Normalize::parse_convention_string(up_str, gaussians.means.data(), N)};
  const auto up_axis {up_conv.up_axis};
  const char* up_label {
    up_axis == Normalize::UpAxis::y_up ? "Y-up" :
    up_axis == Normalize::UpAxis::z_up ? "Z-up" : "unknown"};
  std::cout << "  Up axis (" << up_str << "): " << up_label << "\n";

  if (up_axis == Normalize::UpAxis::y_up)
  {
    const auto R {Normalize::rotation_to_z_up(up_axis)};
    GpuBuffer<float> d_R(9);
    GpuBuffer<float> d_t(3);
    cudaMemcpyAsync(
      d_R.ptr, R.data(), 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
    d_t.zero();
    Normalize::launch_apply_similarity_transform(
      N, d_R.ptr, 1.0f, d_t.ptr,
      d_means.ptr, d_quats.ptr, d_log_scales.ptr, stream);
    cudaStreamSynchronize(stream);

    // Pull rotated means back so the host-side centroid + radius scan below
    // operates on the Z-up scene the orbit camera will actually see. (SH
    // coefficients are NOT rotated — proper rotation requires Wigner D
    // matrices; view-dependent appearance will be slightly off.)
    gaussians.means = d_means.download();
    std::cout << "  Rotated scene to Z-up.\n";
  }

  // Robust scene center + radius: use median per axis and 90th-percentile
  // distance to that center. Trained 3DGS scenes routinely have a long tail
  // of low-opacity flyaway splats far from the real geometry — the mean
  // centroid drifts toward them and the max distance is dominated by them,
  // so the auto orbit radius would shoot past the actual scene.
  std::vector<float> xs(N), ys(N), zs(N);
  for (std::uint32_t n = 0; n < N; ++n)
  {
    xs[n] = gaussians.means[n * 3 + 0];
    ys[n] = gaussians.means[n * 3 + 1];
    zs[n] = gaussians.means[n * 3 + 2];
  }
  const auto median = [](std::vector<float>& v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
  };
  const float cx {median(xs)};
  const float cy {median(ys)};
  const float cz {median(zs)};

  std::vector<float> dists(N);
  for (std::uint32_t n = 0; n < N; ++n)
  {
    const float dx {gaussians.means[n * 3 + 0] - cx};
    const float dy {gaussians.means[n * 3 + 1] - cy};
    const float dz {gaussians.means[n * 3 + 2] - cz};
    dists[n] = std::sqrt(dx*dx + dy*dy + dz*dz);
  }
  const std::size_t pct90_idx {static_cast<std::size_t>(0.90f * (N - 1))};
  std::nth_element(dists.begin(), dists.begin() + pct90_idx, dists.end());
  const float scene_radius {dists[pct90_idx]};
  const float max_dist {*std::max_element(dists.begin(), dists.end())};

  const float orbit_radius {
    argc >= 7 ? std::atof(argv[6]) : scene_radius * 2.5f};
  const float elevation_deg {
    argc >= 8 ? std::atof(argv[7]) : 30.0f};

  std::cout << "  Centroid (median): (" << cx << ", " << cy << ", " << cz
            << ")\n"
            << "  Scene radius (90%): " << scene_radius
            << " (max: " << max_dist << ")"
            << ", orbit radius: " << orbit_radius
            << ", elevation: " << elevation_deg << "°\n"
            << "  FOV: " << fov_deg << "°, "
            << image_w << "x" << image_h << ", "
            << num_frames << " frames\n";

  // Compute intrinsics from FOV
  constexpr float pi {3.14159265358979323846f};
  const float fx {
    static_cast<float>(image_w) / (2.0f * std::tan(fov_deg * pi / 360.0f))};
  const float fy {fx};
  const float cx_img {static_cast<float>(image_w) / 2.0f};
  const float cy_img {static_cast<float>(image_h) / 2.0f};
  const std::vector<float> h_K {
    fx, 0.0f, cx_img,
    0.0f, fy, cy_img,
    0.0f, 0.0f, 1.0f};

  Training::launch_exp_forward(
    N * 3, d_log_scales.ptr, d_actual_scales.ptr, stream);
  Training::launch_sigmoid_forward(
    N, d_logit_opacities.ptr, d_actual_opacities.ptr, stream);
  cudaStreamSynchronize(stream);

  // Create output directory
  fs::create_directories(output_dir);

  // Render frames
  constexpr std::uint32_t tile_size {16};
  std::cout << "\nRendering " << num_frames << " frames...\n";

  // Cinematic camera: azimuth pans once around (0 -> 360°) while elevation
  // sinusoidally bobs ±15° around the base elevation, completing two full
  // bob cycles per orbit. The bob frequency is an integer multiple of the
  // azimuth cycle so the loop closes cleanly.
  constexpr float bob_amplitude_deg {15.0f};
  constexpr int bob_cycles {2};
  for (std::uint32_t f = 0; f < num_frames; ++f)
  {
    const float t {static_cast<float>(f) / static_cast<float>(num_frames)};
    const float azimuth {360.0f * t};
    const float elevation_t {
      elevation_deg
      + bob_amplitude_deg * std::sin(2.0f * pi * bob_cycles * t)};
    const auto viewmat {orbit_viewmat(
      cx, cy, cz, orbit_radius, elevation_t, azimuth)};

    std::vector<float> pixels;
    render_frame(
      N, K, gaussians.sh_degree,
      d_means, d_actual_scales, d_actual_opacities, d_quats, d_sh_coeffs,
      viewmat.data(), h_K.data(),
      image_w, image_h, tile_size, pixels, stream);

    std::ostringstream fname;
    fname << output_dir << "/frame_"
          << std::setw(4) << std::setfill('0') << f << ".png";
    IO::save_image_png(fname.str(), image_w, image_h, pixels.data());

    if ((f + 1) % 10 == 0 || f == 0)
    {
      std::cout << "  Frame " << (f + 1) << "/" << num_frames << "\n";
    }
  }

  // Stitch with ffmpeg
  const std::string mp4_path {output_dir + "/orbit.mp4"};
  const std::string ffmpeg_cmd {
    "ffmpeg -y -framerate 60 -i " + output_dir + "/frame_%04d.png"
    + " -c:v libx264 -pix_fmt yuv420p -crf 18 " + mp4_path
    + " 2>/dev/null"};

  std::cout << "\nStitching " << mp4_path << "...\n";
  const int ret {std::system(ffmpeg_cmd.c_str())};
  if (ret == 0)
  {
    std::cout << "Done: " << mp4_path << "\n";
  }
  else
  {
    std::cerr << "ffmpeg failed (exit " << ret << "). Frames saved in "
              << output_dir << "/\n";
  }

  cudaStreamDestroy(stream);
  return 0;
}
