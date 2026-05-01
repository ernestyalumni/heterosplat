#include "IO/PlyReader.h"
#include "IO/PlyWriter.h"
#include "Normalize/Convention.h"
#include "Normalize/Transform.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace
{

struct GpuFloats
{
  float* ptr {nullptr};
  std::uint32_t count {0};

  GpuFloats() = default;
  explicit GpuFloats(std::uint32_t n) : count{n}
  {
    cudaMalloc(&ptr, n * sizeof(float));
  }
  ~GpuFloats() { if (ptr) cudaFree(ptr); }
  GpuFloats(const GpuFloats&) = delete;
  GpuFloats& operator=(const GpuFloats&) = delete;

  void upload(const float* host)
  {
    cudaMemcpy(ptr, host, count * sizeof(float), cudaMemcpyHostToDevice);
  }
  void download(float* host) const
  {
    cudaMemcpy(host, ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
  }
};

void normalize_in_place(
  IO::GaussianData& data,
  const std::string& convention_str)
{
  const auto conv {Normalize::parse_convention_string(
    convention_str, data.means.data(), data.num_gaussians)};

  const auto xform {Normalize::compute_normalization_transform(
    data.means.data(), data.num_gaussians, conv.up_axis)};

  const std::uint32_t N {data.num_gaussians};

  // Upload to GPU
  GpuFloats d_means(N * 3);
  GpuFloats d_quats(N * 4);
  GpuFloats d_log_scales(N * 3);
  GpuFloats d_rotation(9);
  GpuFloats d_translation(3);

  d_means.upload(data.means.data());
  d_quats.upload(data.quats.data());
  d_log_scales.upload(data.scales.data());
  d_rotation.upload(xform.rotation.data());
  d_translation.upload(xform.translation.data());

  Normalize::launch_apply_similarity_transform(
    N,
    d_rotation.ptr,
    xform.scale,
    d_translation.ptr,
    d_means.ptr,
    d_quats.ptr,
    d_log_scales.ptr,
    nullptr);

  cudaDeviceSynchronize();

  // Download results
  d_means.download(data.means.data());
  d_quats.download(data.quats.data());
  d_log_scales.download(data.scales.data());
}

void print_usage()
{
  std::printf(
    "Usage: NormalizeAndFuse --source-a <A.ply> --source-b <B.ply> "
    "[--convention-a auto|y-up|z-up] [--convention-b auto|y-up|z-up] "
    "[--output <fused.ply>]\n\n"
    "Single-source mode:\n"
    "  NormalizeAndFuse --source-a <A.ply> [--convention-a auto|y-up|z-up] "
    "[--output <normalized.ply>]\n");
}

} // namespace

int main(int argc, char** argv)
{
  std::string source_a;
  std::string source_b;
  std::string convention_a_str {"auto"};
  std::string convention_b_str {"auto"};
  std::string output {"fused.ply"};

  for (int i = 1; i < argc; ++i)
  {
    const std::string arg {argv[i]};
    if (arg == "--source-a" && i + 1 < argc) source_a = argv[++i];
    else if (arg == "--source-b" && i + 1 < argc) source_b = argv[++i];
    else if (arg == "--convention-a" && i + 1 < argc) convention_a_str = argv[++i];
    else if (arg == "--convention-b" && i + 1 < argc) convention_b_str = argv[++i];
    else if (arg == "--output" && i + 1 < argc) output = argv[++i];
    else if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
  }

  if (source_a.empty())
  {
    print_usage();
    return 1;
  }

  std::printf("[NormalizeAndFuse] Loading source A: %s\n", source_a.c_str());
  auto data_a {IO::read_gaussians_ply(source_a)};
  std::printf("  -> %u Gaussians, SH degree %u\n",
    data_a.num_gaussians, data_a.sh_degree);

  std::printf("[NormalizeAndFuse] Normalizing A (convention: %s)\n",
    convention_a_str.c_str());
  normalize_in_place(data_a, convention_a_str);

  if (source_b.empty())
  {
    // Single-source: just normalize and write
    std::printf("[NormalizeAndFuse] Writing normalized output: %s\n",
      output.c_str());
    IO::write_gaussians_ply(
      output,
      data_a.num_gaussians,
      data_a.means.data(),
      data_a.sh_degree,
      data_a.sh_coeffs.data(),
      data_a.opacities.data(),
      data_a.scales.data(),
      data_a.quats.data());
    std::printf("[NormalizeAndFuse] Done. %u Gaussians written.\n",
      data_a.num_gaussians);
    return 0;
  }

  // Two-source fusion
  std::printf("[NormalizeAndFuse] Loading source B: %s\n", source_b.c_str());
  auto data_b {IO::read_gaussians_ply(source_b)};
  std::printf("  -> %u Gaussians, SH degree %u\n",
    data_b.num_gaussians, data_b.sh_degree);

  std::printf("[NormalizeAndFuse] Normalizing B (convention: %s)\n",
    convention_b_str.c_str());
  normalize_in_place(data_b, convention_b_str);

  // Fuse: concatenate both datasets
  // Use the minimum SH degree (drop higher-order terms from the richer one)
  const std::uint32_t fused_sh_degree {
    std::min(data_a.sh_degree, data_b.sh_degree)};
  const std::uint32_t K {(fused_sh_degree + 1) * (fused_sh_degree + 1)};
  const std::uint32_t fused_N {data_a.num_gaussians + data_b.num_gaussians};

  std::printf("[NormalizeAndFuse] Fusing: %u + %u = %u Gaussians (SH degree %u)\n",
    data_a.num_gaussians, data_b.num_gaussians, fused_N, fused_sh_degree);

  std::vector<float> fused_means(fused_N * 3);
  std::vector<float> fused_sh(fused_N * K * 3);
  std::vector<float> fused_opacities(fused_N);
  std::vector<float> fused_scales(fused_N * 3);
  std::vector<float> fused_quats(fused_N * 4);

  // Copy A
  const std::uint32_t K_a {(data_a.sh_degree + 1) * (data_a.sh_degree + 1)};
  for (std::uint32_t i = 0; i < data_a.num_gaussians; ++i)
  {
    std::copy_n(&data_a.means[i * 3], 3, &fused_means[i * 3]);
    std::copy_n(&data_a.sh_coeffs[i * K_a * 3], K * 3, &fused_sh[i * K * 3]);
    fused_opacities[i] = data_a.opacities[i];
    std::copy_n(&data_a.scales[i * 3], 3, &fused_scales[i * 3]);
    std::copy_n(&data_a.quats[i * 4], 4, &fused_quats[i * 4]);
  }

  // Copy B
  const std::uint32_t K_b {(data_b.sh_degree + 1) * (data_b.sh_degree + 1)};
  const std::uint32_t offset {data_a.num_gaussians};
  for (std::uint32_t i = 0; i < data_b.num_gaussians; ++i)
  {
    const std::uint32_t j {offset + i};
    std::copy_n(&data_b.means[i * 3], 3, &fused_means[j * 3]);
    std::copy_n(&data_b.sh_coeffs[i * K_b * 3], K * 3, &fused_sh[j * K * 3]);
    fused_opacities[j] = data_b.opacities[i];
    std::copy_n(&data_b.scales[i * 3], 3, &fused_scales[j * 3]);
    std::copy_n(&data_b.quats[i * 4], 4, &fused_quats[j * 4]);
  }

  std::printf("[NormalizeAndFuse] Writing fused output: %s\n", output.c_str());
  IO::write_gaussians_ply(
    output,
    fused_N,
    fused_means.data(),
    fused_sh_degree,
    fused_sh.data(),
    fused_opacities.data(),
    fused_scales.data(),
    fused_quats.data());

  std::printf("[NormalizeAndFuse] Done. %u Gaussians written.\n", fused_N);
  return 0;
}
