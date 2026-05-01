#include "IO/PlyWriter.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace IO
{

void write_gaussians_ply(
  const std::string& path,
  const std::uint32_t num_gaussians,
  const float* means,
  const std::uint32_t sh_degree,
  const float* sh_coeffs,
  const float* opacities,
  const float* scales,
  const float* quats)
{
  const std::uint32_t sh_coeffs_per_channel {
    (sh_degree + 1) * (sh_degree + 1)};
  const std::uint32_t num_dc {3};
  const std::uint32_t num_rest {
    (sh_coeffs_per_channel > 1)
      ? (sh_coeffs_per_channel - 1) * 3
      : 0};

  std::ofstream file(path, std::ios::binary);
  if (!file)
  {
    throw std::runtime_error{"cannot open PLY for writing: " + path};
  }

  // PLY header
  file << "ply\n";
  file << "format binary_little_endian 1.0\n";
  file << "element vertex " << num_gaussians << "\n";

  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";

  file << "property float nx\n";
  file << "property float ny\n";
  file << "property float nz\n";

  file << "property float f_dc_0\n";
  file << "property float f_dc_1\n";
  file << "property float f_dc_2\n";

  // f_rest properties declared in Inria 3DGS strided order:
  // f_rest_N where N = c * (K-1) + (j-1). Channel-major outer (R, G, B),
  // coefficient-minor inner (j = 1..K-1). This matches the universally-read
  // 3DGS PLY layout from the original "3D Gaussian Splatting" code release.
  for (std::uint32_t i = 0; i < num_rest; ++i)
  {
    file << "property float f_rest_" << i << "\n";
  }

  file << "property float opacity\n";

  file << "property float scale_0\n";
  file << "property float scale_1\n";
  file << "property float scale_2\n";

  file << "property float rot_0\n";
  file << "property float rot_1\n";
  file << "property float rot_2\n";
  file << "property float rot_3\n";

  file << "end_header\n";

  // Binary data — one record per Gaussian
  const float zero {0.0f};
  for (std::uint32_t n = 0; n < num_gaussians; ++n)
  {
    // Position
    file.write(reinterpret_cast<const char*>(&means[n * 3]), 3 * sizeof(float));

    // Normal (always zero)
    file.write(reinterpret_cast<const char*>(&zero), sizeof(float));
    file.write(reinterpret_cast<const char*>(&zero), sizeof(float));
    file.write(reinterpret_cast<const char*>(&zero), sizeof(float));

    // SH DC coefficients: stored as [f_dc_0, f_dc_1, f_dc_2]
    // In our layout sh_coeffs is [N, K, 3], so DC for channel c is
    // sh_coeffs[n * K * 3 + 0 * 3 + c] = sh_coeffs[n * K * 3 + c]
    const float* gauss_sh {&sh_coeffs[n * sh_coeffs_per_channel * 3]};
    file.write(
      reinterpret_cast<const char*>(&gauss_sh[0]),
      num_dc * sizeof(float));

    // SH rest in strided (channel-major) order to match Inria's PLY layout.
    for (std::uint32_t c = 0; c < 3; ++c)
    {
      for (std::uint32_t j = 1; j < sh_coeffs_per_channel; ++j)
      {
        file.write(
          reinterpret_cast<const char*>(&gauss_sh[j * 3 + c]),
          sizeof(float));
      }
    }

    // Opacity (raw logit)
    file.write(
      reinterpret_cast<const char*>(&opacities[n]),
      sizeof(float));

    // Scale (log-space)
    file.write(
      reinterpret_cast<const char*>(&scales[n * 3]),
      3 * sizeof(float));

    // Rotation quaternion (w, x, y, z)
    file.write(
      reinterpret_cast<const char*>(&quats[n * 4]),
      4 * sizeof(float));
  }
}

} // namespace IO
