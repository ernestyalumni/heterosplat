#ifndef IO_PLY_READER_H
#define IO_PLY_READER_H

#include <cstdint>
#include <string>
#include <vector>

namespace IO
{

struct GaussianData
{
  std::uint32_t num_gaussians;
  std::uint32_t sh_degree;

  std::vector<float> means;           // [N, 3]
  std::vector<float> sh_coeffs;       // [N, K, 3], K = (sh_degree+1)^2
  std::vector<float> opacities;       // [N], raw logit
  std::vector<float> scales;          // [N, 3], log-space
  std::vector<float> quats;           // [N, 4], (w, x, y, z)
};

GaussianData read_gaussians_ply(const std::string& path);

} // namespace IO

#endif // IO_PLY_READER_H
