#include "IO/PlyReader.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace IO
{

GaussianData read_gaussians_ply(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file)
  {
    throw std::runtime_error{"cannot open PLY for reading: " + path};
  }

  // Parse header
  std::string line;
  std::uint32_t num_vertices {0};
  std::uint32_t num_float_properties {0};
  std::uint32_t num_rest {0};
  bool in_vertex_element {false};

  while (std::getline(file, line))
  {
    if (line == "end_header") break;

    std::istringstream iss(line);
    std::string token;
    iss >> token;

    if (token == "element")
    {
      std::string elem_type;
      iss >> elem_type;
      if (elem_type == "vertex")
      {
        iss >> num_vertices;
        in_vertex_element = true;
      }
      else
      {
        in_vertex_element = false;
      }
    }
    else if (token == "property" && in_vertex_element)
    {
      std::string dtype, name;
      iss >> dtype >> name;
      if (dtype == "float" || dtype == "float32")
      {
        ++num_float_properties;
        if (name.substr(0, 7) == "f_rest_")
        {
          ++num_rest;
        }
      }
    }
  }

  if (num_vertices == 0)
  {
    throw std::runtime_error{"PLY has 0 vertices"};
  }

  // Determine SH degree from rest count.
  // num_rest = (K - 1) * 3 where K = (degree+1)^2
  // K = num_rest / 3 + 1
  // degree = sqrt(K) - 1
  std::uint32_t sh_degree {0};
  if (num_rest > 0)
  {
    const std::uint32_t K {num_rest / 3 + 1};
    sh_degree = static_cast<std::uint32_t>(
      std::round(std::sqrt(static_cast<double>(K)))) - 1;
  }

  const std::uint32_t K {(sh_degree + 1) * (sh_degree + 1)};

  // Expected: 3(pos) + 3(normal) + 3(DC) + num_rest + 1(opacity) + 3(scale) + 4(quat)
  const std::uint32_t expected_props {3 + 3 + 3 + num_rest + 1 + 3 + 4};
  if (num_float_properties != expected_props)
  {
    throw std::runtime_error{
      "unexpected property count: " + std::to_string(num_float_properties)
      + " (expected " + std::to_string(expected_props) + ")"};
  }

  GaussianData data;
  data.num_gaussians = num_vertices;
  data.sh_degree = sh_degree;
  data.means.resize(num_vertices * 3);
  data.sh_coeffs.resize(num_vertices * K * 3, 0.0f);
  data.opacities.resize(num_vertices);
  data.scales.resize(num_vertices * 3);
  data.quats.resize(num_vertices * 4);

  const std::size_t record_bytes {num_float_properties * sizeof(float)};
  std::vector<float> record(num_float_properties);

  for (std::uint32_t n = 0; n < num_vertices; ++n)
  {
    file.read(reinterpret_cast<char*>(record.data()), record_bytes);
    if (!file)
    {
      throw std::runtime_error{
        "PLY read failed at vertex " + std::to_string(n)};
    }

    std::uint32_t offset {0};

    // Position
    data.means[n * 3 + 0] = record[offset++];
    data.means[n * 3 + 1] = record[offset++];
    data.means[n * 3 + 2] = record[offset++];

    // Normal (skip)
    offset += 3;

    // SH DC
    float* gauss_sh {&data.sh_coeffs[n * K * 3]};
    gauss_sh[0] = record[offset++];
    gauss_sh[1] = record[offset++];
    gauss_sh[2] = record[offset++];

    // SH rest: interleaved as [coeff1_R, coeff1_G, coeff1_B, coeff2_R, ...]
    for (std::uint32_t j = 1; j < K; ++j)
    {
      for (std::uint32_t c = 0; c < 3; ++c)
      {
        gauss_sh[j * 3 + c] = record[offset++];
      }
    }

    // Opacity
    data.opacities[n] = record[offset++];

    // Scale
    data.scales[n * 3 + 0] = record[offset++];
    data.scales[n * 3 + 1] = record[offset++];
    data.scales[n * 3 + 2] = record[offset++];

    // Quaternion (w, x, y, z)
    data.quats[n * 4 + 0] = record[offset++];
    data.quats[n * 4 + 1] = record[offset++];
    data.quats[n * 4 + 2] = record[offset++];
    data.quats[n * 4 + 3] = record[offset++];
  }

  return data;
}

} // namespace IO
