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

  // Parse header. Record property names in declaration order so we can
  // dispatch each binary float to its correct slot regardless of layout
  // variant (Inria 3DGS = with normals + strided f_rest;
  // gsplat export_splats = no normals + interleaved f_rest).
  std::vector<std::string> prop_names;
  std::string line;
  std::uint32_t num_vertices {0};
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
        prop_names.push_back(name);
      }
    }
  }

  if (num_vertices == 0)
  {
    throw std::runtime_error{"PLY has 0 vertices"};
  }

  std::uint32_t num_rest {0};
  bool has_normals {false};
  for (const auto& name : prop_names)
  {
    if (name == "nx") has_normals = true;
    if (name.rfind("f_rest_", 0) == 0) ++num_rest;
  }

  // Layout heuristic: Inria 3DGS writes normals + strided f_rest
  // (channel-major: all R coeffs, then G, then B). gsplat export_splats
  // omits normals and writes interleaved f_rest (per-coefficient RGB).
  const bool strided_sh_rest {has_normals};

  std::uint32_t sh_degree {0};
  if (num_rest > 0)
  {
    const std::uint32_t K {num_rest / 3 + 1};
    sh_degree = static_cast<std::uint32_t>(
      std::round(std::sqrt(static_cast<double>(K)))) - 1;
  }

  const std::uint32_t K {(sh_degree + 1) * (sh_degree + 1)};
  const std::uint32_t num_float_properties {
    static_cast<std::uint32_t>(prop_names.size())};
  const std::uint32_t expected_props {
    3 + (has_normals ? 3u : 0u) + 3 + num_rest + 1 + 3 + 4};

  if (num_float_properties != expected_props)
  {
    throw std::runtime_error{
      "unexpected property count: " + std::to_string(num_float_properties)
      + " (expected " + std::to_string(expected_props)
      + ", has_normals=" + (has_normals ? "true" : "false") + ")"};
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

    data.means[n * 3 + 0] = record[offset++];
    data.means[n * 3 + 1] = record[offset++];
    data.means[n * 3 + 2] = record[offset++];

    if (has_normals) offset += 3;

    float* gauss_sh {&data.sh_coeffs[n * K * 3]};
    gauss_sh[0] = record[offset++];
    gauss_sh[1] = record[offset++];
    gauss_sh[2] = record[offset++];

    if (strided_sh_rest)
    {
      // Inria layout: f_rest_N where N = c*(K-1) + (j-1).
      // Channel-major outer loop, coefficient-minor inner loop.
      for (std::uint32_t c = 0; c < 3; ++c)
      {
        for (std::uint32_t j = 1; j < K; ++j)
        {
          gauss_sh[j * 3 + c] = record[offset++];
        }
      }
    }
    else
    {
      // gsplat layout: per-coefficient RGB triple.
      for (std::uint32_t j = 1; j < K; ++j)
      {
        for (std::uint32_t c = 0; c < 3; ++c)
        {
          gauss_sh[j * 3 + c] = record[offset++];
        }
      }
    }

    data.opacities[n] = record[offset++];

    data.scales[n * 3 + 0] = record[offset++];
    data.scales[n * 3 + 1] = record[offset++];
    data.scales[n * 3 + 2] = record[offset++];

    data.quats[n * 4 + 0] = record[offset++];
    data.quats[n * 4 + 1] = record[offset++];
    data.quats[n * 4 + 2] = record[offset++];
    data.quats[n * 4 + 3] = record[offset++];
  }

  return data;
}

} // namespace IO
