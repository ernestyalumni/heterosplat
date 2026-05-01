#include "Normalize/Convention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

namespace Normalize
{

UpAxis detect_up_axis(const float* means, const std::uint32_t N)
{
  if (N < 10) return UpAxis::unknown;

  // Compute mean and variance for each axis
  double sum[3] {0.0, 0.0, 0.0};
  for (std::uint32_t i = 0; i < N; ++i)
  {
    sum[0] += means[i * 3 + 0];
    sum[1] += means[i * 3 + 1];
    sum[2] += means[i * 3 + 2];
  }

  const double inv_n {1.0 / static_cast<double>(N)};
  const double mean[3] {sum[0] * inv_n, sum[1] * inv_n, sum[2] * inv_n};

  double var[3] {0.0, 0.0, 0.0};
  for (std::uint32_t i = 0; i < N; ++i)
  {
    for (int a = 0; a < 3; ++a)
    {
      const double d {means[i * 3 + a] - mean[a]};
      var[a] += d * d;
    }
  }
  var[0] *= inv_n;
  var[1] *= inv_n;
  var[2] *= inv_n;

  // The up axis typically has the smallest variance (vertical extent is smaller
  // for most scenes — buildings are wider than tall, terrain is flat, etc.)
  // Among the axes with small variance, pick the one with positive mean offset
  // (objects tend to be above the ground plane).

  // Find axis with minimum variance
  int min_var_axis {0};
  if (var[1] < var[min_var_axis]) min_var_axis = 1;
  if (var[2] < var[min_var_axis]) min_var_axis = 2;

  // Heuristic: if the minimum variance axis is clearly smaller (< 50% of
  // the next smallest), use it as up. Otherwise fall back to checking for
  // the standard conventions.
  double sorted_var[3] {var[0], var[1], var[2]};
  std::sort(sorted_var, sorted_var + 3);

  if (sorted_var[0] < sorted_var[1] * 0.5)
  {
    if (min_var_axis == 1) return UpAxis::y_up;
    if (min_var_axis == 2) return UpAxis::z_up;
  }

  // Fallback: check if the point cloud has a clear floor plane.
  // In Y-up (OpenGL/Blender default): Y has positive mean
  // In Z-up (COLMAP/engineering): Z has positive mean
  if (mean[1] > 0.0 && mean[1] > mean[2]) return UpAxis::y_up;
  if (mean[2] > 0.0 && mean[2] > mean[1]) return UpAxis::z_up;

  // Default to Z-up (COLMAP convention)
  return UpAxis::z_up;
}

SceneExtent compute_scene_extent(const float* means, const std::uint32_t N)
{
  SceneExtent result {};
  if (N == 0) return result;

  float min_v[3] {
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max()};
  float max_v[3] {
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest()};

  double sum[3] {0.0, 0.0, 0.0};

  for (std::uint32_t i = 0; i < N; ++i)
  {
    for (int a = 0; a < 3; ++a)
    {
      const float v {means[i * 3 + a]};
      min_v[a] = std::min(min_v[a], v);
      max_v[a] = std::max(max_v[a], v);
      sum[a] += v;
    }
  }

  const double inv_n {1.0 / static_cast<double>(N)};
  for (int a = 0; a < 3; ++a)
  {
    result.centroid[a] = static_cast<float>(sum[a] * inv_n);
    result.extent[a] = max_v[a] - min_v[a];
  }
  result.max_extent = std::max({
    result.extent[0], result.extent[1], result.extent[2], 1e-6f});

  return result;
}

std::array<float, 9> rotation_to_z_up(const UpAxis src)
{
  // Identity
  std::array<float, 9> R {1, 0, 0, 0, 1, 0, 0, 0, 1};

  if (src == UpAxis::y_up)
  {
    // Y-up to Z-up: rotate -90° around X
    // x' = x, y' = -z, z' = y
    // R = [[1,0,0],[0,0,-1],[0,1,0]]
    R = {1, 0, 0, 0, 0, -1, 0, 1, 0};
  }

  return R;
}

SimilarityTransform compute_normalization_transform(
  const float* means,
  const std::uint32_t N,
  const UpAxis convention)
{
  SimilarityTransform xform {};
  xform.rotation = rotation_to_z_up(convention);
  xform.scale = 1.0f;
  xform.translation = {0.0f, 0.0f, 0.0f};

  if (N == 0) return xform;

  const auto extent {compute_scene_extent(means, N)};

  // Scale to fit in unit sphere (diameter 2, radius 1)
  xform.scale = 2.0f / extent.max_extent;

  // Translation: after rotation and scaling, center at origin
  // Full transform: p' = scale * R * (p - centroid)
  // Rewritten as: p' = scale * R * p + (- scale * R * centroid)
  const auto& R {xform.rotation};
  const float cx {extent.centroid[0]};
  const float cy {extent.centroid[1]};
  const float cz {extent.centroid[2]};

  // t = -scale * R * centroid
  xform.translation[0] = -xform.scale * (R[0]*cx + R[1]*cy + R[2]*cz);
  xform.translation[1] = -xform.scale * (R[3]*cx + R[4]*cy + R[5]*cz);
  xform.translation[2] = -xform.scale * (R[6]*cx + R[7]*cy + R[8]*cz);

  return xform;
}

Convention parse_convention_string(
  const std::string& str,
  const float* means,
  const std::uint32_t N)
{
  Convention conv {};
  conv.scale_factor = 1.0f;

  if (str == "y-up" || str == "Y-up" || str == "yup")
  {
    conv.up_axis = UpAxis::y_up;
  }
  else if (str == "z-up" || str == "Z-up" || str == "zup")
  {
    conv.up_axis = UpAxis::z_up;
  }
  else
  {
    conv.up_axis = detect_up_axis(means, N);
  }

  return conv;
}

} // namespace Normalize
