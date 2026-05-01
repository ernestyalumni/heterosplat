#ifndef NORMALIZE_CONVENTION_H
#define NORMALIZE_CONVENTION_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace Normalize
{

enum class UpAxis : std::uint8_t
{
  y_up,
  z_up,
  unknown
};

struct SceneExtent
{
  std::array<float, 3> centroid;
  std::array<float, 3> extent;
  float max_extent;
};

struct Convention
{
  UpAxis up_axis;
  float scale_factor;
};

/// Detect the up-axis convention from a point cloud by analyzing the
/// distribution of coordinates. Assumes the "up" direction has:
/// 1) A biased mean (points tend to be above the ground plane)
/// 2) Less spread than the two horizontal axes
///
/// \param means     Flat array of [N, 3] positions.
/// \param N         Number of points.
/// \return Detected up axis.
UpAxis detect_up_axis(const float* means, std::uint32_t N);

/// Compute the axis-aligned bounding box extent of a point cloud.
SceneExtent compute_scene_extent(const float* means, std::uint32_t N);

/// Build the 3x3 rotation matrix that transforms from `src` convention to Z-up.
/// Returns identity if src is already Z-up or unknown.
std::array<float, 9> rotation_to_z_up(UpAxis src);

/// Build a similarity transform that normalizes a scene to fit within a unit
/// sphere centered at the origin.
/// Returns {rotation[9], scale, translation[3]}.
struct SimilarityTransform
{
  std::array<float, 9> rotation;
  float scale;
  std::array<float, 3> translation;
};

SimilarityTransform compute_normalization_transform(
  const float* means,
  std::uint32_t N,
  UpAxis convention);

/// Parse convention from command-line string.
/// Accepted: "auto", "y-up", "z-up"
Convention parse_convention_string(
  const std::string& str,
  const float* means,
  std::uint32_t N);

} // namespace Normalize

#endif // NORMALIZE_CONVENTION_H
