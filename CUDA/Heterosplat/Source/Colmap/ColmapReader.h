#ifndef COLMAP_COLMAP_READER_H
#define COLMAP_COLMAP_READER_H

#include <cstdint>
#include <string>
#include <vector>

namespace Colmap
{

enum class CameraModel : std::int32_t
{
  simple_pinhole = 0,
  pinhole = 1,
  simple_radial = 2,
  radial = 3,
  opencv = 4,
  opencv_fisheye = 5,
  full_opencv = 6,
  fov = 7,
  simple_radial_fisheye = 8,
  radial_fisheye = 9,
  thin_prism_fisheye = 10,
};

std::uint32_t camera_model_param_count(CameraModel model);

struct Camera
{
  std::uint32_t id;
  CameraModel model;
  std::uint64_t width;
  std::uint64_t height;
  std::vector<double> params;

  /// Extract pinhole intrinsics regardless of the actual model.
  /// For models with a single focal length, fx == fy.
  /// Distortion coefficients are discarded.
  void pinhole_intrinsics(
    double& fx, double& fy, double& cx, double& cy) const;

  /// Write a row-major 3x3 intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
  /// into `out`, casting from double to float.
  void intrinsic_matrix(float* out) const;
};

struct Image
{
  std::uint32_t id;
  std::uint32_t camera_id;
  double quaternion[4]; // (w, x, y, z) — world-to-camera rotation
  double translation[3]; // world-to-camera translation
  std::string name;

  /// Write a row-major 4x4 world-to-camera matrix into `out`,
  /// casting from double to float.
  void viewmat(float* out) const;
};

struct Point3D
{
  std::uint64_t id;
  double position[3];
  std::uint8_t color[3];
  double error;
};

std::vector<Camera> read_cameras_binary(const std::string& path);
std::vector<Image> read_images_binary(const std::string& path);
std::vector<Point3D> read_points3d_binary(const std::string& path);

} // namespace Colmap

#endif // COLMAP_COLMAP_READER_H
