#include "Colmap/ColmapReader.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace Colmap
{

namespace
{

template <typename T>
T read_binary(std::ifstream& file)
{
  T value;
  file.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!file)
  {
    throw std::runtime_error{"unexpected end of COLMAP binary file"};
  }
  return value;
}

std::string read_null_terminated_string(std::ifstream& file)
{
  std::string result;
  char c;
  while (file.get(c) && c != '\0')
  {
    result += c;
  }
  return result;
}

} // namespace

std::uint32_t camera_model_param_count(const CameraModel model)
{
  switch (model)
  {
    case CameraModel::simple_pinhole: return 3;
    case CameraModel::pinhole: return 4;
    case CameraModel::simple_radial: return 4;
    case CameraModel::radial: return 5;
    case CameraModel::opencv: return 8;
    case CameraModel::opencv_fisheye: return 8;
    case CameraModel::full_opencv: return 12;
    case CameraModel::fov: return 5;
    case CameraModel::simple_radial_fisheye: return 4;
    case CameraModel::radial_fisheye: return 5;
    case CameraModel::thin_prism_fisheye: return 12;
    default:
      throw std::runtime_error{
        "unknown COLMAP camera model: " +
        std::to_string(static_cast<int>(model))};
  }
}

void Camera::pinhole_intrinsics(
  double& fx, double& fy, double& cx, double& cy) const
{
  switch (model)
  {
    case CameraModel::simple_pinhole:
    case CameraModel::simple_radial:
    case CameraModel::simple_radial_fisheye:
      fx = fy = params[0];
      cx = params[1];
      cy = params[2];
      break;

    case CameraModel::pinhole:
    case CameraModel::opencv:
    case CameraModel::opencv_fisheye:
    case CameraModel::full_opencv:
    case CameraModel::thin_prism_fisheye:
      fx = params[0];
      fy = params[1];
      cx = params[2];
      cy = params[3];
      break;

    case CameraModel::radial:
    case CameraModel::radial_fisheye:
    case CameraModel::fov:
      fx = fy = params[0];
      cx = params[1];
      cy = params[2];
      break;

    default:
      throw std::runtime_error{"unknown camera model for intrinsic extraction"};
  }
}

void Camera::intrinsic_matrix(float* out) const
{
  double fx, fy, cx, cy;
  pinhole_intrinsics(fx, fy, cx, cy);

  out[0] = static_cast<float>(fx);
  out[1] = 0.0f;
  out[2] = static_cast<float>(cx);
  out[3] = 0.0f;
  out[4] = static_cast<float>(fy);
  out[5] = static_cast<float>(cy);
  out[6] = 0.0f;
  out[7] = 0.0f;
  out[8] = 1.0f;
}

void Image::viewmat(float* out) const
{
  const double w {quaternion[0]};
  const double x {quaternion[1]};
  const double y {quaternion[2]};
  const double z {quaternion[3]};

  // Quaternion to rotation matrix (world-to-camera)
  out[0]  = static_cast<float>(1.0 - 2.0 * (y * y + z * z));
  out[1]  = static_cast<float>(2.0 * (x * y - w * z));
  out[2]  = static_cast<float>(2.0 * (x * z + w * y));
  out[3]  = static_cast<float>(translation[0]);

  out[4]  = static_cast<float>(2.0 * (x * y + w * z));
  out[5]  = static_cast<float>(1.0 - 2.0 * (x * x + z * z));
  out[6]  = static_cast<float>(2.0 * (y * z - w * x));
  out[7]  = static_cast<float>(translation[1]);

  out[8]  = static_cast<float>(2.0 * (x * z - w * y));
  out[9]  = static_cast<float>(2.0 * (y * z + w * x));
  out[10] = static_cast<float>(1.0 - 2.0 * (x * x + y * y));
  out[11] = static_cast<float>(translation[2]);

  out[12] = 0.0f;
  out[13] = 0.0f;
  out[14] = 0.0f;
  out[15] = 1.0f;
}

std::vector<Camera> read_cameras_binary(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file)
  {
    throw std::runtime_error{"cannot open cameras binary: " + path};
  }

  const auto num_cameras {read_binary<std::uint64_t>(file)};
  std::vector<Camera> cameras;
  cameras.reserve(num_cameras);

  for (std::uint64_t i = 0; i < num_cameras; ++i)
  {
    Camera camera;
    camera.id = read_binary<std::uint32_t>(file);
    camera.model =
      static_cast<CameraModel>(read_binary<std::int32_t>(file));
    camera.width = read_binary<std::uint64_t>(file);
    camera.height = read_binary<std::uint64_t>(file);

    const auto num_params {camera_model_param_count(camera.model)};
    camera.params.resize(num_params);
    for (std::uint32_t j = 0; j < num_params; ++j)
    {
      camera.params[j] = read_binary<double>(file);
    }

    cameras.push_back(std::move(camera));
  }

  return cameras;
}

std::vector<Image> read_images_binary(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file)
  {
    throw std::runtime_error{"cannot open images binary: " + path};
  }

  const auto num_images {read_binary<std::uint64_t>(file)};
  std::vector<Image> images;
  images.reserve(num_images);

  for (std::uint64_t i = 0; i < num_images; ++i)
  {
    Image image;
    image.id = read_binary<std::uint32_t>(file);

    image.quaternion[0] = read_binary<double>(file);
    image.quaternion[1] = read_binary<double>(file);
    image.quaternion[2] = read_binary<double>(file);
    image.quaternion[3] = read_binary<double>(file);

    image.translation[0] = read_binary<double>(file);
    image.translation[1] = read_binary<double>(file);
    image.translation[2] = read_binary<double>(file);

    image.camera_id = read_binary<std::uint32_t>(file);
    image.name = read_null_terminated_string(file);

    const auto num_points_2d {read_binary<std::uint64_t>(file)};
    for (std::uint64_t j = 0; j < num_points_2d; ++j)
    {
      read_binary<double>(file); // x
      read_binary<double>(file); // y
      read_binary<std::uint64_t>(file); // point3D_id
    }

    images.push_back(std::move(image));
  }

  return images;
}

std::vector<Point3D> read_points3d_binary(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file)
  {
    throw std::runtime_error{"cannot open points3D binary: " + path};
  }

  const auto num_points {read_binary<std::uint64_t>(file)};
  std::vector<Point3D> points;
  points.reserve(num_points);

  for (std::uint64_t i = 0; i < num_points; ++i)
  {
    Point3D point;
    point.id = read_binary<std::uint64_t>(file);

    point.position[0] = read_binary<double>(file);
    point.position[1] = read_binary<double>(file);
    point.position[2] = read_binary<double>(file);

    point.color[0] = read_binary<std::uint8_t>(file);
    point.color[1] = read_binary<std::uint8_t>(file);
    point.color[2] = read_binary<std::uint8_t>(file);

    point.error = read_binary<double>(file);

    const auto track_length {read_binary<std::uint64_t>(file)};
    for (std::uint64_t j = 0; j < track_length; ++j)
    {
      read_binary<std::uint32_t>(file); // image_id
      read_binary<std::uint32_t>(file); // point2D_idx
    }

    points.push_back(std::move(point));
  }

  return points;
}

} // namespace Colmap
