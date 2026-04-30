#include "Colmap/ColmapReader.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace GoogleUnitTests
{
namespace Colmap
{

namespace
{

template <typename T>
void write_binary(std::ofstream& file, const T value)
{
  file.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_null_terminated(std::ofstream& file, const std::string& s)
{
  file.write(s.c_str(), static_cast<std::streamsize>(s.size() + 1));
}

class TempFile
{
  public:
    TempFile()
    {
      path_ = std::tmpnam(nullptr);
    }

    ~TempFile()
    {
      std::remove(path_.c_str());
    }

    const std::string& path() const { return path_; }

  private:
    std::string path_;
};

} // namespace

TEST(ColmapReader, ReadsCamerasBinaryPinhole)
{
  TempFile tmp;
  {
    std::ofstream file(tmp.path(), std::ios::binary);
    write_binary<std::uint64_t>(file, 1); // num_cameras

    write_binary<std::uint32_t>(file, 1); // camera_id
    write_binary<std::int32_t>(file, 1); // PINHOLE
    write_binary<std::uint64_t>(file, 1920); // width
    write_binary<std::uint64_t>(file, 1080); // height
    write_binary<double>(file, 500.0); // fx
    write_binary<double>(file, 500.0); // fy
    write_binary<double>(file, 960.0); // cx
    write_binary<double>(file, 540.0); // cy
  }

  const auto cameras {::Colmap::read_cameras_binary(tmp.path())};
  ASSERT_EQ(cameras.size(), 1u);
  EXPECT_EQ(cameras[0].id, 1u);
  EXPECT_EQ(cameras[0].model, ::Colmap::CameraModel::pinhole);
  EXPECT_EQ(cameras[0].width, 1920u);
  EXPECT_EQ(cameras[0].height, 1080u);
  ASSERT_EQ(cameras[0].params.size(), 4u);
  EXPECT_DOUBLE_EQ(cameras[0].params[0], 500.0);
  EXPECT_DOUBLE_EQ(cameras[0].params[1], 500.0);
  EXPECT_DOUBLE_EQ(cameras[0].params[2], 960.0);
  EXPECT_DOUBLE_EQ(cameras[0].params[3], 540.0);
}

TEST(ColmapReader, ReadsCamerasBinarySimplePinhole)
{
  TempFile tmp;
  {
    std::ofstream file(tmp.path(), std::ios::binary);
    write_binary<std::uint64_t>(file, 1);
    write_binary<std::uint32_t>(file, 2);
    write_binary<std::int32_t>(file, 0); // SIMPLE_PINHOLE
    write_binary<std::uint64_t>(file, 640);
    write_binary<std::uint64_t>(file, 480);
    write_binary<double>(file, 300.0); // f
    write_binary<double>(file, 320.0); // cx
    write_binary<double>(file, 240.0); // cy
  }

  const auto cameras {::Colmap::read_cameras_binary(tmp.path())};
  ASSERT_EQ(cameras.size(), 1u);
  EXPECT_EQ(cameras[0].model, ::Colmap::CameraModel::simple_pinhole);
  ASSERT_EQ(cameras[0].params.size(), 3u);

  double fx, fy, cx, cy;
  cameras[0].pinhole_intrinsics(fx, fy, cx, cy);
  EXPECT_DOUBLE_EQ(fx, 300.0);
  EXPECT_DOUBLE_EQ(fy, 300.0);
  EXPECT_DOUBLE_EQ(cx, 320.0);
  EXPECT_DOUBLE_EQ(cy, 240.0);
}

TEST(ColmapReader, ReadsImagesBinary)
{
  TempFile tmp;
  {
    std::ofstream file(tmp.path(), std::ios::binary);
    write_binary<std::uint64_t>(file, 1); // num_images

    write_binary<std::uint32_t>(file, 7); // image_id
    write_binary<double>(file, 1.0); // qw
    write_binary<double>(file, 0.0); // qx
    write_binary<double>(file, 0.0); // qy
    write_binary<double>(file, 0.0); // qz
    write_binary<double>(file, 0.5); // tx
    write_binary<double>(file, -0.3); // ty
    write_binary<double>(file, 2.0); // tz
    write_binary<std::uint32_t>(file, 1); // camera_id
    write_null_terminated(file, "frame_0001.jpg");

    write_binary<std::uint64_t>(file, 2); // num_points2D
    write_binary<double>(file, 100.0); // x
    write_binary<double>(file, 200.0); // y
    write_binary<std::uint64_t>(file, 42); // point3D_id
    write_binary<double>(file, 300.0);
    write_binary<double>(file, 400.0);
    write_binary<std::uint64_t>(file, UINT64_MAX); // invalid point3D_id
  }

  const auto images {::Colmap::read_images_binary(tmp.path())};
  ASSERT_EQ(images.size(), 1u);
  EXPECT_EQ(images[0].id, 7u);
  EXPECT_EQ(images[0].camera_id, 1u);
  EXPECT_DOUBLE_EQ(images[0].quaternion[0], 1.0);
  EXPECT_DOUBLE_EQ(images[0].quaternion[1], 0.0);
  EXPECT_DOUBLE_EQ(images[0].quaternion[2], 0.0);
  EXPECT_DOUBLE_EQ(images[0].quaternion[3], 0.0);
  EXPECT_DOUBLE_EQ(images[0].translation[0], 0.5);
  EXPECT_DOUBLE_EQ(images[0].translation[1], -0.3);
  EXPECT_DOUBLE_EQ(images[0].translation[2], 2.0);
  EXPECT_EQ(images[0].name, "frame_0001.jpg");
}

TEST(ColmapReader, ReadsPoints3DBinary)
{
  TempFile tmp;
  {
    std::ofstream file(tmp.path(), std::ios::binary);
    write_binary<std::uint64_t>(file, 2); // num_points

    write_binary<std::uint64_t>(file, 100); // point3D_id
    write_binary<double>(file, 1.5);
    write_binary<double>(file, -2.0);
    write_binary<double>(file, 3.0);
    write_binary<std::uint8_t>(file, 255);
    write_binary<std::uint8_t>(file, 128);
    write_binary<std::uint8_t>(file, 0);
    write_binary<double>(file, 0.5); // error
    write_binary<std::uint64_t>(file, 1); // track_length
    write_binary<std::uint32_t>(file, 7); // image_id
    write_binary<std::uint32_t>(file, 0); // point2D_idx

    write_binary<std::uint64_t>(file, 200);
    write_binary<double>(file, -1.0);
    write_binary<double>(file, 0.0);
    write_binary<double>(file, 4.0);
    write_binary<std::uint8_t>(file, 10);
    write_binary<std::uint8_t>(file, 20);
    write_binary<std::uint8_t>(file, 30);
    write_binary<double>(file, 1.2);
    write_binary<std::uint64_t>(file, 0); // no tracks
  }

  const auto points {::Colmap::read_points3d_binary(tmp.path())};
  ASSERT_EQ(points.size(), 2u);

  EXPECT_EQ(points[0].id, 100u);
  EXPECT_DOUBLE_EQ(points[0].position[0], 1.5);
  EXPECT_DOUBLE_EQ(points[0].position[1], -2.0);
  EXPECT_DOUBLE_EQ(points[0].position[2], 3.0);
  EXPECT_EQ(points[0].color[0], 255);
  EXPECT_EQ(points[0].color[1], 128);
  EXPECT_EQ(points[0].color[2], 0);
  EXPECT_DOUBLE_EQ(points[0].error, 0.5);

  EXPECT_EQ(points[1].id, 200u);
  EXPECT_DOUBLE_EQ(points[1].position[0], -1.0);
}

TEST(ColmapReader, IdentityQuaternionProducesIdentityViewmat)
{
  ::Colmap::Image image {};
  image.quaternion[0] = 1.0;
  image.quaternion[1] = 0.0;
  image.quaternion[2] = 0.0;
  image.quaternion[3] = 0.0;
  image.translation[0] = 0.0;
  image.translation[1] = 0.0;
  image.translation[2] = 0.0;

  float viewmat[16];
  image.viewmat(viewmat);

  const float expected[16] {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1};

  for (int i = 0; i < 16; ++i)
  {
    EXPECT_NEAR(viewmat[i], expected[i], 1e-6f) << "index " << i;
  }
}

TEST(ColmapReader, ViewmatEncodesTranslation)
{
  ::Colmap::Image image {};
  image.quaternion[0] = 1.0;
  image.quaternion[1] = 0.0;
  image.quaternion[2] = 0.0;
  image.quaternion[3] = 0.0;
  image.translation[0] = 1.5;
  image.translation[1] = -2.0;
  image.translation[2] = 3.0;

  float viewmat[16];
  image.viewmat(viewmat);

  EXPECT_NEAR(viewmat[3], 1.5f, 1e-6f);
  EXPECT_NEAR(viewmat[7], -2.0f, 1e-6f);
  EXPECT_NEAR(viewmat[11], 3.0f, 1e-6f);
}

TEST(ColmapReader, Rotation90DegreesAboutZ)
{
  // 90-degree rotation about Z: quat = (cos(45), 0, 0, sin(45))
  const double s {std::sin(M_PI / 4.0)};
  const double c {std::cos(M_PI / 4.0)};

  ::Colmap::Image image {};
  image.quaternion[0] = c;
  image.quaternion[1] = 0.0;
  image.quaternion[2] = 0.0;
  image.quaternion[3] = s;
  image.translation[0] = 0.0;
  image.translation[1] = 0.0;
  image.translation[2] = 0.0;

  float viewmat[16];
  image.viewmat(viewmat);

  // R for 90-deg about Z: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
  EXPECT_NEAR(viewmat[0], 0.0f, 1e-6f);
  EXPECT_NEAR(viewmat[1], -1.0f, 1e-6f);
  EXPECT_NEAR(viewmat[4], 1.0f, 1e-6f);
  EXPECT_NEAR(viewmat[5], 0.0f, 1e-6f);
  EXPECT_NEAR(viewmat[10], 1.0f, 1e-6f);
}

TEST(ColmapReader, IntrinsicMatrixMatchesPinholeParams)
{
  ::Colmap::Camera camera;
  camera.model = ::Colmap::CameraModel::pinhole;
  camera.params = {500.0, 500.0, 320.0, 240.0};

  float K[9];
  camera.intrinsic_matrix(K);

  EXPECT_FLOAT_EQ(K[0], 500.0f);
  EXPECT_FLOAT_EQ(K[1], 0.0f);
  EXPECT_FLOAT_EQ(K[2], 320.0f);
  EXPECT_FLOAT_EQ(K[3], 0.0f);
  EXPECT_FLOAT_EQ(K[4], 500.0f);
  EXPECT_FLOAT_EQ(K[5], 240.0f);
  EXPECT_FLOAT_EQ(K[6], 0.0f);
  EXPECT_FLOAT_EQ(K[7], 0.0f);
  EXPECT_FLOAT_EQ(K[8], 1.0f);
}

TEST(ColmapReader, MultipleCamerasRoundTrip)
{
  TempFile tmp;
  {
    std::ofstream file(tmp.path(), std::ios::binary);
    write_binary<std::uint64_t>(file, 2); // num_cameras

    // Camera 1: PINHOLE
    write_binary<std::uint32_t>(file, 1);
    write_binary<std::int32_t>(file, 1);
    write_binary<std::uint64_t>(file, 800);
    write_binary<std::uint64_t>(file, 600);
    write_binary<double>(file, 400.0);
    write_binary<double>(file, 400.0);
    write_binary<double>(file, 400.0);
    write_binary<double>(file, 300.0);

    // Camera 2: SIMPLE_RADIAL
    write_binary<std::uint32_t>(file, 2);
    write_binary<std::int32_t>(file, 2);
    write_binary<std::uint64_t>(file, 640);
    write_binary<std::uint64_t>(file, 480);
    write_binary<double>(file, 350.0); // f
    write_binary<double>(file, 320.0); // cx
    write_binary<double>(file, 240.0); // cy
    write_binary<double>(file, 0.05); // k
  }

  const auto cameras {::Colmap::read_cameras_binary(tmp.path())};
  ASSERT_EQ(cameras.size(), 2u);
  EXPECT_EQ(cameras[0].id, 1u);
  EXPECT_EQ(cameras[1].id, 2u);
  EXPECT_EQ(cameras[1].model, ::Colmap::CameraModel::simple_radial);
  ASSERT_EQ(cameras[1].params.size(), 4u);
  EXPECT_DOUBLE_EQ(cameras[1].params[3], 0.05);
}

} // namespace Colmap
} // namespace GoogleUnitTests
