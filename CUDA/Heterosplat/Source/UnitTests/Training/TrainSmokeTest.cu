#include "Colmap/ColmapReader.h"
#include "IO/ImageIO.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace GoogleUnitTests::Training
{

namespace fs = std::filesystem;

namespace
{

struct TempDir
{
  std::string path;
  TempDir()
  {
    path = std::tmpnam(nullptr);
    fs::create_directories(path);
  }
  ~TempDir() { fs::remove_all(path); }
};

template <typename T>
void write_binary(std::ofstream& f, const T& val)
{
  f.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

void write_null_string(std::ofstream& f, const std::string& s)
{
  f.write(s.c_str(), s.size() + 1);
}

void create_synthetic_colmap(
  const std::string& sparse_dir,
  const std::string& images_dir,
  const std::uint32_t num_cameras,
  const std::uint32_t num_images,
  const std::uint32_t num_points,
  const std::uint32_t img_width,
  const std::uint32_t img_height)
{
  fs::create_directories(sparse_dir);
  fs::create_directories(images_dir);

  // cameras.bin — all pinhole
  {
    std::ofstream f(sparse_dir + "/cameras.bin", std::ios::binary);
    const std::uint64_t count {num_cameras};
    write_binary(f, count);
    for (std::uint32_t c = 0; c < num_cameras; ++c)
    {
      const std::uint32_t cam_id {c + 1};
      const std::int32_t model {1}; // PINHOLE
      const std::uint64_t w {img_width};
      const std::uint64_t h {img_height};
      write_binary(f, cam_id);
      write_binary(f, model);
      write_binary(f, w);
      write_binary(f, h);
      // PINHOLE: fx, fy, cx, cy
      const double fx {500.0};
      const double fy {500.0};
      const double cx {static_cast<double>(img_width) / 2.0};
      const double cy {static_cast<double>(img_height) / 2.0};
      write_binary(f, fx);
      write_binary(f, fy);
      write_binary(f, cx);
      write_binary(f, cy);
    }
  }

  // images.bin
  {
    std::ofstream f(sparse_dir + "/images.bin", std::ios::binary);
    const std::uint64_t count {num_images};
    write_binary(f, count);
    for (std::uint32_t i = 0; i < num_images; ++i)
    {
      const std::uint32_t img_id {i + 1};
      // Identity quaternion
      const double qw {1.0}, qx {0.0}, qy {0.0}, qz {0.0};
      // Camera at z = -5, looking down +Z
      const double tx {0.0}, ty {0.0}, tz {5.0};
      const std::uint32_t cam_id {1};
      const std::string name {"img_" + std::to_string(i) + ".png"};
      const std::uint64_t num_points2d {0};

      write_binary(f, img_id);
      write_binary(f, qw);
      write_binary(f, qx);
      write_binary(f, qy);
      write_binary(f, qz);
      write_binary(f, tx);
      write_binary(f, ty);
      write_binary(f, tz);
      write_binary(f, cam_id);
      write_null_string(f, name);
      write_binary(f, num_points2d);
    }
  }

  // points3D.bin
  {
    std::ofstream f(sparse_dir + "/points3D.bin", std::ios::binary);
    const std::uint64_t count {num_points};
    write_binary(f, count);
    for (std::uint32_t p = 0; p < num_points; ++p)
    {
      const std::uint64_t point_id {p + 1};
      const double x {(static_cast<double>(p % 10) - 4.5) * 0.5};
      const double y {(static_cast<double>(p / 10) - 4.5) * 0.5};
      const double z {0.0};
      const std::uint8_t r {128}, g {128}, b {128};
      const double error {0.5};
      const std::uint64_t track_length {0};

      write_binary(f, point_id);
      write_binary(f, x);
      write_binary(f, y);
      write_binary(f, z);
      write_binary(f, r);
      write_binary(f, g);
      write_binary(f, b);
      write_binary(f, error);
      write_binary(f, track_length);
    }
  }

  // Synthetic training images (solid gray)
  for (std::uint32_t i = 0; i < num_images; ++i)
  {
    const std::string name {"img_" + std::to_string(i) + ".png"};
    std::vector<float> pixels(img_width * img_height * 3, 0.5f);
    IO::save_image_png(
      images_dir + "/" + name, img_width, img_height, pixels.data());
  }
}

} // namespace

TEST(TrainSmokeTest, SyntheticColmapDataLoadsCorrectly)
{
  TempDir tmp;
  const std::string sparse {tmp.path + "/sparse"};
  const std::string images {tmp.path + "/images"};

  create_synthetic_colmap(sparse, images, 1, 2, 100, 64, 64);

  const auto cameras {Colmap::read_cameras_binary(sparse + "/cameras.bin")};
  const auto imgs {Colmap::read_images_binary(sparse + "/images.bin")};
  const auto pts {Colmap::read_points3d_binary(sparse + "/points3D.bin")};

  ASSERT_EQ(cameras.size(), 1u);
  ASSERT_EQ(imgs.size(), 2u);
  ASSERT_EQ(pts.size(), 100u);

  EXPECT_EQ(cameras[0].model, Colmap::CameraModel::pinhole);
  EXPECT_DOUBLE_EQ(cameras[0].params[0], 500.0);
}

TEST(TrainSmokeTest, SyntheticImagesLoadCorrectly)
{
  TempDir tmp;
  const std::string sparse {tmp.path + "/sparse"};
  const std::string images {tmp.path + "/images"};

  create_synthetic_colmap(sparse, images, 1, 1, 10, 32, 32);

  const auto img {IO::load_image(images + "/img_0.png")};
  ASSERT_EQ(img.width, 32u);
  ASSERT_EQ(img.height, 32u);
  ASSERT_EQ(img.pixels.size(), 32u * 32u * 3u);

  // Should be gray (0.5)
  for (std::uint32_t i = 0; i < 10; ++i)
  {
    EXPECT_NEAR(img.pixels[i], 0.5f, 0.01f);
  }
}

} // namespace GoogleUnitTests::Training
