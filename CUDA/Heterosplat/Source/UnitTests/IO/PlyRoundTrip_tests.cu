#include "IO/PlyReader.h"
#include "IO/PlyWriter.h"

#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace GoogleUnitTests::IO
{

namespace
{

struct TempFile
{
  std::string path;
  TempFile() : path{std::tmpnam(nullptr)} {}
  ~TempFile() { std::remove(path.c_str()); }
};

} // namespace

TEST(PlyRoundTrip, Degree0WriteThenRead)
{
  const std::uint32_t N {3};
  const std::uint32_t degree {0};
  const std::uint32_t K {1};

  const std::vector<float> means {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f};
  const std::vector<float> sh {
    0.1f, 0.2f, 0.3f,
    0.4f, 0.5f, 0.6f,
    0.7f, 0.8f, 0.9f};
  const std::vector<float> opacities {-1.0f, 0.0f, 1.0f};
  const std::vector<float> scales {
    -3.0f, -2.0f, -1.0f,
    0.0f, 1.0f, 2.0f,
    3.0f, 4.0f, 5.0f};
  const std::vector<float> quats {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.5f, 0.5f};

  TempFile tmp;
  ::IO::write_gaussians_ply(
    tmp.path, N, means.data(), degree, sh.data(),
    opacities.data(), scales.data(), quats.data());

  const auto data {::IO::read_gaussians_ply(tmp.path)};

  ASSERT_EQ(data.num_gaussians, N);
  ASSERT_EQ(data.sh_degree, degree);
  ASSERT_EQ(data.means.size(), N * 3);
  ASSERT_EQ(data.sh_coeffs.size(), N * K * 3);
  ASSERT_EQ(data.opacities.size(), N);
  ASSERT_EQ(data.scales.size(), N * 3);
  ASSERT_EQ(data.quats.size(), N * 4);

  for (std::uint32_t i = 0; i < N * 3; ++i)
    EXPECT_FLOAT_EQ(data.means[i], means[i]) << "means[" << i << "]";

  for (std::uint32_t i = 0; i < N * K * 3; ++i)
    EXPECT_FLOAT_EQ(data.sh_coeffs[i], sh[i]) << "sh[" << i << "]";

  for (std::uint32_t i = 0; i < N; ++i)
    EXPECT_FLOAT_EQ(data.opacities[i], opacities[i]);

  for (std::uint32_t i = 0; i < N * 3; ++i)
    EXPECT_FLOAT_EQ(data.scales[i], scales[i]);

  for (std::uint32_t i = 0; i < N * 4; ++i)
    EXPECT_FLOAT_EQ(data.quats[i], quats[i]);
}

TEST(PlyRoundTrip, Degree3WriteThenRead)
{
  const std::uint32_t N {2};
  const std::uint32_t degree {3};
  const std::uint32_t K {16};

  std::vector<float> means(N * 3);
  std::vector<float> sh(N * K * 3);
  std::vector<float> opacities(N);
  std::vector<float> scales(N * 3);
  std::vector<float> quats(N * 4);

  for (std::uint32_t i = 0; i < means.size(); ++i)
    means[i] = static_cast<float>(i) * 0.1f;
  for (std::uint32_t i = 0; i < sh.size(); ++i)
    sh[i] = static_cast<float>(i) * 0.01f;
  for (std::uint32_t i = 0; i < opacities.size(); ++i)
    opacities[i] = static_cast<float>(i) - 0.5f;
  for (std::uint32_t i = 0; i < scales.size(); ++i)
    scales[i] = static_cast<float>(i) * 0.5f - 1.0f;
  for (std::uint32_t i = 0; i < quats.size(); ++i)
    quats[i] = static_cast<float>(i) * 0.25f;

  TempFile tmp;
  ::IO::write_gaussians_ply(
    tmp.path, N, means.data(), degree, sh.data(),
    opacities.data(), scales.data(), quats.data());

  const auto data {::IO::read_gaussians_ply(tmp.path)};

  ASSERT_EQ(data.num_gaussians, N);
  ASSERT_EQ(data.sh_degree, degree);
  ASSERT_EQ(data.sh_coeffs.size(), N * K * 3);

  for (std::uint32_t i = 0; i < N * K * 3; ++i)
    EXPECT_FLOAT_EQ(data.sh_coeffs[i], sh[i])
      << "sh[" << i << "] (n=" << (i / (K * 3))
      << " k=" << ((i % (K * 3)) / 3)
      << " c=" << (i % 3) << ")";
}

} // namespace GoogleUnitTests::IO
