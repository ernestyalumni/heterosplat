#ifndef UNIT_TESTS_ORACLE_FIXTURE_H
#define UNIT_TESTS_ORACLE_FIXTURE_H

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GoogleUnitTests
{
namespace OracleFixture
{

//------------------------------------------------------------------------------
/// Build a path under `Fixtures/<group>/<filename>`. The compile-time define
/// `HETEROSPLAT_FIXTURE_DIR` is set by `UnitTests/CMakeLists.txt` to the
/// absolute path of `Source/UnitTests/Fixtures` so tests find their data
/// regardless of the build directory.
//------------------------------------------------------------------------------
inline std::string fixture_path(
  const std::string& group,
  const std::string& filename)
{
#ifndef HETEROSPLAT_FIXTURE_DIR
#error "HETEROSPLAT_FIXTURE_DIR must be defined by CMake"
#endif
  return std::string{HETEROSPLAT_FIXTURE_DIR} + "/" + group + "/" + filename;
}

//------------------------------------------------------------------------------
/// Read raw little-endian float32 array from disk. Element count is inferred
/// from file size; caller asserts shape against expected `N * stride`.
//------------------------------------------------------------------------------
inline std::vector<float> load_floats(const std::string& path)
{
  std::ifstream input{path, std::ios::binary | std::ios::ate};
  if (!input)
  {
    throw std::runtime_error{"OracleFixture: cannot open " + path};
  }
  const std::streamsize size_bytes {input.tellg()};
  if (size_bytes < 0 || (size_bytes % static_cast<std::streamsize>(sizeof(float))) != 0)
  {
    throw std::runtime_error{
      "OracleFixture: invalid float32 file size: " + path};
  }
  input.seekg(0, std::ios::beg);

  const std::size_t count {
    static_cast<std::size_t>(size_bytes) / sizeof(float)};
  std::vector<float> result(count);
  if (!input.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(count * sizeof(float))))
  {
    throw std::runtime_error{"OracleFixture: read failed: " + path};
  }
  return result;
}

//------------------------------------------------------------------------------
/// Read a single little-endian uint32 (used for shape scalars N, K, degree).
//------------------------------------------------------------------------------
inline std::uint32_t load_uint32(const std::string& path)
{
  std::ifstream input{path, std::ios::binary};
  if (!input)
  {
    throw std::runtime_error{"OracleFixture: cannot open " + path};
  }
  std::uint32_t value {0};
  if (!input.read(
        reinterpret_cast<char*>(&value),
        static_cast<std::streamsize>(sizeof(value))))
  {
    throw std::runtime_error{"OracleFixture: uint32 read failed: " + path};
  }
  return value;
}

} // namespace OracleFixture
} // namespace GoogleUnitTests

#endif // UNIT_TESTS_ORACLE_FIXTURE_H
