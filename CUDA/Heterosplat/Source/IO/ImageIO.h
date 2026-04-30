#ifndef IO_IMAGE_IO_H
#define IO_IMAGE_IO_H

#include <cstdint>
#include <string>
#include <vector>

namespace IO
{

struct Image
{
  std::vector<float> pixels; // [height * width * 3], RGB, normalized to [0,1]
  std::uint32_t width;
  std::uint32_t height;
};

/// Load an image (PNG, JPG, BMP, etc.) as RGB float32 in [0,1].
/// Always forces 3-channel output regardless of source format.
Image load_image(const std::string& path);

/// Save an RGB float32 image as PNG. Pixels are in [0,1], clamped to [0,255].
void save_image_png(
  const std::string& path,
  std::uint32_t width,
  std::uint32_t height,
  const float* pixels);

} // namespace IO

#endif // IO_IMAGE_IO_H
