#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "IO/ImageIO.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace IO
{

Image load_image(const std::string& path)
{
  int w, h, channels_in_file;
  unsigned char* data {stbi_load(path.c_str(), &w, &h, &channels_in_file, 3)};
  if (data == nullptr)
  {
    throw std::runtime_error{"failed to load image: " + path};
  }

  Image image;
  image.width = static_cast<std::uint32_t>(w);
  image.height = static_cast<std::uint32_t>(h);
  image.pixels.resize(image.width * image.height * 3);

  for (std::uint32_t i = 0; i < image.width * image.height * 3; ++i)
  {
    image.pixels[i] = static_cast<float>(data[i]) / 255.0f;
  }

  stbi_image_free(data);
  return image;
}

void save_image_png(
  const std::string& path,
  const std::uint32_t width,
  const std::uint32_t height,
  const float* pixels)
{
  std::vector<unsigned char> data(width * height * 3);

  for (std::uint32_t i = 0; i < width * height * 3; ++i)
  {
    const float clamped {std::clamp(pixels[i], 0.0f, 1.0f)};
    data[i] = static_cast<unsigned char>(clamped * 255.0f + 0.5f);
  }

  const int result {stbi_write_png(
    path.c_str(),
    static_cast<int>(width),
    static_cast<int>(height),
    3,
    data.data(),
    static_cast<int>(width * 3))};

  if (result == 0)
  {
    throw std::runtime_error{"failed to write PNG: " + path};
  }
}

} // namespace IO
