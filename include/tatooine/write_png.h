#ifndef TATOOINE_WRITE_PNG_H
#define TATOOINE_WRITE_PNG_H
#include <png++/png.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <size_t NumChannels, typename T>
void write_png(const std::string& filepath, std::vector<T> data, size_t width,
               size_t height) {
  png::image<png::rgb_pixel> image(width, height);
  for (unsigned int y = 0; y < image.get_height(); ++y) {
    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
      unsigned int idx = x + width * y;

      image[image.get_height() - 1 - y][x].red =
          std::max<T>(0, std::min<T>(1, data[idx])) * 255;
      image[image.get_height() - 1 - y][x].green =
          std::max<T>(0, std::min<T>(1, data[idx])) * 255;
      image[image.get_height() - 1 - y][x].blue =
          std::max<T>(0, std::min<T>(1, data[idx])) * 255;
    }
  }
  image.write(filepath);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
