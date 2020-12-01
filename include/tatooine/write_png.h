#ifndef TATOOINE_WRITE_PNG_H
#define TATOOINE_WRITE_PNG_H
//==============================================================================
#include <tatooine/concepts.h>

#include <filesystem>
#include <png++/png.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_number T>
void write_png(std::filesystem::path const& path,
               std::vector<T> const& data, size_t width, size_t height) {
  png::image<png::rgb_pixel> image(width, height);
  for (unsigned int y = 0; y < image.get_height(); ++y) {
    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
      unsigned int idx = x + width * y;
      auto         d   = data[idx];
      if (std::isnan(d)) {
        d = 0;
      } else {
        d = std::max<T>(0, std::min<T>(1, d));
      }
      image[image.get_height() - 1 - y][x].red =
          image[image.get_height() - 1 - y][x].green =
              image[image.get_height() - 1 - y][x].blue = d * 255;
    }
  }
  image.write(path.string());
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
