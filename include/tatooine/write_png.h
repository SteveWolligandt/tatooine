#ifndef TATOOINE_WRITE_PNG_H
#define TATOOINE_WRITE_PNG_H
//==============================================================================
#include <tatooine/concepts.h>

#include <tatooine/filesystem.h>
#include <tatooine/png.h>
//==============================================================================
namespace tatooine {
//==============================================================================
#if TATOOINE_PNG_AVAILABLE
template <arithmetic T>
void write_png(filesystem::path const& path,
               std::vector<T> const& data, std::size_t width, std::size_t height) {
  auto image = png::image<png::rgb_pixel>{static_cast<png::uint_32>(width),
                                          static_cast<png::uint_32>(height)};
  for (png::uint_32 y = 0; y < image.get_height(); ++y) {
    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
      auto idx = x + static_cast<png::uint_32>(width) * y;
      auto         d   = data[idx];
      if (std::isnan(d)) {
        d = 0;
      } else {
        d = std::clamp(d, T(0), T(1));
      }
      image[image.get_height() - 1 - y][x].red =
          image[image.get_height() - 1 - y][x].green =
              image[image.get_height() - 1 - y][x].blue = d * 255;
    }
  }
  image.write(path.string());
}
#endif
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
