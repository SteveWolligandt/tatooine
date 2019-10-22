#ifndef TATOOINE_CUDA_TEST_WRITE_PPM_H
#define TATOOINE_CUDA_TEST_WRITE_PPM_H

#include <fstream>
#include <vector>

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================
inline void write_ppm(const std::string&        filename,
                      const std::vector<float>& transformed, const size_t width,
                      const size_t height, const unsigned short c) {
  std::ofstream file{filename};
  if (file.is_open()) {
    file << "P3\n" << width << ' ' << height << "\n255\n";
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        const size_t i = x + width * (height - 1 - y);
        file << static_cast<unsigned int>(transformed[i * c] * 255) << ' '
             << static_cast<unsigned int>(transformed[i * c + 1] * 255) << ' '
             << static_cast<unsigned int>(transformed[i * c + 2] * 255) << ' ';
      }
      file << '\n';
    }
  }
}
//==============================================================================
}  // namespace test
}  // namespace gpu
}  // namespace tatooine
//==============================================================================

#endif
