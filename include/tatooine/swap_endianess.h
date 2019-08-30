#ifndef __TATOOINE_SWAP_ENDIANESS_H__
#define __TATOOINE_SWAP_ENDIANESS_H__

#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================

template <typename data_t>
data_t swap_endianess(data_t data) {
  constexpr auto size = sizeof(data_t);
  using mem_t         = unsigned char *;
  auto mem            = reinterpret_cast<mem_t>(&data);

  // swap bytes
  for (size_t i = 0; i < size / 2; i++) std::swap(mem[i], mem[size - 1 - i]);
  return data;
}

//------------------------------------------------------------------------------

template <typename data_t>
void swap_endianess(data_t *data, size_t n) {
  constexpr size_t size = sizeof(data_t);
  using mem_t           = unsigned char *;
  for (size_t i = 0; i < n; ++i) {
    auto mem = reinterpret_cast<mem_t>(&data[i]);
    // swap bytes
    for (size_t j = 0; j < size / 2; j++) std::swap(mem[j], mem[size - 1 - j]);
  }
}

//------------------------------------------------------------------------------

template <typename data_t>
void swap_endianess(std::vector<data_t> &data) {
  swap_endianess(data.data(), data.size());
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
