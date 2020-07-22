#ifndef TATOOINE_SWAP_ENDIANESS_H
#define TATOOINE_SWAP_ENDIANESS_H
//==============================================================================
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Data>
Data swap_endianess(Data data) {
  constexpr auto size = sizeof(Data);
  using mem_t         = unsigned char *;
  auto mem            = reinterpret_cast<mem_t>(&data);

  // swap bytes
  for (size_t i = 0; i < size / 2; i++) std::swap(mem[i], mem[size - 1 - i]);
  return data;
}
//------------------------------------------------------------------------------
template <typename Data>
void swap_endianess(Data *data, size_t n) {
  constexpr size_t size = sizeof(Data);
  using mem_t           = unsigned char *;
  for (size_t i = 0; i < n; ++i) {
    auto mem = reinterpret_cast<mem_t>(&data[i]);
    // swap bytes
    for (size_t j = 0; j < size / 2; j++) std::swap(mem[j], mem[size - 1 - j]);
  }
}
//------------------------------------------------------------------------------
template <typename Data>
void swap_endianess(std::vector<Data> &data) {
  swap_endianess(data.data(), data.size());
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
