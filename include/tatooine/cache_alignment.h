#ifndef TATOOINE_CACHE_ALIGNMENT_H
#define TATOOINE_CACHE_ALIGNMENT_H
//==============================================================================
#include <cstdint>
#include <new>
//==============================================================================
namespace tatooine {
//==============================================================================
#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
// 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │
// ...
constexpr std::size_t hardware_constructive_interference_size = 64;
constexpr std::size_t hardware_destructive_interference_size  = 64;
#endif

template <typename T, std::size_t N = hardware_destructive_interference_size>
struct alignas(N) aligned {
  T value;
  template <typename... Args>
  explicit aligned(Args&&... args) : value(std::forward<Args>(args)...) {}
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
