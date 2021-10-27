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
static constexpr std::size_t hardware_constructive_interference_size = 64;
static constexpr std::size_t hardware_destructive_interference_size  = 64;
#endif

template <typename T, std::size_t N = hardware_destructive_interference_size>
class alignas(N) aligned {
  T value;

 public:
  template <typename... Args>
  explicit aligned(Args&&... args) : value(std::forward<Args>(args)...) {}

  auto operator*() const -> auto const& { return value; }
  auto operator*() -> auto& { return value; }
  auto operator->() const { return &value; }
  auto operator->() { return &value; }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
