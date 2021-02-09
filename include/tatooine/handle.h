#ifndef TATOOINE_HANDLE_H
#define TATOOINE_HANDLE_H
//============================================================================
#include <cstdint>
#include <tatooine/concepts.h>
//============================================================================
namespace tatooine {
//============================================================================
struct handle {
  static constexpr std::size_t invalid_idx =
      std::numeric_limits<std::size_t>::max();
  //==========================================================================
  std::size_t i;
  //==========================================================================
  handle() : i{invalid_idx} {}
#ifdef __cpp_concepts
  template <integral Int>
#else
  template <typename Int, enable_if<is_integral<Int>> = true>
#endif
  explicit handle(Int _i) : i{static_cast<std::size_t>(_i)} {}
  handle(const handle&)                    = default;
  handle(handle&&)                         = default;
  auto operator=(const handle&) -> handle& = default;
  auto operator=(handle &&)     -> handle& = default;
#ifdef __cpp_concepts
  template <integral Int>
#else
  template <typename Int, enable_if<is_integral<Int>> = true>
#endif
  auto operator=(Int i_) -> handle& {
    i = i_;
    return *this;
  }

  //==========================================================================
  auto& operator++() {
    ++this->i;
    return *this;
  }
  //--------------------------------------------------------------------------
  auto& operator--() {
    --this->i;
    return *this;
  }
  //--------------------------------------------------------------------------
  auto& operator=(std::size_t i) {
    this->i = i;
    return *this;
  }
};
//============================================================================
}  // namespace tatooine
//============================================================================
#endif
