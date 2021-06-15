#ifndef TATOOINE_HANDLE_H
#define TATOOINE_HANDLE_H
//============================================================================
#include <tatooine/concepts.h>

#include <cstdint>
#include <limits>
//============================================================================
namespace tatooine {
//============================================================================
template <typename Child>
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
  explicit handle(Int _i) : i{static_cast<std::size_t>(_i)} {
  }
  handle(handle const&) = default;
  handle(handle&&)      = default;
  auto operator=(handle const&) -> handle& = default;
  auto operator=(handle&&) -> handle& = default;
  bool operator==(handle<Child> other) const { return this->i == other.i; }
  bool operator!=(handle<Child> other) const { return this->i != other.i; }
  bool operator<(handle<Child> other) const { return this->i < other.i; }
  static constexpr auto invalid() {
    return handle<Child>{handle<handle<Child>>::invalid_idx};
  }
#ifdef __cpp_concepts
  template <integral Int>
#else
  template <typename Int, enable_if_integral<Int> = true>
#endif
  auto operator=(Int i_) -> handle& {
    i = i_;
    return *this;
  }
  //==========================================================================
  auto operator++() -> auto& {
    ++this->i;
    return *static_cast<Child*>(this);
  }
  //--------------------------------------------------------------------------
  auto operator++(int)  {
    auto const h = Child{i};
    ++i;
    return h;
  }
  //--------------------------------------------------------------------------
  auto operator--() -> auto& {
    --i;
    return *static_cast<Child*>(this);
  }
  //--------------------------------------------------------------------------
  auto operator--(int)  {
    auto const h = Child{i};
    --i;
    return h;
  }
};
//============================================================================
}  // namespace tatooine
//============================================================================
#endif
