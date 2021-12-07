#ifndef TATOOINE_HANDLE_H
#define TATOOINE_HANDLE_H
//==============================================================================
#include <tatooine/concepts.h>

#include <cstdint>
#include <limits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Child, unsigned_integral Int = std::size_t>
struct handle {
  using int_t = Int;
  static constexpr auto invalid_idx = std::numeric_limits<Int>::max();
  //============================================================================
  Int i{};
  //============================================================================
  handle() : i{invalid_idx} {}
  explicit handle(integral auto const _i) : i{static_cast<Int>(_i)} {}
  handle(handle const&)     = default;
  handle(handle&&) noexcept = default;
  ~handle()                 = default;

  auto operator=(handle const&) -> handle& = default;
  auto operator=(integral auto const i) -> handle& {
    this->i = i;
    return *this;
  }
  auto operator=(handle&&) noexcept -> handle& = default;
  auto operator==(handle<Child, Int> const other) const {
    return this->i == other.i;
  }
  auto operator!=(handle<Child, Int> const other) const {
    return this->i != other.i;
  }
  auto operator<(handle<Child, Int> const other) const {
    return this->i < other.i;
  }
  auto operator<=(handle<Child, Int> const other) const {
    return this->i <= other.i;
  }
  auto operator>(handle<Child, Int> const other) const {
    return this->i > other.i;
  }
  auto operator>=(handle<Child, Int> const other) const {
    return this->i >= other.i;
  }
  static constexpr auto invalid() {
    return handle<Child, Int>{handle<handle<Child, Int>>::invalid_idx};
  }
  //============================================================================
  auto operator++() -> auto& {
    ++this->i;
    return *static_cast<Child*>(this);
  }
  //----------------------------------------------------------------------------
  auto operator++(int /*i*/) {
    auto h = Child{i};
    ++i;
    return h;
  }
  //----------------------------------------------------------------------------
  auto operator--() -> auto& {
    --i;
    return *static_cast<Child*>(this);
  }
  //----------------------------------------------------------------------------
  auto operator--(int /*i*/) {
    auto h = Child{i};
    --i;
    return h;
  }
  auto operator+=(integral auto const i) { this->i += i; }
  auto operator-=(integral auto const i) { this->i -= i; }
  auto operator*=(integral auto const i) { this->i *= i; }
  auto operator/=(integral auto const i) { this->i /= i; }
};
//==============================================================================
template <typename Child, unsigned_integral Int>
auto operator+(handle<Child, Int> const h, integral auto const i) {
  return Child{h.i + i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto operator+(integral auto const i, handle<Child, Int> const h) {
  return Child{i + h.i};
}
//------------------------------------------------------------------------------
template <typename Child, unsigned_integral Int>
auto operator-(handle<Child, Int> const h, integral auto const i) {
  return Child{h.i - i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto operator-(integral auto const i, handle<Child, Int> const h) {
  return Child{i - h.i};
}
//------------------------------------------------------------------------------
template <typename Child, unsigned_integral Int>
auto operator*(handle<Child, Int> const h, integral auto const i) {
  return Child{h.i * i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto operator*(integral auto const i, handle<Child, Int> const h) {
  return Child{i * h.i};
}
//------------------------------------------------------------------------------
template <typename Child, unsigned_integral Int>
auto operator/(handle<Child, Int> const h, integral auto const i) {
  return Child{h.i / i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto operator/(integral auto const i, handle<Child, Int> const h) {
  return Child{i / h.i};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
