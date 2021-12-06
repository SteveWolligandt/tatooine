#ifndef TATOOINE_HANDLE_H
#define TATOOINE_HANDLE_H
//==============================================================================
#include <tatooine/concepts.h>

#include <cstdint>
#include <limits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Child>
struct handle {
  static constexpr std::size_t invalid_idx =
      std::numeric_limits<std::size_t>::max();
  //============================================================================
  std::size_t i{};
  //============================================================================
  handle() : i{invalid_idx} {}
  explicit handle(integral auto _i) : i{static_cast<std::size_t>(_i)} {}
  handle(handle const&)     = default;
  handle(handle&&) noexcept = default;
  ~handle()                 = default;

  auto operator=(handle const&) -> handle& = default;
  template <typename Int, enable_if_integral<Int> = true>
  auto operator=(Int const i) -> handle& {
    this->i = i;
  }
  auto operator=(handle&&) noexcept -> handle& = default;
  auto operator==(handle<Child> const other) const {
    return this->i == other.i;
  }
  auto operator!=(handle<Child> const other) const {
    return this->i != other.i;
  }
  auto operator<(handle<Child> const other) const { return this->i < other.i; }
  auto operator<=(handle<Child> const other) const {
    return this->i <= other.i;
  }
  auto operator>(handle<Child> const other) const { return this->i > other.i; }
  auto operator>=(handle<Child> const other) const {
    return this->i >= other.i;
  }
  static constexpr auto invalid() {
    return handle<Child>{handle<handle<Child>>::invalid_idx};
  }
  auto operator=(integral auto i_) -> handle& {
    i = i_;
    return *this;
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
  template <typename Int, enable_if_integral<Int> = true>
  auto operator+=(Int i) {
    this->i += i;
  }
  template <typename Int, enable_if_integral<Int> = true>
  auto operator-=(Int i) {
    this->i -= i;
  }
  template <typename Int, enable_if_integral<Int> = true>
  auto operator*=(Int i) {
    this->i *= i;
  }
  template <typename Int, enable_if_integral<Int> = true>
  auto operator/=(Int i) {
    this->i /= i;
  }
};
//==============================================================================
template <typename Child, typename Int, enable_if_integral<Int> = true>
auto operator+(handle<Child> const h, Int const i) {
  return Child{h.i + i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, typename Int, enable_if_integral<Int> = true>
auto operator+(Int const i, handle<Child> const h) {
  return Child{i + h.i};
}
//------------------------------------------------------------------------------
template <typename Child, typename Int, enable_if_integral<Int> = true>
auto operator-(handle<Child> const h, Int const i) {
  return Child{h.i - i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, typename Int, enable_if_integral<Int> = true>
auto operator-(Int const i, handle<Child> const h) {
  return Child{i - h.i};
}
//------------------------------------------------------------------------------
template <typename Child, typename Int, enable_if_integral<Int> = true>
auto operator*(handle<Child> const h, Int const i) {
  return Child{h.i * i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, typename Int, enable_if_integral<Int> = true>
auto operator*(Int const i, handle<Child> const h) {
  return Child{i * h.i};
}
//------------------------------------------------------------------------------
template <typename Child, typename Int, enable_if_integral<Int> = true>
auto operator/(handle<Child> const h, Int const i) {
  return Child{h.i / i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, typename Int, enable_if_integral<Int> = true>
auto operator/(Int const i, handle<Child> const h) {
  return Child{i / h.i};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
