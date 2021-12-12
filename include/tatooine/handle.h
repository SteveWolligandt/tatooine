#ifndef TATOOINE_HANDLE_H
#define TATOOINE_HANDLE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/demangling.h>

#include <cstdint>
#include <ostream>
#include <limits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Child, unsigned_integral Int = std::size_t>
struct handle {
  using int_t                       = Int;
  static constexpr auto invalid_idx = std::numeric_limits<Int>::max();
  //============================================================================
 private:
  Int i{};
  //============================================================================
 public:
  constexpr handle() : i{invalid_idx} {}
  constexpr explicit handle(integral auto const _i) : i{static_cast<Int>(_i)} {}
  constexpr handle(handle const&)     = default;
  constexpr handle(handle&&) noexcept = default;
  ~handle()                           = default;

  constexpr auto operator=(handle const&) -> handle& = default;
  constexpr auto operator=(integral auto const i) -> handle& {
    this->i = i;
    return *this;
  }
  constexpr auto operator=(handle&&) noexcept -> handle& = default;
  constexpr auto operator==(integral auto const other) const {
    return this->i == static_cast<Int>(other);
  }
  constexpr auto operator==(handle<Child, Int> const other) const {
    return this->i == other.i;
  }
  constexpr auto operator!=(handle<Child, Int> const other) const {
    return this->i != other.i;
  }
  constexpr auto operator<(handle<Child, Int> const other) const {
    return this->i < other.i;
  }
  constexpr auto operator<=(handle<Child, Int> const other) const {
    return this->i <= other.i;
  }
  constexpr auto operator>(handle<Child, Int> const other) const {
    return this->i > other.i;
  }
  constexpr auto operator>=(handle<Child, Int> const other) const {
    return this->i >= other.i;
  }
  static constexpr auto invalid() {
    return handle<Child, Int>{handle<handle<Child, Int>>::invalid_idx};
  }
  //============================================================================
  constexpr auto operator++() -> auto& {
    ++this->i;
    return *static_cast<Child*>(this);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator++(int /*i*/) {
    auto h = Child{i};
    ++i;
    return h;
  }
  //----------------------------------------------------------------------------
  constexpr auto operator--() -> auto& {
    --i;
    return *static_cast<Child*>(this);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator--(int /*i*/) {
    auto h = Child{i};
    --i;
    return h;
  }
  constexpr auto operator+=(integral auto const j) { i += j; }
  constexpr auto operator-=(integral auto const j) { i -= j; }
  constexpr auto operator*=(integral auto const j) { i *= j; }
  constexpr auto operator/=(integral auto const j) { i /= j; }

  auto index() const { return i; }
};
//==============================================================================
template <typename Child, unsigned_integral Int>
auto operator+(handle<Child, Int> const h, integral auto const i) {
  return Child{h.index() + i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto operator+(integral auto const i, handle<Child, Int> const h) {
  return Child{i + h.index()};
}
//------------------------------------------------------------------------------
template <typename Child, unsigned_integral Int>
auto operator-(handle<Child, Int> const h, integral auto const i) {
  return Child{h.index() - i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto operator-(integral auto const i, handle<Child, Int> const h) {
  return Child{i - h.index()};
}
//------------------------------------------------------------------------------
template <typename Child, unsigned_integral Int>
auto operator*(handle<Child, Int> const h, integral auto const i) {
  return Child{h.index() * i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto operator*(integral auto const i, handle<Child, Int> const h) {
  return Child{i * h.index()};
}
//------------------------------------------------------------------------------
template <typename Child, unsigned_integral Int>
auto operator/(handle<Child, Int> const h, integral auto const i) {
  return Child{h.index() / i};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto operator/(integral auto const i, handle<Child, Int> const h) {
  return Child{i / h.index()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto operator<<(std::ostream & stream, handle<Child, Int> const h) -> auto& {
  return stream << type_name<Child>() << "[" << h.index() << "]";
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Child, unsigned_integral Int>
auto constexpr index(handle<Child, Int> const h) {
  return h.index();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
