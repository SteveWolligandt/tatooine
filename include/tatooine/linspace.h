#ifndef TATOOINE_LINSPACE_H
#define TATOOINE_LINSPACE_H
//============================================================================
#include <tatooine/concepts.h>
#include <tatooine/iterator_facade.h>
#include <tatooine/reflection.h>
#include <tatooine/type_traits.h>

#include <cstddef>
#include <functional>
#include <ostream>
//============================================================================
namespace tatooine::detail::linspace {
//============================================================================
template <floating_point Real>
struct iterator;
//============================================================================
}  // namespace tatooine::detail::linspace
//============================================================================
namespace tatooine {
//============================================================================
template <floating_point Real>
struct linspace {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t     = linspace<Real>;
  using real_t     = Real;
  using value_type = real_t;
  using iterator = detail::linspace::iterator<Real>;

  //============================================================================
  // members
  //============================================================================
 private:
  Real        m_min, m_max;
  std::size_t m_size;

  //============================================================================
  // ctors
  //============================================================================
 public:
  constexpr linspace() noexcept : m_min{Real(0)}, m_max{Real(0)}, m_size{0} {}
  //----------------------------------------------------------------------------
  constexpr linspace(arithmetic auto const min, arithmetic auto const max,
                     std::size_t const size) noexcept
      : m_min{static_cast<Real>(std::min<Real>(min, max))},
        m_max{static_cast<Real>(std::max<Real>(min, max))},
        m_size{size} {}
  //----------------------------------------------------------------------------
  constexpr linspace(linspace const&)     = default;
  constexpr linspace(linspace&&) noexcept = default;
  //----------------------------------------------------------------------------
  template <floating_point OtherReal>
  explicit constexpr linspace(linspace<OtherReal> const& other) noexcept
      : m_min{static_cast<Real>(other.front())},
        m_max{static_cast<Real>(other.back())},
        m_size{other.size()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator=(linspace const&) -> linspace& = default;
  constexpr auto operator=(linspace&&) noexcept -> linspace& = default;
  //----------------------------------------------------------------------------
  template <floating_point OtherReal>
  constexpr auto operator=(linspace<OtherReal> const& other) noexcept -> auto& {
    m_min  = other.front();
    m_max  = other.back();
    m_size = other.size();
    return *this;
  }
  //----------------------------------------------------------------------------
  ~linspace() = default;

  //============================================================================
  // methods
  //============================================================================
  constexpr auto at(std::size_t i) const -> Real {
    if (m_size <= 1) {
      return m_min;
    }
    return m_min + spacing() * i;
  }
  //----------------------------------------------------------------------------
  constexpr auto operator[](std::size_t i) const { return at(i); }
  //----------------------------------------------------------------------------
  constexpr auto        begin() const { return iterator{this, 0}; }
  static constexpr auto end() { return typename iterator::sentinel_type{}; }
  //----------------------------------------------------------------------------
  constexpr auto size() const { return m_size; }
  constexpr auto size() -> auto& { return m_size; }
  constexpr auto front() const { return m_min; }
  constexpr auto front() -> auto& { return m_min; }
  constexpr auto back() const { return m_max; }
  constexpr auto back() -> auto& { return m_max; }
  //----------------------------------------------------------------------------
  constexpr auto spacing() const { return (m_max - m_min) / (m_size - 1); }
  //----------------------------------------------------------------------------
  constexpr auto push_back() {
    m_max += spacing();
    ++m_size;
  }
  //----------------------------------------------------------------------------
  constexpr auto pop_back() {
    m_max -= spacing();
    --m_size;
  }
  //----------------------------------------------------------------------------
  constexpr auto push_front() {
    m_min -= spacing();
    ++m_size;
  }
  //----------------------------------------------------------------------------
  constexpr auto pop_front() {
    m_min += spacing();
    --m_size;
  }
  //----------------------------------------------------------------------------
  constexpr auto resize(std::size_t const s) { m_size = s; }
};
//==============================================================================
// free functions
//==============================================================================
template <floating_point Real>
constexpr auto begin(linspace<Real> const& l) {
  return l.begin();
}
//------------------------------------------------------------------------------
template <floating_point Real>
constexpr auto end(linspace<Real> const& l) {
  return l.end();
}
//------------------------------------------------------------------------------
template <floating_point Real>
auto size(linspace<Real> const& l) {
  return l.size();
}
//------------------------------------------------------------------------------
template <floating_point Real>
auto front(linspace<Real> const& l) {
  return l.front();
}
//------------------------------------------------------------------------------
template <floating_point Real>
auto front(linspace<Real>& l) -> auto& {
  return l.front();
}
//------------------------------------------------------------------------------
template <floating_point Real>
auto back(linspace<Real> const& l) {
  return l.back();
}
//------------------------------------------------------------------------------
template <floating_point Real>
auto back(linspace<Real>& l) -> auto& {
  return l.back();
}
//==============================================================================
// deduction guides
//==============================================================================
template <arithmetic Real0, arithmetic Real1>
linspace(Real0 const, Real1 const, std::size_t const)
    -> linspace<common_type<Real0, Real1>>;
//==============================================================================
// type traits
//==============================================================================
template <typename T>
struct is_linspace_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
struct is_linspace_impl<linspace<Real>> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_linspace = is_linspace_impl<T>::value;
//==============================================================================
// I/O
//==============================================================================
template <floating_point Real>
auto operator<<(std::ostream& out, linspace<Real> const& l) -> auto& {
  out << "[" << l[0] << ", " << l[1] << ", ..., " << l.back() << "]";
  return out;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::reflection {
//==============================================================================
template <typename T>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    linspace<T>, TATOOINE_REFLECTION_INSERT_METHOD(front, front()),
    TATOOINE_REFLECTION_INSERT_METHOD(back, back()),
    TATOOINE_REFLECTION_INSERT_METHOD(size, size()))
//==============================================================================
}  // namespace tatooine::reflection
//==============================================================================
namespace tatooine::detail::linspace {
//==============================================================================
template <floating_point Real>
struct iterator : iterator_facade<iterator<Real>> {
  struct sentinel_type {};
  using linspace_type = tatooine::linspace<Real>;
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = iterator<Real>;

  //============================================================================
  // members
  //============================================================================
 private:
  linspace_type const* m_lin = nullptr;
  std::size_t          m_i{};

  //============================================================================
  // ctors
  //============================================================================
 public:
  constexpr iterator() = default;
  //----------------------------------------------------------------------------
  constexpr iterator(linspace_type const* const _lin, std::size_t _i)
      : m_lin{_lin}, m_i{_i} {}
  //----------------------------------------------------------------------------
  constexpr iterator(iterator const&)     = default;
  constexpr iterator(iterator&&) noexcept = default;
  //============================================================================
  // assign operators
  //============================================================================
  constexpr auto operator=(iterator const& other) -> iterator& = default;
  constexpr auto operator=(iterator&& other) noexcept -> iterator& = default;
  //----------------------------------------------------------------------------
  ~iterator() = default;
  //============================================================================
  // iterator_facade implementation
  //============================================================================
  constexpr auto equal(iterator const& other) const { return m_i == other.m_i; }
  constexpr auto dereference() const { return m_lin->at(m_i); }
  constexpr auto increment() { ++m_i; }
  constexpr auto decrement() { --m_i; }
  constexpr auto at_end() const { return m_i == m_lin->size(); }
};
//------------------------------------------------------------------------------
template <floating_point Real>
constexpr auto distance(iterator<Real> const& it0, iterator<Real> const& it1) {
  return it0.distance(it1);
}
//------------------------------------------------------------------------------
template <floating_point Real>
auto next(iterator<Real> const& l, std::size_t diff = 1) {
  iterator<Real> it{l};
  it.increment(diff);
  return it;
}
//------------------------------------------------------------------------------
template <floating_point Real>
auto prev(iterator<Real> const& l, std::size_t diff = 1) {
  iterator<Real> it{l};
  it.decrement(diff);
  return it;
}
//------------------------------------------------------------------------------
template <floating_point Real>
inline auto advance(iterator<Real>& l, long n = 1) -> auto& {
  if (n < 0) {
    while (n++) {
      --l;
    }
  } else {
    while (n--) {
      ++l;
    }
  }
  return l;
}
//==============================================================================
}  // namespace tatooine::detail::linspace
//============================================================================
#endif
