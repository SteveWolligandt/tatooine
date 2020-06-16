#ifndef TATOOINE_LINSPACE_H
#define TATOOINE_LINSPACE_H
//============================================================================
#include <boost/iterator/iterator_facade.hpp>
#include <cstddef>
#include <functional>
#include <ostream>

#include "type_traits.h"
//============================================================================
namespace tatooine {
//============================================================================
// forward declarations
//============================================================================
template <real_number Real>
struct linspace_iterator;

//============================================================================
template <real_number Real>
struct linspace {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t         = linspace<Real>;
  using real_t         = Real;
  using value_type     = Real;
  using iterator       = linspace_iterator<Real>;
  using const_iterator = linspace_iterator<Real>;

  //============================================================================
  // members
  //============================================================================
 private:
  Real   m_min, m_max;
  size_t m_size;

  //============================================================================
  // ctors
  //============================================================================
 public:
  constexpr linspace() noexcept : m_min{Real(0)}, m_max{Real(0)}, m_size{0} {}
  //----------------------------------------------------------------------------
  constexpr linspace(real_number auto min, real_number auto max,
                     size_t size) noexcept
      : m_min{min}, m_max{max}, m_size{size} {}
  //----------------------------------------------------------------------------
  constexpr linspace(linspace const&)     = default;
  constexpr linspace(linspace&&) noexcept = default;
  //----------------------------------------------------------------------------
  template <real_number OtherReal>
  explicit constexpr linspace(linspace<OtherReal> const& other) noexcept
      : m_min{static_cast<Real>(other.front())},
        m_max{static_cast<Real>(other.back())},
        m_size{other.size()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator=(linspace const&) -> linspace& = default;
  constexpr auto operator=(linspace&&) noexcept -> linspace& = default;
  //----------------------------------------------------------------------------
  template <real_number OtherReal>
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
  constexpr auto at(size_t i) const -> Real {
    if (m_size <= 1) { return m_min; }
    return m_min + spacing() * i;
  }
  //----------------------------------------------------------------------------
  constexpr auto operator[](size_t i) const { return at(i); }
  //----------------------------------------------------------------------------
  constexpr auto begin() const { return const_iterator{this, 0}; }
  constexpr auto end() const { return const_iterator{this, m_size}; }
  //----------------------------------------------------------------------------
  constexpr auto size() const { return m_size; }
  constexpr auto front() const { return m_min; }
  constexpr auto back() const { return m_max; }
  //----------------------------------------------------------------------------
  constexpr auto spacing() const { return (m_max - m_min) / (m_size - 1); }
};

//==============================================================================
template <real_number Real>
struct linspace_iterator
    : boost::iterator_facade<linspace_iterator<Real>, Real,
                             std::bidirectional_iterator_tag, Real> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = linspace_iterator<Real>;

  //============================================================================
  // members
  //============================================================================
 private:
  linspace<Real> const* m_lin;
  size_t                m_i;

  //============================================================================
  // ctors
  //============================================================================
 public:
  linspace_iterator() : m_lin{nullptr}, m_i{0} {}
  //----------------------------------------------------------------------------
  linspace_iterator(linspace<Real> const* const _lin, size_t _i)
      : m_lin{_lin}, m_i{_i} {}
  //----------------------------------------------------------------------------
  linspace_iterator(linspace_iterator const&)     = default;
  linspace_iterator(linspace_iterator&&) noexcept = default;
  //============================================================================
  // assign operators
  //============================================================================
  auto operator=(linspace_iterator const& other)
    -> linspace_iterator& = default;
  auto operator=(linspace_iterator&& other) noexcept
    -> linspace_iterator& = default;
  //----------------------------------------------------------------------------
  ~linspace_iterator() = default;

  //============================================================================
  // methods
  //============================================================================
  auto operator<(linspace_iterator const& other) const -> bool {
    return m_i < other.m_i;
  }
  //----------------------------------------------------------------------------
  auto operator<=(linspace_iterator const& other) const -> bool {
    return m_i <= other.m_i;
  }
  //----------------------------------------------------------------------------
  auto operator>(linspace_iterator const& other) const -> bool {
    return m_i > other.m_i;
  }
  //----------------------------------------------------------------------------
  auto operator>=(linspace_iterator const& other) const -> bool {
    return m_i >= other.m_i;
  }
  //----------------------------------------------------------------------------
  auto distance(linspace_iterator const& other) const {
    return m_i - other.m_i;
  }

  //============================================================================
  // iterator_facade implementation
  //============================================================================
 private:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  void increment() { ++m_i; }
  //----------------------------------------------------------------------------
  void decrement() { --m_i; }
  //----------------------------------------------------------------------------
  auto equal(linspace_iterator const& other) const { return m_i == other.m_i; }
  //----------------------------------------------------------------------------
  auto dereference() const { return m_lin->at(m_i); }
};

//==============================================================================
// free functions
//==============================================================================
template <real_number Real>
constexpr auto begin(linspace<Real> const& l) {
  return l.begin();
}
//------------------------------------------------------------------------------
template <real_number Real>
constexpr auto end(linspace<Real> const& l) {
  return l.end();
}
//------------------------------------------------------------------------------
template <real_number Real>
constexpr auto distance(linspace_iterator<Real> const& it0,
                        linspace_iterator<Real> const& it1) {
  return it0.distance(it1);
}
//------------------------------------------------------------------------------
template <real_number Real>
auto size(linspace<Real> const& l) {
  return l.size();
}
//------------------------------------------------------------------------------
template <real_number Real>
auto next(linspace_iterator<Real> const& l, size_t diff = 1) {
  return linspace_iterator<Real>{&l.linspace(), l.m_i + diff};
}
//------------------------------------------------------------------------------
template <real_number Real>
auto prev(linspace_iterator<Real> const& l, size_t diff = 1) {
  return linspace_iterator<Real>{&l.linspace(), l.m_i - diff};
}
//------------------------------------------------------------------------------
template <real_number Real>
inline auto advance(linspace_iterator<Real>& l, long n = 1) -> auto& {
  if (n < 0) {
    while (n++) { --l; }
  } else {
    while (n--) { ++l; }
  }
  return l;
}

//==============================================================================
// deduction guides
//==============================================================================
template <real_number Real0, real_number Real1>
linspace(Real0, Real1, size_t) -> linspace<promote_t<Real0, Real1>>;

//==============================================================================
// I/O
//==============================================================================
template <real_number Real>
auto operator<<(std::ostream& out, linspace<Real> const& l) -> auto& {
  out << "[" << l[0] << ", " << l[1] << ", ... , " << l.back() << "]";
  return out;
}
//============================================================================
}  // namespace tatooine
//============================================================================
#endif
