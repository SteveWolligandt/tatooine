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
template <typename Real>
struct linspace_iterator;

//============================================================================
template <typename Real>
struct linspace {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t         = linspace<Real>;
  using real_t         = Real;
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
  constexpr linspace() noexcept
      : m_min{0}, m_max{0}, m_size{0} {}
  //----------------------------------------------------------------------------
  constexpr linspace(Real min, Real max, size_t size) noexcept
      : m_min{min}, m_max{max}, m_size{size} {}
  //----------------------------------------------------------------------------
  constexpr linspace(const linspace&) = default;
  constexpr linspace(linspace&&) noexcept = default;
  //----------------------------------------------------------------------------
  template <typename OtherReal>
  explicit constexpr linspace(const linspace<OtherReal>& other) noexcept
      : m_min{static_cast<Real>(other.front())},
        m_max{static_cast<Real>(other.back())},
        m_size{other.size()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator=(const linspace<Real>&) -> linspace& = default;
  constexpr auto operator=(linspace<Real> &&) noexcept -> linspace& = default;
  //----------------------------------------------------------------------------
  ~linspace() = default;
  //----------------------------------------------------------------------------
  template <typename OtherReal>
  constexpr auto operator=(const linspace<OtherReal>& other) noexcept -> auto& {
    m_min        = other.front();
    m_max        = other.back();
    m_size = other.size();
    return *this;
  }

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
  constexpr auto begin() const { return iterator{this, 0}; }
  constexpr auto end() const { return iterator{this, m_size}; }
  //----------------------------------------------------------------------------
  constexpr auto size() const { return m_size; }
  constexpr auto front() const { return m_min; }
  constexpr auto back() const { return m_max; }
  //----------------------------------------------------------------------------
  constexpr auto spacing() const { return (m_max - m_min) / (m_size - 1); }
}; 

//==============================================================================
template <typename Real>
struct linspace_iterator
    : boost::iterator_facade<linspace_iterator<Real>, Real,
                             boost::bidirectional_traversal_tag, Real> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = linspace_iterator<Real>;

  //============================================================================
  // members
  //============================================================================
 private:
  const linspace<Real>* m_lin;
  size_t             m_i;

  //============================================================================
  // ctors
  //============================================================================
 public:
  linspace_iterator(const linspace<Real>* _lin, size_t _i)
      : m_lin{_lin}, m_i{_i} {}
  //----------------------------------------------------------------------------
  linspace_iterator(const linspace_iterator&)     = default;
  linspace_iterator(linspace_iterator&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const linspace_iterator& other)
      -> linspace_iterator& = default;
  auto operator=(linspace_iterator&& other) noexcept
      -> linspace_iterator& = default;
  //----------------------------------------------------------------------------
  ~linspace_iterator() = default;

  //============================================================================
  // methods
  //============================================================================
  constexpr void to_begin() { m_i = 0; }
  //----------------------------------------------------------------------------
  constexpr void to_end() { m_i = m_lin->size(); }
  //----------------------------------------------------------------------------
  auto& i() { return m_i; }
  auto  i() const { return m_i; }
  //----------------------------------------------------------------------------
  auto begin() const { return m_lin->begin(); }
  auto end() const { return m_lin->end(); }
  //----------------------------------------------------------------------------
  auto operator<(const linspace_iterator& other) const -> bool {
    return i() < other.i();
  }
  //----------------------------------------------------------------------------
  auto operator<=(const linspace_iterator& other) const -> bool {
    return i() <= other.i();
  }
  //----------------------------------------------------------------------------
  auto operator>(const linspace_iterator& other) const -> bool {
    return i() > other.i();
  }
  //----------------------------------------------------------------------------
  auto operator>=(const linspace_iterator& other) const -> bool {
    return i() >= other.i();
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
  auto equal(const linspace_iterator& other) const { return m_i == other.m_i; }
  //----------------------------------------------------------------------------
  auto dereference() const { return m_lin->at(m_i); }
};

//==============================================================================
// free functions
//==============================================================================
template <typename Real>
constexpr auto begin(const linspace<Real>& l) {
  return l.begin();
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto end(const linspace<Real>& l) {
  return l.end();
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto distance(const linspace_iterator<Real>& it0,
                        const linspace_iterator<Real>& it1) -> long {
  return it1.i() - it0.i();
}
//------------------------------------------------------------------------------
template <typename Real>
auto size(const linspace<Real>& l) {
  return l.size();
}
//------------------------------------------------------------------------------
template <typename Real>
auto next(const linspace_iterator<Real>& l, size_t diff = 1) {
  return linspace_iterator<Real>{&l.linspace(), l.i() + diff};
}
//------------------------------------------------------------------------------
template <typename Real>
auto prev(const linspace_iterator<Real>& l, size_t diff = 1) {
  return linspace_iterator<Real>{&l.linspace(), l.i() - diff};
}
//------------------------------------------------------------------------------
template <typename Real>
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
template <typename Real>
linspace(Real, Real, size_t)->linspace<Real>;

//==============================================================================
// I/O
//==============================================================================
template <typename Real>
auto operator<<(std::ostream& out, const linspace<Real>& l) -> auto& {
  out << "[" << l[0] << ", " << l[1] << ", ... , " << l.back() << "]";
  return out;
}
//============================================================================
}  // namespace tatooine
//============================================================================

#endif
