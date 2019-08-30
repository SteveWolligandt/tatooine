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

template <typename real_t>
struct linspace_iterator;

//============================================================================
template <typename Real>
struct linspace {
  //============================================================================
  using real_t         = Real;
  using iterator       = linspace_iterator<real_t>;
  using const_iterator = linspace_iterator<real_t>;

  //============================================================================
 private:
  real_t      m_min, m_max;
  std::size_t m_size;

  //============================================================================
 public:
  constexpr linspace(real_t min, real_t max, std::size_t size) noexcept
      : m_min{min}, m_max{max}, m_size{size} {}

  //----------------------------------------------------------------------------
  constexpr linspace(const linspace& other) noexcept
      : m_min{other.m_min}, m_max{other.m_max}, m_size{other.m_size} {}

  //----------------------------------------------------------------------------
  template <typename other_real_t>
  constexpr linspace(const linspace<other_real_t>& other) noexcept
      : m_min{static_cast<real_t>(other.m_min)},
        m_max{static_cast<real_t>(other.m_max)},
        m_size{other.m_size} {}

  //----------------------------------------------------------------------------
  template <typename other_real_t>
  constexpr auto& operator=(const linspace<other_real_t>& other) noexcept {
    m_min        = other.m_min;
    m_max        = other.m_max;
    m_size = other.m_size;
    return *this;
  }

  //============================================================================
  constexpr auto at(std::size_t i) const {
    if (m_size <= 1) { return m_min; }
    return m_min + offset() * i;
  }
  constexpr auto operator[](std::size_t i) const { return at(i); }

  //----------------------------------------------------------------------------
  constexpr auto begin() const { return iterator{this, 0}; }
  constexpr auto end() const { return iterator{this, m_size}; }

  //----------------------------------------------------------------------------
  constexpr auto size() const { return m_size; }
  constexpr auto front() const { return m_min; }
  constexpr auto back() const { return m_max; }
  
  //----------------------------------------------------------------------------
  constexpr auto offset() const { return (m_max - m_min) / (m_size - 1); }
};  // class linspace

//==============================================================================
template <typename real_t>
linspace(real_t, real_t, std::size_t)->linspace<real_t>;

//------------------------------------------------------------------------------
template <typename real_t>
constexpr auto begin(const linspace<real_t>& l) {
  return l.begin();
}

//------------------------------------------------------------------------------
template <typename real_t>
constexpr auto end(const linspace<real_t>& l) {
  return l.end();
}

//------------------------------------------------------------------------------
template <typename real_t>
constexpr long distance(const linspace_iterator<real_t>& it0,
                        const linspace_iterator<real_t>& it1) {
  return it1.i() - it0.i();
}

//==============================================================================
template <typename real_t>
struct linspace_iterator
    : boost::iterator_facade<linspace_iterator<real_t>, real_t,
                             boost::bidirectional_traversal_tag, real_t> {
  //============================================================================
  using this_t = linspace_iterator<real_t>;

  //============================================================================
 private:
  const linspace<real_t>* m_lin;
  std::size_t             m_i;

  //============================================================================
 public:
  linspace_iterator(const linspace<real_t>* _lin, std::size_t _i)
      : m_lin{_lin}, m_i{_i} {}
  linspace_iterator(const linspace_iterator& other)
      : m_lin{other.m_lin}, m_i{other.m_i} {}
  auto& operator=(const linspace_iterator& other) {
    m_lin = other.m_lin;
    m_i   = other.m_i;
    return *this;
  }

  //----------------------------------------------------------------------------
  constexpr void to_begin() { m_i = 0; }

  //----------------------------------------------------------------------------
  constexpr void to_end() { m_i = m_lin->size(); }

  //----------------------------------------------------------------------------
  auto i() const { return m_i; }

  //----------------------------------------------------------------------------
  const auto& get() const { return *m_lin; }

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
template <typename real_t>
auto next(const linspace_iterator<real_t>& l, std::size_t diff = 1) {
  return linspace_iterator<real_t>{&l.get(), l.i() + diff};
}

//------------------------------------------------------------------------------
template <typename real_t>
auto prev(const linspace_iterator<real_t>& l, std::size_t diff = 1) {
  return linspace_iterator<real_t>{&l.get(), l.i() - diff};
}

//------------------------------------------------------------------------------
template <typename real_t>
inline auto& advance(linspace_iterator<real_t>& l, long n = 1) {
  if (n < 0) {
    while (n++) { --l; }
  } else {
    while (n--) { ++l; }
  }
  return l;
}

//------------------------------------------------------------------------------
template <typename real_t>
auto& operator<<(std::ostream& out, const linspace<real_t>& l) {
  out << "[" << l.m_min << ".. " << l.m_size << " .." << l.m_max << "]";
  return out;
}

//============================================================================
}  // namespace tatooine
//============================================================================

#endif
