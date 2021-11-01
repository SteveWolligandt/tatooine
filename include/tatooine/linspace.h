#ifndef TATOOINE_LINSPACE_H
#define TATOOINE_LINSPACE_H
//============================================================================
#ifdef __cpp_concepts
#include <tatooine/concepts.h>
#endif
#include <tatooine/reflection.h>
#include <tatooine/type_traits.h>

#include <boost/iterator/iterator_facade.hpp>
#include <cstddef>
#include <functional>
#include <ostream>
//============================================================================
namespace tatooine {
//============================================================================
// forward declarations
//============================================================================
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real>
#endif
struct linspace_iterator;
//============================================================================
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real>
#endif
struct linspace {
#ifndef __cpp_concepts
  static_assert(is_arithmetic<Real>);
#endif
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
#if __cpp_concepts
  template <arithmetic Min, arithmetic Max>
#else
  template <typename Min, typename Max>
#endif
  constexpr linspace(Min const min, Max const max, size_t size) noexcept
      : m_min{std::min<Real>(min, max)},
        m_max{std::max<Real>(min, max)},
        m_size{size} {
  }
  //----------------------------------------------------------------------------
  constexpr linspace(linspace const&)     = default;
  constexpr linspace(linspace&&) noexcept = default;
  //----------------------------------------------------------------------------
#if __cpp_concepts
  template <arithmetic OtherReal>
#else
  template <typename OtherReal, enable_if<is_arithmetic<OtherReal>> = true>
#endif
  explicit constexpr linspace(linspace<OtherReal> const& other) noexcept
      : m_min{static_cast<Real>(other.front())},
        m_max{static_cast<Real>(other.back())},
        m_size{other.size()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator=(linspace const&) -> linspace& = default;
  constexpr auto operator=(linspace&&) noexcept -> linspace& = default;
  //----------------------------------------------------------------------------
#if __cpp_concepts
  template <arithmetic OtherReal>
#else
  template <typename OtherReal, enable_if<is_arithmetic<OtherReal>> = true>
#endif
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
  constexpr auto resize(size_t const s) {m_size = s;}
};
namespace reflection{
template <typename T>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    linspace<T>,
    TATOOINE_REFLECTION_INSERT_METHOD(front, front()),
    TATOOINE_REFLECTION_INSERT_METHOD(back, back()),
    TATOOINE_REFLECTION_INSERT_METHOD(size, size()))
}
//==============================================================================
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real>
#endif
struct linspace_iterator
    : boost::iterator_facade<linspace_iterator<Real>, Real,
                             std::bidirectional_iterator_tag, Real> {
#ifndef __cpp_concepts
  static_assert(is_arithmetic<Real>);
#endif
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
  auto equal(linspace_iterator const& other) const { return m_i == other.m_i; }
  //----------------------------------------------------------------------------
  auto dereference() const { return m_lin->at(m_i); }
 public:
  //----------------------------------------------------------------------------
  void increment() { ++m_i; }
  void increment(size_t n) { m_i += n; }
  //----------------------------------------------------------------------------
  void decrement() { --m_i; }
  void decrement(size_t n) { m_i -= n; }
};

//==============================================================================
// free functions
//==============================================================================
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
constexpr auto begin(linspace<Real> const& l) {
  return l.begin();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
constexpr auto end(linspace<Real> const& l) {
  return l.end();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
constexpr auto distance(linspace_iterator<Real> const& it0,
                        linspace_iterator<Real> const& it1) {
  return it0.distance(it1);
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
auto size(linspace<Real> const& l) {
  return l.size();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
auto front(linspace<Real> const& l) {
  return l.front();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
auto front(linspace<Real>& l) -> auto& {
  return l.front();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
auto back(linspace<Real> const& l) {
  return l.back();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
auto back(linspace<Real>& l) -> auto& {
  return l.back();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
auto next(linspace_iterator<Real> const& l, size_t diff = 1) {
  linspace_iterator<Real> it{l};
  it.increment(diff);
  return it;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
auto prev(linspace_iterator<Real> const& l, size_t diff = 1) {
  linspace_iterator<Real> it{l};
  it.decrement(diff);
  return it;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
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
#ifdef __cpp_concepts
template <arithmetic Real0, arithmetic Real1>
#else
template <typename Real0, typename Real1>
#endif
linspace(Real0 const, Real1 const, size_t) -> linspace<common_type<Real0, Real1>>;

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
#ifdef __cpp_concepts
template <arithmetic Real>
#else
template <typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
auto operator<<(std::ostream& out, linspace<Real> const& l) -> auto& {
  out << "[" << l[0] << ", " << l[1] << ", ..., " << l.back() << "]";
  return out;
}
//============================================================================
}  // namespace tatooine
//============================================================================
#endif
