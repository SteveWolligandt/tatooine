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
//==============================================================================
template <floating_point Real>
struct iterator_sentinel;
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
  using this_type     = linspace<Real>;
  using real_type     = Real;
  using value_type = real_type;
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
      : m_min{std::min(static_cast<Real>(min), static_cast<Real>(max))},
        m_max{std::max(static_cast<Real>(min), static_cast<Real>(max))},
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
    return m_min + spacing() * static_cast<Real>(i);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator[](std::size_t const i) const { return at(i); }
  //----------------------------------------------------------------------------
  constexpr auto begin() const { return iterator{this, 0}; }
  constexpr auto end() const { return typename iterator::sentinel_type{this}; }
  //----------------------------------------------------------------------------
  constexpr auto size() const { return m_size; }
  constexpr auto size() -> auto& { return m_size; }
  constexpr auto front() const { return m_min; }
  constexpr auto front() -> auto& { return m_min; }
  constexpr auto back() const { return m_max; }
  constexpr auto back() -> auto& { return m_max; }
  //----------------------------------------------------------------------------
  constexpr auto spacing() const {
    return (m_max - m_min) / static_cast<Real>(m_size - 1);
  }
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
//template <typename T>
//TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
//    linspace<T>, TATOOINE_REFLECTION_INSERT_METHOD(front, front()),
//    TATOOINE_REFLECTION_INSERT_METHOD(back, back()),
//    TATOOINE_REFLECTION_INSERT_METHOD(size, size()))
//==============================================================================
}  // namespace tatooine::reflection
//==============================================================================
namespace tatooine::detail::linspace {
//==============================================================================
template <floating_point Real>
struct iterator_sentinel {
  tatooine::linspace<Real> const* m_lin;
};
//==============================================================================
template <floating_point Real>
struct iterator : iterator_facade<iterator<Real>> {
  using sentinel_type = detail::linspace::iterator_sentinel<Real>;
  using linspace_type = tatooine::linspace<Real>;
  //============================================================================
  // typedefs
  //============================================================================
  using this_type = iterator<Real>;

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
  constexpr auto increment(std::size_t inc = 1) { m_i += inc; }
  constexpr auto decrement(std::size_t dec = 1) { m_i -= dec; }
  constexpr auto at_end() const { return m_i == m_lin->size(); }
  constexpr auto distance_to(iterator const& other) const -> std::ptrdiff_t {
    return other.m_i - m_i;
  }
  constexpr auto distance_to(sentinel_type const /*sentinel*/) const
      -> std::ptrdiff_t {
    return m_lin->size() - m_i;
  }
  constexpr auto advance(std::ptrdiff_t const off) { m_i += off; }
};
//==============================================================================
template <floating_point Real>
auto prev(iterator_sentinel<Real>const& sent) {
  return iterator{sent.m_lin, sent.m_lin->size() - 1};
}
//==============================================================================
}  // namespace tatooine::detail::linspace
//============================================================================
#endif
