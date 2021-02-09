#ifndef TATOOINE_MULTIDIM_SIZE_H
#define TATOOINE_MULTIDIM_SIZE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/functional.h>
#include <tatooine/index_ordering.h>
#include <tatooine/multidim.h>
#include <tatooine/template_helper.h>
#include <tatooine/type_traits.h>
#include <tatooine/utility.h>

#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <sstream>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Indexing, size_t... Resolution>
struct static_multidim_size {
  static constexpr auto num_dimensions() { return sizeof...(Resolution); }
  static constexpr auto num_components() { return (Resolution * ...); }
  //----------------------------------------------------------------------------
  static constexpr auto size() {
    return std::array{Resolution...};
  }
  //----------------------------------------------------------------------------
  static constexpr auto size(size_t const i) { return size()[i]; }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  static constexpr auto in_range(Is const... is) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return ((is >= 0) && ...) &&
           ((static_cast<size_t>(is) < Resolution) && ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  static constexpr auto in_range(Indices const& is) {
    for (size_t i = 0; i < is.size(); ++i) {
      if (is[i] >= size(i)) { return false; }
    }
    return true;
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  static constexpr auto plain_index(Is const... is) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return Indexing::plain_index(size(), is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  static auto plain_index(Indices const& is) {
    static_assert(std::is_integral_v<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(is.size() == num_dimensions() &&
           "number of indices does not match number of dimensions");
    return Indexing::plain_index(size(), is);
  }
  //----------------------------------------------------------------------------
  static constexpr auto indices() { return static_multidim{Resolution...}; }
};
//==============================================================================
template <typename Indexing>
class dynamic_multidim_size {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  std::vector<size_t> m_size;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  dynamic_multidim_size() = default;
  //----------------------------------------------------------------------------
  dynamic_multidim_size(dynamic_multidim_size const& other)     = default;
  dynamic_multidim_size(dynamic_multidim_size&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator                 =(dynamic_multidim_size const& other)
      -> dynamic_multidim_size& = default;
  auto operator                 =(dynamic_multidim_size&& other) noexcept
      -> dynamic_multidim_size& = default;
  //----------------------------------------------------------------------------
  ~dynamic_multidim_size() = default;
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  explicit dynamic_multidim_size(
      dynamic_multidim_size<OtherIndexing> const& other)
      : m_size{other.size()} {}

  template <typename OtherIndexing>
  explicit dynamic_multidim_size(dynamic_multidim_size<OtherIndexing>&& other)
      : m_size{std::move(other.m_size)} {}

  template <typename OtherIndexing>
  auto operator=(dynamic_multidim_size const& other) -> dynamic_multidim_size& {
    m_size = other.m_size;
    return *this;
  }
  template <typename OtherIndexing>
  auto operator=(dynamic_multidim_size&& other) -> dynamic_multidim_size& {
    m_size = std::move(other.m_size);
    return *this;
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Size>
#else
  template <typename... Size, enable_if<is_integral<Size...>> = true>
#endif
  explicit dynamic_multidim_size(Size... size)
      : m_size{static_cast<size_t>(size)...} {}
  //----------------------------------------------------------------------------
  explicit dynamic_multidim_size(std::vector<size_t>&& size)
      : m_size(std::move(size)) {}
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <range Size>
#else
  template <typename Size, enable_if<is_range<Size>> = true>
#endif
  explicit dynamic_multidim_size(Size const& size)
      : m_size(begin(size), end(size)) {
    static_assert(std::is_integral_v<typename Size::value_type>,
                  "size range must hold integral type");
  }
  //----------------------------------------------------------------------------
  // comparisons
  //----------------------------------------------------------------------------
 public:
  template <typename OtherIndexing>
  auto operator==(dynamic_multidim_size<OtherIndexing> const& other) const {
    if (num_dimensions() != other.num_dimensions()) { return false; }
    for (size_t i = 0; i < num_dimensions(); ++i) {
      if (m_size[i] != other.size(i)) { return false; }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  auto operator!=(dynamic_multidim_size<OtherIndexing> const& other) const {
    if (num_dimensions() == other.num_dimensions()) { return false; }
    for (size_t i = 0; i < num_dimensions(); ++i) {
      if (m_size[i] == other.size(i)) { return false; }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  auto num_dimensions() const { return m_size.size(); }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto size() const -> auto const& { return m_size; }
  /// \return size of dimensions i
  auto size(size_t const i) const { return m_size[i]; }
  //----------------------------------------------------------------------------
  auto num_components() const {
    return std::accumulate(begin(m_size), end(m_size), size_t(1),
                           std::multiplies<size_t>{});
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Size>
#else
  template <typename... Size, enable_if<is_integral<Size...>> = true>
#endif
  void resize(Size const... size) {
    m_size = {static_cast<size_t>(size)...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Size>
#else
  template <typename Size, enable_if<is_range<Size>> = true>
#endif
  void resize(Size const& size) {
    m_size = std::vector<size_t>(begin(size), end(size));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(std::vector<size_t>&& size) { m_size = std::move(size); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(std::vector<size_t> const& size) { m_size = size; }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  constexpr auto in_range(Is const... is) const {
    assert(sizeof...(is) == num_dimensions());
    return in_range(std::array{static_cast<size_t>(is)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  constexpr auto in_range(Indices const& is) const {
    static_assert(std::is_integral_v<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(is.size() == num_dimensions());
    for (size_t i = 0; i < is.size(); ++i) {
      if (is[i] >= size(i)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  constexpr auto plain_index(Is const... is) const {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return Indexing::plain_index(m_size, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  constexpr auto plain_index(Indices const& is) const {
    static_assert(std::is_integral_v<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return Indexing::plain_index(m_size, is);
  }
  //----------------------------------------------------------------------------
  constexpr auto multi_index(size_t const gi) const {
    return Indexing::multi_index(m_size, gi);
  }
  //----------------------------------------------------------------------------
  constexpr auto indices() const {
    return dynamic_multidim{m_size};
  }
};
//==============================================================================
// deduction guides
//==============================================================================
dynamic_multidim_size()->dynamic_multidim_size<x_fastest>;
template <typename Indexing>
dynamic_multidim_size(dynamic_multidim_size<Indexing> const&)
    -> dynamic_multidim_size<Indexing>;
template <typename Indexing>
dynamic_multidim_size(dynamic_multidim_size<Indexing> &&)
    -> dynamic_multidim_size<Indexing>;
template <typename... Resolution>
dynamic_multidim_size(Resolution...) -> dynamic_multidim_size<x_fastest>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
