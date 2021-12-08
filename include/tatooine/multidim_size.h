#ifndef TATOOINE_MULTIDIM_SIZE_H
#define TATOOINE_MULTIDIM_SIZE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/functional.h>
#include <tatooine/index_order.h>
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
template <typename IndexOrder, size_t... Resolution>
struct static_multidim_size {
  static constexpr auto num_dimensions() { return sizeof...(Resolution); }
  static constexpr auto num_components() { return (Resolution * ...); }
  //----------------------------------------------------------------------------
  static constexpr auto size() { return std::array{Resolution...}; }
  //----------------------------------------------------------------------------
  static constexpr auto size(size_t const i) { return size()[i]; }
  //----------------------------------------------------------------------------
  static constexpr auto in_range(integral auto const... indices) requires(
      sizeof...(indices) == num_dimensions()) {
    return ((indices >= 0) && ...) &&
           ((static_cast<size_t>(indices) < Resolution) && ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static constexpr auto in_range(integral_range auto const& indices) {
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= size(i)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  static constexpr auto plain_index(integral auto const... indices) requires(
      sizeof...(indices) == num_dimensions()) {
    return IndexOrder::plain_index(size(), indices...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto plain_index(integral_range auto const& indices) {
    return IndexOrder::plain_index(size(), indices);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... indices) const
      requires(sizeof...(indices) == num_dimensions()) {
    return plain_index(indices...);
  }
  //----------------------------------------------------------------------------
  static constexpr auto indices() { return static_multidim{Resolution...}; }
};
//==============================================================================
template <typename IndexOrder>
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
  explicit dynamic_multidim_size(integral auto const... size)
      : m_size{static_cast<size_t>(size)...} {}
  //----------------------------------------------------------------------------
  explicit dynamic_multidim_size(std::vector<size_t>&& size)
      : m_size(std::move(size)) {}
  //----------------------------------------------------------------------------
  explicit dynamic_multidim_size(integral_range auto const& size)
      : m_size(begin(size), end(size)) {}
  //----------------------------------------------------------------------------
  // comparisons
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  auto operator==(dynamic_multidim_size<OtherIndexing> const& other) const {
    if (num_dimensions() != other.num_dimensions()) {
      return false;
    }
    for (size_t i = 0; i < num_dimensions(); ++i) {
      if (m_size[i] != other.size(i)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  auto operator!=(dynamic_multidim_size<OtherIndexing> const& other) const {
    if (num_dimensions() == other.num_dimensions()) {
      return false;
    }
    for (size_t i = 0; i < num_dimensions(); ++i) {
      if (m_size[i] == other.size(i)) {
        return false;
      }
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
  void resize(integral auto const... size) {
    m_size = {static_cast<size_t>(size)...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(integral_range auto const& size) {
    m_size = std::vector<size_t>(begin(size), end(size));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(std::vector<size_t>&& size) { m_size = std::move(size); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(std::vector<size_t> const& size) { m_size = size; }
  //----------------------------------------------------------------------------
  constexpr auto in_range(integral auto const... indices) const {
    assert(sizeof...(indices) == num_dimensions());
    return in_range(std::array{static_cast<size_t>(indices)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto in_range(integral_range auto const& indices) const {
    assert(indices.size() == num_dimensions());
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= size(i)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto plain_index(integral auto const... indices) const {
    assert(sizeof...(indices) == num_dimensions());
    assert(in_range(indices...));
    return IndexOrder::plain_index(m_size, indices...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto plain_index(integral_range auto const& indices) const {
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return IndexOrder::plain_index(m_size, indices);
  }
  //----------------------------------------------------------------------------
  constexpr auto multi_index(size_t const gi) const {
    return IndexOrder::multi_index(m_size, gi);
  }
  //----------------------------------------------------------------------------
  constexpr auto indices() const { return dynamic_multidim{m_size}; }
};
//==============================================================================
// deduction guides
//==============================================================================
dynamic_multidim_size()->dynamic_multidim_size<x_fastest>;
template <typename IndexOrder>
dynamic_multidim_size(dynamic_multidim_size<IndexOrder> const&)
    -> dynamic_multidim_size<IndexOrder>;
template <typename IndexOrder>
dynamic_multidim_size(dynamic_multidim_size<IndexOrder>&&)
    -> dynamic_multidim_size<IndexOrder>;
template <typename... Resolution>
dynamic_multidim_size(Resolution...) -> dynamic_multidim_size<x_fastest>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
