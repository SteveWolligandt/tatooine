#ifndef TATOOINE_STATIC_MULTIDIM_SIZE_H
#define TATOOINE_STATIC_MULTIDIM_SIZE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/index_order.h>

#include <array>
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::size_t... Resolution>
struct static_multidim_iterator {
  static std::size_t constexpr num_dimensions() { return sizeof...(Resolution); }
  using array_type = std::array<std::size_t, num_dimensions()>;
  //----------------------------------------------------------------------------
 private:
  array_type m_status;
  static constexpr array_type resolution = array_type{Resolution...};
  //----------------------------------------------------------------------------
 public:
  static constexpr auto begin() {
    return static_multidim_iterator{std::array{(Resolution, std::size_t{})...}};
  }
  //----------------------------------------------------------------------------
  static constexpr auto end() {
    auto e = begin();
    e.m_status.back() = resolution.back();
    return e;
  }

 private:
  constexpr static_multidim_iterator(array_type const& status)
      :  m_status{status} {}

 public:
  //----------------------------------------------------------------------------
  constexpr static_multidim_iterator(static_multidim_iterator const& other) =
      default;
  constexpr static_multidim_iterator(
      static_multidim_iterator&& other) noexcept = default;
  //----------------------------------------------------------------------------
  constexpr auto operator=(static_multidim_iterator const& other)
      -> static_multidim_iterator& = default;
  constexpr auto operator=(static_multidim_iterator&& other) noexcept
      -> static_multidim_iterator& = default;
  //----------------------------------------------------------------------------
  ~static_multidim_iterator() = default;
  //----------------------------------------------------------------------------
  constexpr auto operator++() {
    ++m_status.front();
    auto status_it = std::begin(m_status);
    auto end_it = std::begin(resolution);
    for (; end_it != prev(std::end(resolution)); ++status_it, ++end_it) {
      auto & i = *status_it;
      auto & j = *std::next(status_it);
      auto const& end = *end_it;
      if (i >= end) {
        i = 0;
        ++j;
      }
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto operator==(static_multidim_iterator const& other) const {
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      if (m_status[i] != other.m_status[i]) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto operator!=(static_multidim_iterator const& other) const {
    return !operator==(other);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator*() const { return m_status; }
};
//==============================================================================
template <typename IndexOrder, std::size_t... Resolution>
struct static_multidim_size {
  static auto constexpr num_dimensions() { return sizeof...(Resolution); }
  static auto constexpr num_components() { return (Resolution * ...); }
  //----------------------------------------------------------------------------
  static auto constexpr size() { return std::array{Resolution...}; }
  //----------------------------------------------------------------------------
  static auto constexpr size(std::size_t const i) { return size()[i]; }
  //----------------------------------------------------------------------------
  static auto constexpr in_range(integral auto const... indices)
  requires(sizeof...(indices) == num_dimensions()) {
    return ((indices >= 0) && ...) &&
           ((static_cast<std::size_t>(indices) < Resolution) && ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto constexpr in_range(integral_range auto const& indices) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= size(i)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  static auto constexpr plain_index(integral auto const... indices) requires(
      sizeof...(indices) == num_dimensions()) {
    return IndexOrder::plain_index(size(), indices...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto plain_index(integral_range auto const& indices) {
    return IndexOrder::plain_index(size(), indices);
  }
  //----------------------------------------------------------------------------
  auto constexpr operator()(integral auto const... indices) const
      requires(sizeof...(indices) == num_dimensions()) {
    return plain_index(indices...);
  }
  //----------------------------------------------------------------------------
  auto begin() { return static_multidim_iterator<Resolution...>::begin(); }
  auto end() { return static_multidim_iterator<Resolution...>::end(); }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
