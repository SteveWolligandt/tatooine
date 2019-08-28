#ifndef TATOOINE_MULTIINDEXEDDATA_H
#define TATOOINE_MULTIINDEXEDDATA_H

#include <array>
#include <numeric>
#include "utility.h"

//==============================================================================
namespace tatooine {
//==============================================================================
template <size_t... Resolution>
struct static_multi_indexed_data {
  static constexpr size_t N        = sizeof...(Resolution);
  static constexpr size_t num_data = (Resolution * ...);
  static constexpr auto   resolution() { return std::array{Resolution...}; }

  //----------------------------------------------------------------------------
  template <typename... Indices, size_t... Is>
  static constexpr bool in_range(Indices&&... indices) {
    static_assert(sizeof...(Indices) == sizeof...(Resolution),
                  "number of indices does not match number of dimensions");
    return ((static_cast<size_t>(indices) >= 0) && ...) &&
           ((static_cast<size_t>(indices) < Resolution) && ...);
  }

  //----------------------------------------------------------------------------
  static constexpr auto in_range(const std::array<size_t, N>& indices) {
    return in_range(indices, std::make_index_sequence<N>{});
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  static constexpr auto in_range(const std::array<size_t, N>& indices,
                          std::index_sequence<Is...>)  {
    return in_range(indices[Is]...);
  }

  //----------------------------------------------------------------------------
  template <typename... Indices>
  static constexpr auto global_idx(Indices&&... indices) {
#ifndef NDEBUG
    if (!in_range(std::forward<Indices>(indices)...)) {
      throw std::runtime_error{"indices out of bounds: [ " +
                               ((std::to_string(indices) + " ") + ...) + "]"};
    }
#endif
    static_assert(
        (std::is_integral_v<std::decay_t<Indices>> && ...),
        "static_multi_indexed::global_idx() only takes integral types");
    static_assert(sizeof...(Indices) == sizeof...(Resolution),
                  "number of indices does not match number of dimensions");
    size_t multiplier = 1;
    size_t gi         = 0;
    for_each(
        [&](std::pair<size_t, size_t> i) {
          gi += i.first * multiplier;
          multiplier *= i.second;
        },
        std::pair{indices, Resolution}...);
    return gi;
  }

  //----------------------------------------------------------------------------
  static constexpr auto global_idx(const std::array<size_t, N>& indices) {
    return global_idx(indices, std::make_index_sequence<N>{});
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  static constexpr auto global_idx(const std::array<size_t, N>& indices,
                            std::index_sequence<Is...>) {
    return global_idx(indices[Is]...);
  }

  //----------------------------------------------------------------------------
  static constexpr auto multi_index(size_t gi)  {
    constexpr std::array resolution{Resolution...};
    auto                 is = make_array<size_t, N>();
    size_t               multiplier =
        std::accumulate(begin(resolution), std::prev(end(resolution)),
                        size_t(1), std::multiplies<size_t>{});

    auto res_it = std::prev(end(resolution), 2);
    for (size_t j = 0; j < N; ++j, --res_it) {
      size_t i = N - 1 - j;
      is[i]    = gi / multiplier;
      gi -= is[i] * multiplier;
      if (res_it >= begin(resolution)) { multiplier /= *res_it; }
    }

    return is;
  }
};

//==============================================================================
template <size_t N>
struct dynamic_multi_indexed_data {
  std::array<size_t, N> m_resolution;

  //----------------------------------------------------------------------------
  const auto& resolution() const { return m_resolution; }
  auto        resolution(const size_t i) const { return m_resolution[i]; }

  constexpr size_t num_data() const {
    return std::accumulate(begin(m_resolution), end(m_resolution), size_t(1),
                           std::multiplies<size_t>{});
  }

  //----------------------------------------------------------------------------
  template <typename... Resolution,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Resolution>> && ...)>,
            typename = std::enable_if_t<sizeof...(Resolution) == N>>
  void resize(Resolution&&... resolution) {
    m_resolution = {static_cast<size_t>(resolution)...};
  }
  void resize(std::array<size_t, N>&& resolution) {
    m_resolution = std::move(resolution);
  }
  void resize(const std::array<size_t, N>& resolution) {
    m_resolution = resolution;
  }

  //----------------------------------------------------------------------------
  template <typename... Indices, size_t... Is,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Indices>> && ...)>>
  constexpr bool in_range(std::index_sequence<Is...>,
                          Indices&&... indices) const {
    static_assert(sizeof...(Indices) == N,
                  "number of indices does not match number of dimensions");
    return ((static_cast<size_t>(indices) < m_resolution[Is]) && ...);
  }

  //----------------------------------------------------------------------------
  template <typename... Indices,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Indices>> && ...)>>
  constexpr auto in_range(Indices&&... indices) const {
    static_assert(sizeof...(Indices) == N,
                  "number of indices does not match number of dimensions");
    return in_range(std::make_index_sequence<N>{},
                    std::forward<Indices>(indices)...);
  }

  //----------------------------------------------------------------------------
  constexpr auto in_range(const std::array<size_t, N>& indices) const {
    return in_range(indices, std::make_index_sequence<N>{});
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto in_range(const std::array<size_t, N>& indices,
                          std::index_sequence<Is...>) const {
    return in_range(indices[Is]...);
  }

  //----------------------------------------------------------------------------
  template <typename... Indices,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Indices>> && ...)>>
  constexpr auto global_idx(Indices&&... indices) const {
    assert(in_range(std::forward<Indices>(indices)...));
    static_assert((std::is_integral_v<std::decay_t<Indices>> && ...),
                  "chunk::global_idx() only takes integral types");
    static_assert(sizeof...(Indices) == N,
                  "number of indices does not match number of dimensions");
    size_t multiplier = 1;
    size_t gi         = 0;
    auto   res_it     = begin(m_resolution);
    for_each(
        [&](size_t i) {
          gi += i * multiplier;
          multiplier *= *res_it;
          ++res_it;
        },
        std::forward<Indices>(indices)...);
    return gi;
  }

  //----------------------------------------------------------------------------
  constexpr auto global_idx(const std::array<size_t, N>& indices) const {
    return global_idx(indices, std::make_index_sequence<N>{});
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto global_idx(const std::array<size_t, N>& indices,
                            std::index_sequence<Is...>) const {
    return global_idx(indices[Is]...);
  }

  //----------------------------------------------------------------------------
  constexpr auto multi_index(size_t gi) const {
    auto   is = make_array<size_t, N>();
    size_t multiplier =
        std::accumulate(begin(m_resolution), std::prev(end(m_resolution)),
                        size_t(1), std::multiplies<size_t>{});

    auto res_it = std::prev(end(m_resolution), 2);
    for (size_t j = 0; j < N; ++j, --res_it) {
      size_t i = N - 1 - j;
      is[i]    = gi / multiplier;
      gi -= is[i] * multiplier;
      if (res_it >= begin(m_resolution)) { multiplier /= *res_it; }
    }

    return is;
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
