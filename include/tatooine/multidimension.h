#ifndef TATOOINE_MULTIDIMENSION_H
#define TATOOINE_MULTIDIMENSION_H

#include <array>
#include <numeric>
#include "template_helper.h"
#include "utility.h"
#include "functional.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <size_t n>
struct multi_index_iterator;

//==============================================================================
template <size_t n>
struct multi_index {
  //----------------------------------------------------------------------------
  constexpr multi_index(std::array<std::pair<size_t, size_t>, n> ranges)
      : m_ranges{ranges} {}

  //----------------------------------------------------------------------------
  template <typename... Ts,
            typename = std::enable_if_t<(std::is_integral_v<Ts> && ...)>>
  constexpr multi_index(const std::pair<Ts, Ts>&... ranges)
      : m_ranges{std::pair{static_cast<size_t>(ranges.first),
                           static_cast<size_t>(ranges.second)}...} {}

  //----------------------------------------------------------------------------
  template <typename... Ts,
            typename = std::enable_if_t<(std::is_integral_v<Ts> && ...)>>
  constexpr multi_index(Ts const (&... ranges)[2])
      : m_ranges{std::pair{static_cast<size_t>(ranges[0]),
                           static_cast<size_t>(ranges[1])}...} {}

  //----------------------------------------------------------------------------
  constexpr auto&       operator[](size_t i) { return m_ranges[i]; }
  constexpr const auto& operator[](size_t i) const { return m_ranges[i]; }

  //----------------------------------------------------------------------------
  constexpr auto&       ranges() { return m_ranges; }
  constexpr const auto& ranges() const { return m_ranges; }

  //----------------------------------------------------------------------------
  constexpr auto begin() { return begin(std::make_index_sequence<n>{}); }
  constexpr auto end() { return end(std::make_index_sequence<n>{}); }

 private:
  //============================================================================
  std::array<std::pair<size_t, size_t>, n> m_ranges;

  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto begin(std::index_sequence<Is...> /*is*/) {
    return multi_index_iterator<n>{*this,
                                   std::array<size_t, n>{((void)Is, 0)...}};
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto end(std::index_sequence<Is...> /*is*/) {
    std::array<size_t, n> a{((void)Is, 0)...};
    a.back() = m_ranges.back().second + 1;
    return multi_index_iterator<n>{*this, std::move(a)};
  }
};

//==============================================================================
template <size_t n>
struct multi_index_iterator {
  //----------------------------------------------------------------------------
  const multi_index<n>  m_cont;
  std::array<size_t, n> m_status;

  //----------------------------------------------------------------------------
  constexpr multi_index_iterator(const multi_index<n>&        c,
                                 const std::array<size_t, n>& status)
      : m_cont{c}, m_status{status} {}

  //----------------------------------------------------------------------------
  constexpr multi_index_iterator(const multi_index_iterator& other)
      : m_cont{other.m_cont}, m_status{other.m_status} {}

  //----------------------------------------------------------------------------
  constexpr void operator++() {
    ++m_status.front();
    auto range_it  = begin(m_cont.ranges());
    auto status_it = begin(m_status);
    for (; range_it != prev(end(m_cont.ranges())); ++status_it, ++range_it) {
      if (range_it->second < *status_it) {
        *status_it = 0;
        ++(*(status_it + 1));
      }
    }
  }

  //----------------------------------------------------------------------------
  constexpr auto operator==(const multi_index_iterator& other) const {
    for (size_t i = 0; i < n; ++i) {
      if (m_status[i] != other.m_status[i]) { return false; }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  constexpr auto operator!=(const multi_index_iterator& other) const {
    return !operator==(other);
  }

  //----------------------------------------------------------------------------
  constexpr auto operator*() const { return m_status; }
};

//==============================================================================
template <typename... Ts>
multi_index(const std::pair<Ts, Ts>&... ranges)->multi_index<sizeof...(Ts)>;
template <typename... Ts>
multi_index(Ts const (&... ranges)[2])->multi_index<sizeof...(Ts)>;

template <size_t... Resolution>
struct static_multidimension {
  static constexpr size_t N        = sizeof...(Resolution);
  static constexpr size_t num_data = (Resolution * ...);
  static constexpr auto   resolution() { return std::array{Resolution...}; }
  template <size_t i>
  static constexpr auto resolution() {
    return temp_helper::getval<i, size_t, Resolution...>;
  }

  //----------------------------------------------------------------------------
  template <typename... Indices, size_t... Is>
  static constexpr bool in_range(Indices&&... indices) {
    static_assert(sizeof...(Indices) == sizeof...(Resolution),
                  "number of indices does not match number of dimensions");
    return ((indices >= 0) && ...) && ((indices < Resolution) && ...);
  }

  //----------------------------------------------------------------------------
  static constexpr auto in_range(const std::array<size_t, N>& indices)  {
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
    map(
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
  static constexpr auto multi_index(size_t gi) {
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
struct dynamic_multidimension {
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
    map(
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
