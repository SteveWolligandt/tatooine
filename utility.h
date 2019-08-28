#ifndef TATOOINE_UTILITY_H
#define TATOOINE_UTILITY_H

#include <array>
#include <boost/core/demangle.hpp>

//==============================================================================
namespace tatooine {
//==============================================================================

/// creates an index_sequence and removes an element from it
template <size_t Omit, size_t... Is, size_t... Js>
constexpr auto sliced_indices(std::index_sequence<Is...>,
                              std::index_sequence<Js...>) {
  std::array indices{Is...};
  (++indices[Js + Omit], ...);
  return indices;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// creates an index sequence and removes an element from it
template <size_t N, size_t Omit>
constexpr auto sliced_indices() {
  return sliced_indices<Omit>(std::make_index_sequence<N - 1>{},
                              std::make_index_sequence<N - Omit - 1>{});
}
//==============================================================================
template <typename F, typename... Ts>
void for_each(F&& f, Ts&&... ts) {
  (f(std::forward<Ts>(ts)), ...);
}

//==============================================================================
template <typename T, typename... Ts>
struct front {
  using type = T;
};
template <typename... Ts>
using front_t = typename front<Ts...>::type;

//==============================================================================
template <typename... T>
struct back;
template <typename T>
struct back<T> {
  using type = T;
};
template <typename T, typename... Ts>
struct back<T, Ts...> {
  using type = typename back<Ts...>::type;
};
template <typename... Ts>
using back_t = typename back<Ts...>::type;

//==============================================================================
template <typename T, size_t... Is>
auto make_array(const T& t, std::index_sequence<Is...> /*is*/) {
  return std::array<T, sizeof...(Is)>{((void)Is, t)...};
}
//------------------------------------------------------------------------------
template <typename T, size_t N>
auto make_array() {
  return make_array<T>(T{}, std::make_index_sequence<N>{});
}
//------------------------------------------------------------------------------
template <typename T, size_t N>
auto make_array(const T& t) {
  return make_array<T>(t, std::make_index_sequence<N>{});
}

//==============================================================================
template <size_t I, size_t Begin, size_t End, typename Cont>
constexpr auto& extract(Cont& extracted_data) {
  return extracted_data;
}
//------------------------------------------------------------------------------
template <size_t I, size_t Begin, size_t End, typename Cont, typename T,
          typename... Ts>
constexpr auto& extract(Cont& extracted_data, T&& t, Ts&&... ts) {
  static_assert(Begin <= End);
  if constexpr (I > End) { return extracted_data; }
  if constexpr (Begin >= I) { extracted_data[I - Begin] = t; }
  return extract<I + 1, Begin, End>(extracted_data, std::forward<Ts>(ts)...);
}
//------------------------------------------------------------------------------
template <size_t Begin, size_t End, typename... Ts>
constexpr auto extract(Ts&&... ts) {
  static_assert(Begin <= End);
  auto extracted_data =
      make_array<std::decay_t<front_t<Ts...>>, End - Begin + 1>();
  return extract<0, Begin, End>(extracted_data, std::forward<Ts>(ts)...);
}

//==============================================================================
template <size_t n>
struct multi_index_iterator;

//==============================================================================
template <size_t n>
struct multi_index {
  //----------------------------------------------------------------------------
  template <typename... Ts,
            std::enable_if_t<(std::is_integral_v<Ts> && ...)>...>
  constexpr multi_index(const std::pair<Ts, Ts>&... ranges)
      : m_ranges{ranges...} {}
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

//==============================================================================
/// returns demangled typename
template <typename T>
constexpr inline std::string type_name(T&& /*t*/) {
  return boost::core::demangle(typeid(T).name());
}

//------------------------------------------------------------------------------
/// returns demangled typename
template <typename T>
constexpr inline std::string type_name() {
  return boost::core::demangle(typeid(T).name());
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
