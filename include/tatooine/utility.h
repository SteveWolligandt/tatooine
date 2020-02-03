#ifndef TATOOINE_UTILITY_H
#define TATOOINE_UTILITY_H

#include "cxxstd.h"
#include "make_array.h"
#include <array>
#include <boost/core/demangle.hpp>

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename T>
struct internal_data_type;

template <>
struct internal_data_type<double> {
  using type = double;
};

template <>
struct internal_data_type<float> {
  using type = float;
};

//==============================================================================
template <typename T>
using internal_data_type_t = typename internal_data_type<T>::type;

/// creates an index_sequence and removes an element from it
template <size_t Omit, size_t... Is, size_t... Js>
constexpr auto sliced_indices(std::index_sequence<Is...>,
                              std::index_sequence<Js...>) {
#if has_cxx17_support()
  std::array indices{Is...};
  (++indices[Js + Omit], ...);
  return indices;
#else
  constexpr std::array<size_t, sizeof...(Is)> indices{Is...};
  constexpr std::array<size_t, sizeof...(Js)> js{Js...};
  for (auto j : js) { ++indices[j + Omit]; }
  return indices;
#endif
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// creates an index sequence and removes an element from it
template <size_t N, size_t Omit>
constexpr auto sliced_indices() {
  return sliced_indices<Omit>(std::make_index_sequence<N - 1>{},
                              std::make_index_sequence<N - Omit - 1>{});
}
//==============================================================================
#if has_cxx17_support()
template <typename F, typename... Ts>
constexpr void for_each(F&& f, Ts&&... ts) {
  (f(std::forward<Ts>(ts)), ...);
}
#else
template <typename F, typename T>
void for_each(F&& f, T&& t) {
  f(std::forward<T>(t));
}
template <typename F, typename T, typename... Ts>
void for_each(F&& f, T&& t, Ts&&... ts) {
  f(std::forward<T>(t));
  for_each(f, std::forward<Ts>(ts)...);
}
#endif

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
template <size_t I, size_t Begin, size_t End, typename Cont>
constexpr auto& extract(Cont& extracted_data) {
  return extracted_data;
}
//------------------------------------------------------------------------------

#if has_cxx17_support()
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
#endif
//==============================================================================
/// partitions a resolution into chunked resolutions.
/// borders of the partions are redundant.
template <size_t N>
auto partition_resolution(const std::array<size_t, N>& resolution,
                          const std::array<size_t, N>& max_chunk_resolution) {
  auto partitions = make_array<std::vector<std::pair<size_t, size_t>>, N>();
  for (size_t j = 0; j < N; ++j) {
    const auto num_partitions = static_cast<size_t>(
        ceil(static_cast<double>(resolution[j]) / (max_chunk_resolution[j] - 1)));
    partitions[j] = std::vector<std::pair<size_t, size_t>>(
        num_partitions, {0, max_chunk_resolution[j]});
    for (size_t i = 0; i < num_partitions; ++i) {
      partitions[j][i].first = (max_chunk_resolution[j] - 1) * i;
    }
    partitions[j].back().second = resolution[j] - partitions[j].back().first;
  }
  return partitions;
}
//==============================================================================
/// returns demangled typename
template <typename T>
inline std::string type_name(T&& /*t*/) {
  return boost::core::demangle(typeid(T).name());
}

//------------------------------------------------------------------------------
/// returns demangled typename
template <typename T>
inline std::string type_name() {
  return boost::core::demangle(typeid(T).name());
}

//==============================================================================
inline constexpr auto debug_mode() {
#ifndef NDEBUG
  return true;
#else
  return false;
#endif
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
inline constexpr auto release_mode() {
#ifdef NDEBUG
  return true;
#else
  return false;
#endif
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
