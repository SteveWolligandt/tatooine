#ifndef TATOOINE_UTILITY_H
#define TATOOINE_UTILITY_H

#include <array>
#include <boost/core/demangle.hpp>
#include <vector>

#include "cxxstd.h"
#include "extract.h"
#include "make_array.h"
#include "variadic_helpers.h"

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
struct internal_data_type {
  using type = T;
};
//------------------------------------------------------------------------------
template <typename T>
using internal_data_type_t = typename internal_data_type<T>::type;
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
constexpr void for_each(F&& f, Ts&&... ts) {
  (f(std::forward<Ts>(ts)), ...);
}




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
inline auto type_name(T && /*t*/) -> std::string {
  return boost::core::demangle(typeid(T).name());
}
//------------------------------------------------------------------------------
/// returns demangled typename
template <typename T>
inline auto type_name() -> std::string {
  return boost::core::demangle(typeid(T).name());
}
//------------------------------------------------------------------------------
/// returns demangled typename
template <typename T>
inline auto type_name(std::string const& name) -> std::string {
  return boost::core::demangle(name.c_str());
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
//------------------------------------------------------------------------------
template <typename T>
constexpr void tat_swap(T& t0, T& t1) {
  T tmp = std::move(t0);
  t0    = std::move(t1);
  t1    = std::move(tmp);
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
