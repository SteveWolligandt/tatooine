#ifndef TATOOINE_UTILITY_H
#define TATOOINE_UTILITY_H
//==============================================================================
//#include <tatooine/rank.h>
#include <tatooine/demangling.h>
#include <tatooine/make_array.h>
#include <tatooine/variadic_helpers.h>

#include <array>
#include <vector>
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
    const auto num_partitions = static_cast<size_t>(ceil(
        static_cast<double>(resolution[j]) / (max_chunk_resolution[j] - 1)));
    partitions[j]             = std::vector<std::pair<size_t, size_t>>(
        num_partitions, {0, max_chunk_resolution[j]});
    for (size_t i = 0; i < num_partitions; ++i) {
      partitions[j][i].first = (max_chunk_resolution[j] - 1) * i;
    }
    partitions[j].back().second = resolution[j] - partitions[j].back().first;
  }
  return partitions;
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
template <typename T, typename Ret, typename ... Args>
using const_method_ptr = Ret (T::*)(Args...) const;
template <typename T, typename Ret, typename ... Args>
using non_const_method_ptr = Ret (T::*)(Args...);
//==============================================================================
template <typename F, typename... Args>
auto repeat(size_t const n, F&& f, Args&&... args) {
  for (size_t i = 0; i < n; ++i) {
    f(std::forward<Args>(args)...);
  }
}
//==============================================================================
auto copy_or_keep_if_rvalue(auto&& x) -> decltype(auto) {
  if constexpr (std::is_rvalue_reference_v<decltype(x)>) {
    return std::forward<decltype(x)>(x);
  } else {
    return std::decay_t<decltype(x)>{x};
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
