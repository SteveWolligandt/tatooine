#ifndef TATOOINE_MATH_H
#define TATOOINE_MATH_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/pow.h>
#include <tatooine/type_traits.h>

#include <cmath>
#include <gcem.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename A, typename B> requires same_as<std::decay_t<A>, std::decay_t<B>>
constexpr auto min(A&& a, B&& b) { return gcem::min(std::forward<A>(a), std::forward<B>(b)); }
template <typename A, typename B> requires same_as<std::decay_t<A>, std::decay_t<B>>
constexpr auto max(A&& a, B&& b) { return gcem::max(std::forward<A>(a), std::forward<B>(b)); }
constexpr auto abs(arithmetic auto const x) { return gcem::abs(x); }
constexpr auto sin(arithmetic auto const x) { return gcem::sin(x); }
constexpr auto cos(arithmetic auto const x) { return gcem::cos(x); }
constexpr auto sqrt(arithmetic auto const x) { return gcem::sqrt(x); }
constexpr auto pow(arithmetic auto const x) { return gcem::pow(x); }
//==============================================================================
template <template <typename> typename Comparator, typename T0, typename T1,
          typename... TRest>
requires requires(T0 a, T1 b, Comparator<std::decay_t<T0>> comp) {
  { comp(a, b) } -> std::convertible_to<bool>;
}
&&  same_as<std::decay_t<T0>, std::decay_t<T1>>
&& (same_as<std::decay_t<T0>, std::decay_t<TRest>> && ...)
constexpr auto compare_variadic(T0&& a, T1&& b, TRest&&... rest) {
  if constexpr (sizeof...(TRest) == 0) {
    return Comparator<std::decay_t<T0>>{}(a, b) ? std::forward<T0>(a)
                                                : std::forward<T1>(b);
  } else {
    return tatooine::compare_variadic<Comparator>(
        std::forward<T0>(a),
        tatooine::compare_variadic<Comparator>(std::forward<T1>(b),
                                               std::forward<TRest>(rest)...));
  }
}
//------------------------------------------------------------------------------
template <typename T0>
constexpr auto max(T0&& a) -> decltype(auto) {
  return std::forward<T0>(a);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename T2, typename... TRest>
requires requires(T0&& a, T1&& b) {
  { a > b } -> std::convertible_to<bool>;
}
constexpr auto max(T0&& a, T1&& b, T2&& c, TRest&&... rest) -> decltype(auto) {
  return compare_variadic<std::greater>(
      std::forward<T0>(a), std::forward<T1>(b), std::forward<T2>(c), std::forward<TRest>(rest)...);
}
//------------------------------------------------------------------------------
template <typename T0, typename T1, typename T2, typename... TRest>
requires requires(T0&& a, T1&& b) {
  { a > b } -> std::convertible_to<bool>;
}
constexpr auto min(T0&& a, T1&& b, T2&& c, TRest&&... rest) -> decltype(auto) {
  return compare_variadic<std::less>(
      std::forward<T0>(a), std::forward<T1>(b), std::forward<T2>(c), std::forward<TRest>(rest)...);
}
//------------------------------------------------------------------------------
template <typename T, std::size_t N, std::size_t... Is>
constexpr auto min(std::array<T, N> const& arr,
                   std::index_sequence<Is...> /*seq*/) {
  return min(arr[Is]...);
}
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
constexpr auto min(std::array<T, N> const& arr) {
  return min(arr, std::make_index_sequence<N>{});
}
//------------------------------------------------------------------------------
template <typename T, std::size_t N, std::size_t... Is>
constexpr auto max(std::array<T, N> const& arr,
                   std::index_sequence<Is...> /*seq*/) {
  return max(arr[Is]...);
}
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
constexpr auto max(std::array<T, N> const& arr) {
  return max(arr, std::make_index_sequence<N>{});
}
//------------------------------------------------------------------------------
constexpr auto ipow(integral auto const base, integral auto const exp) {
  std::decay_t<decltype(base)> p = 1;
  for (std::decay_t<decltype(exp)> i = 0; i < exp; ++i) {
    p *= base;
  }
  return p;
}
//------------------------------------------------------------------------------
template <integral Int>  // Windows needs this
constexpr auto factorial(Int const i) -> std::decay_t<Int> {
  if (i == 0) {
    return 1;
  }
  return factorial(i - 1) * i;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
