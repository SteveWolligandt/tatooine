#ifndef TATOOINE_MATH_H
#define TATOOINE_MATH_H
//==============================================================================
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// max for comparable objects.
/// If all types are the same a const reference is returned.
template <template <typename> typename Comparator, typename T0, typename T1,
          typename... TRest>
#ifdef __cpp_concepts
requires requires(
    T0&& a, T1&& b,
    Comparator<std::common_type_t<std::decay_t<T0>, std::decay_t<T1>>>&& comp) {
  { comp(a, b) }
  ->std::convertible_to<bool>;
}
#endif
constexpr auto compare_variadic(T0&& a, T1&& b, TRest&&... rest)
    -> std::conditional_t<
        std::is_same_v<std::decay_t<T0>,
        std::decay_t<T1>> &&
        (std::is_same_v<std::decay_t<T0>, std::decay_t<TRest>> && ...) &&
        std::is_lvalue_reference_v<T0> && std::is_lvalue_reference_v<T1> &&
        std::is_lvalue_reference_v<T1> &&
        (std::is_lvalue_reference_v<TRest> && ...),
        // if all raw types are equal and l-value references
        std::conditional_t<
            // if at least one of the types is const
            std::is_const_v<std::remove_reference_t<T0>> ||
                std::is_const_v<std::remove_reference_t<T1>> ||
                (std::is_const_v<std::remove_reference_t<TRest>> || ...),
            // return const-ref
            std::decay_t<T0> const&,
            // else return non-const-ref
            std::decay_t<T0>&>,
        // else return copy of common type
        std::common_type_t<std::decay_t<T0>, std::decay_t<T1>,
                           std::decay_t<TRest>...>> {
  using common_t = std::common_type_t<std::decay_t<T0>, std::decay_t<T1>>;
  if constexpr (sizeof...(TRest) == 0) {
    return Comparator<common_t>{}(a, b) ? std::forward<T0>(a) : std::forward<T1>(b);
  } else {
    return tatooine::compare_variadic<Comparator>(
        std::forward<T0>(a), tatooine::compare_variadic<Comparator>(
                                 std::forward<T1>(b),
                                 std::forward<TRest>(rest)...));
  }
}
//------------------------------------------------------------------------------
template <typename T0, typename T1, typename... TRest>
#ifdef __cpp_concepts
requires requires (T0&& a, T1&& b) {
  { a > b } -> std::convertible_to<bool>;
}
#endif
constexpr auto max(T0&& a, T1&& b, TRest&&... rest) -> decltype(auto) {
  return compare_variadic<std::greater>(
      std::forward<T0>(a), std::forward<T1>(b), std::forward<TRest>(rest)...);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... TRest>
#ifdef __cpp_concepts
requires requires (T0&& a, T1&& b) {
  { a < b } -> std::convertible_to<bool>;
}
#endif
constexpr auto min(T0&& a, T1&& b, TRest&&... rest) -> decltype(auto) {
  return compare_variadic<std::less>(
      std::forward<T0>(a), std::forward<T1>(b), std::forward<TRest>(rest)...);
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <integral Base, integral Exp>
#else
template <typename Base, typename Exp, enable_if_integral<Base, Exp> = true>
#endif
constexpr auto ipow(Base const base, Exp const exp) {
  std::decay_t<decltype(base)> p = 1;
  for (std::decay_t<decltype(exp)> i = 0; i < exp; ++i) {
    p *= base;
  }
  return p;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <integral Int>
#else
template <typename Int, enable_if_integral<Int> = true>
#endif
constexpr Int factorial(Int const i) {
  if (i == 0) {
    return 1;
  }
  return factorial(i - 1) * i;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
