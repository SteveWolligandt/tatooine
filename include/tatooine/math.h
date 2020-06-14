#ifndef TATOOINE_MATH_H
#define TATOOINE_MATH_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
constexpr T max(T t0, T t1) {
  return t0 > t1 ? t0 : t1;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename... Ts,
          std::enable_if_t<(sizeof...(Ts) > 1), bool> = true>
constexpr T max(T&& t0, Ts&&... ts) {
  return tatooine::max(t0, tatooine::max(std::forward<Ts>(ts)...));
}
//------------------------------------------------------------------------------
template <typename T>
constexpr T min(T t0, T t1) {
  return t0 < t1 ? t0 : t1;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename... Ts,
          std::enable_if_t<(sizeof...(Ts) > 1), bool> = true>
constexpr T min(T&& t0, Ts&&... ts) {
  return tatooine::min(t0, tatooine::min(std::forward<Ts>(ts)...));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
constexpr auto ipow(integral auto const base, integral auto const exp) {
  std::decay_t<decltype(base)> p = 1;
  for (std::decay_t<decltype(exp)> i = 0; i < exp; ++i) { p *= base; }
  return p;
}
//------------------------------------------------------------------------------
template <integral Int>
constexpr Int factorial(Int const i) {
  if (i == 0) { return 1; }
  return factorial(i - 1) * i;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
