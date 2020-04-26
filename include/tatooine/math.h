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
template <typename T, typename... Ts>
constexpr decltype(auto) max(T&& t0, Ts&&... ts) {
  return max(t0, max(std::forward<Ts>(ts)...));
}
//------------------------------------------------------------------------------
template <typename T>
constexpr T min(T t0, T t1) {
  return t0 < t1 ? t0 : t1;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename... Ts>
constexpr decltype(auto) min(T&& t0, Ts&&... ts) {
  return min(t0, min(std::forward<Ts>(ts)...));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Base, typename Exp, enable_if_integral<Base> = true,
          enable_if_integral<Exp> = true>
constexpr auto ipow(Base base, Exp exp) {
  using IntOut = Base;
  IntOut p     = 1;
  for (Exp i = 0; i < exp; ++i) { p *= base; }
  return p;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
