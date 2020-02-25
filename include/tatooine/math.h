#ifndef TATOOINE_MATH_H
#define TATOOINE_MATH_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
constexpr decltype(auto) max(T&& t0, T&& t1) {
  return t0 > t1 ? t0 : t1;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename... Ts>
constexpr decltype(auto) max(T&& t0, Ts&&... ts) {
  return max(t0, max(std::forward<Ts>(ts)...));
}
//------------------------------------------------------------------------------
template <typename T>
constexpr decltype(auto) min(T&& t0, T&& t1) {
  return t0 < t1 ? t0 : t1;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename... Ts>
constexpr decltype(auto) min(T&& t0, Ts&&... ts) {
  return min(t0, min(std::forward<Ts>(ts)...));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
