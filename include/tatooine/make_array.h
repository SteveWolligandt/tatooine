#ifndef TATOOINE_MAKE_ARRAY_H
#define TATOOINE_MAKE_ARRAY_H
//==============================================================================
#include <array>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, size_t... Is>
constexpr auto make_array(T&& t, std::index_sequence<Is...> /*is*/) {
  return std::array<std::decay_t<T>, sizeof...(Is)>{((void)Is, t)...};
}
//------------------------------------------------------------------------------
template <typename T, typename ... Data>
constexpr auto make_array(Data&&... data) {
  return std::array<T, sizeof...(Data)>{static_cast<T>(data)...};
}
//------------------------------------------------------------------------------
template <typename T, size_t N>
constexpr auto make_array() {
  return make_array(T{}, std::make_index_sequence<N>{});
}
//------------------------------------------------------------------------------
template <size_t N, typename T>
constexpr auto make_array(T&& t) {
  return make_array(t, std::make_index_sequence<N>{});
}
////------------------------------------------------------------------------------
//template <typename T, size_t N>
//constexpr auto make_array(vec<T, N> const& data) {
//  return invoke_unpacked([](auto&&... data) { return make_array<T>(data...); },
//                         unpack(data));
//}
//------------------------------------------------------------------------------
template <typename T, size_t N>
constexpr auto make_array(std::array<T, N> const& data) {
  return invoke_unpacked([](auto&&... data) { return make_array<T>(data...); },
                         unpack(data));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
