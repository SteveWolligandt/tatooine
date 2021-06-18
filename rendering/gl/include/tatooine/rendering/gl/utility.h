#ifndef TATOOINE_RENDERING_GL_UTILITY_H
#define TATOOINE_RENDERING_GL_UTILITY_H
//==============================================================================
#include <array>
#include <concepts>
#include <utility>
#include <tatooine/rendering/gl/gltype.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine::rendering::gl {
//==============================================================================
template <typename T>
struct value_type;
template <typename T>
static constexpr auto value_type_v = value_type<T>::value;
template <typename T, size_t N>
struct value_type<std::array<T, N>>
    : std::integral_constant<GLenum, gl_type_v<T>> {};
template <typename T> requires std::is_arithmetic_v<T>
struct value_type<T> : std::integral_constant<GLenum, gl_type_v<T>> {};
template <typename T, size_t N>
struct value_type<vec<T, N>> : std::integral_constant<GLenum, gl_type_v<T>> {};
//==============================================================================
/// Applies function F to all elements of parameter pack ts
template <typename F, typename... Ts>
void for_each(F&& f, Ts&&... ts) {
  using discard_t = int[];
  // creates an array filled with zeros. while doing this f gets called with
  // elements of ts
  (void)discard_t{0, ((void)f(std::forward<Ts>(ts)), 0)...};
}
//==============================================================================
template <typename T, typename... Ts>
struct head {
  using type = T;
};
template <typename... Ts>
using head_t = typename head<Ts...>::type;
//==============================================================================
}  // namespace tatooine::rendering::gl
//==============================================================================
#endif
