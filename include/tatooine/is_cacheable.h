#ifndef TATOOINE_IS_CACHEABLE_H
#define TATOOINE_IS_CACHEABLE_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Void = void>
struct is_cacheable_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
struct is_cacheable_impl<T, std::void_t<decltype(&T::use_caching)>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_cacheable_v = is_cacheable_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T>
constexpr auto is_cacheable() {
  return is_cacheable_v<T>;
}
//------------------------------------------------------------------------------
template <typename T>
constexpr auto is_cacheable(T&&) {
  return is_cacheable_v<std::decay_t<T>>;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
