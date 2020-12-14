#ifndef TATOOINE_IS_CACHEABLE_H
#define TATOOINE_IS_CACHEABLE_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Void = void>
struct is_cacheable : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
struct is_cacheable<T, std::void_t<decltype(&T::use_caching)>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_cacheable_v = is_cacheable<T>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
