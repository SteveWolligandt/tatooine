#ifndef TATOOINE_COMMON_TYPE_H
#define TATOOINE_COMMON_TYPE_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename...>
struct common_type_impl;
//------------------------------------------------------------------------------
template <typename T>
struct common_type_impl<T> {
  using type = T;
};
//------------------------------------------------------------------------------
template <typename T1, typename T2, typename... R>
struct common_type_impl<T1, T2, R...> {
  using type =
      std::common_type_t<T1, typename common_type_impl<T2, R...>::type>;
};
//==============================================================================
template <typename... Ts>
using common_type = typename common_type_impl<Ts...>::type;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
