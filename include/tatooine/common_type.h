#ifndef TATOOINE_COMMON_TYPE_H
#define TATOOINE_COMMON_TYPE_H
//==============================================================================
#include <tatooine/void_t.h>
//==============================================================================
namespace tatooine {
//==============================================================================
// primary template (used for zero types)
template <typename...>
struct common_type_impl {};
 
//////// one type
template <typename T>
struct common_type_impl<T> : common_type_impl<T, T> {};
 
//////// two types
template <typename T1, typename T2>
using common_type_cond_t = decltype(false ? std::declval<T1>() : std::declval<T2>());
 
template <typename T1, typename T2, typename=void>
struct common_type_2_impl {};
 
template <typename T1, typename T2>
struct common_type_2_impl<T1, T2, void_t<common_type_cond_t<T1, T2>>> {
    using type = typename std::decay<common_type_cond_t<T1, T2>>::type;
};
 
template <typename T1, typename T2>
struct common_type_impl<T1, T2> 
  : common_type_2_impl<typename std::decay<T1>::type, 
                       typename std::decay<T2>::type>
{};
 
//////// 3+ types
template <typename AlwaysVoid, typename T1, typename T2, typename...R>
struct common_type_multi_impl {};
 
template <typename T1, typename T2, typename...R>
struct common_type_multi_impl<
      void_t<typename common_type_impl<T1, T2>::type>, T1, T2, R...>
  : common_type_impl<typename common_type_impl<T1, T2>::type, R...> {};
 
 
template <typename T1, typename T2, typename... R>
struct common_type_impl<T1, T2, R...>
  : common_type_multi_impl<void, T1, T2, R...> {};

//==============================================================================
template <typename... Ts>
using common_type = typename common_type_impl<Ts...>::type;
//==============================================================================
} // namespace tatooine
//==============================================================================
#endif
