#ifndef TATOOINE_VARIADIC_HELPERS_ITH_TYPE_H
#define TATOOINE_VARIADIC_HELPERS_ITH_TYPE_H
//==============================================================================
namespace tatooine::variadic {
//==============================================================================
template <std::size_t I, typename CurType, typename... RestTypes>
struct ith_type_impl {
  using type = typename ith_type_impl<I - 1, RestTypes...>::type;
};
template <typename CurType, typename... RestTypes>
struct ith_type_impl<0, CurType, RestTypes...> {
  using type = CurType;
};
template <std::size_t I, typename... Types>
using ith_type = typename ith_type_impl<I, Types...>::type;
//==============================================================================
}  // namespace tatooine::variadic
  //==============================================================================
#endif
