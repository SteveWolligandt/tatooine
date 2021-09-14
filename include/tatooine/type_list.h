#ifndef TATOOINE_TYPE_LIST_H
#define TATOOINE_TYPE_LIST_H
//==============================================================================
#include <tatooine/type_traits.h>
#include <tatooine/variadic_helpers.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \defgroup type_list List
/// \ingroup template_meta_programming
/// \{
//==============================================================================
template <typename... Ts>
struct type_list;
//==============================================================================
/// \defgroup type_list_size size
/// Using tatooine::type_list_size one can access the size of a
/// tatooine::type_list.
/// \ingroup type_list
/// \{
//==============================================================================
/// Size of a tatooine::type_list
template <typename TypeList>
struct type_list_size_impl;
//------------------------------------------------------------------------------
/// Size of a tatooine::type_list
template <typename TypeList>
static auto constexpr type_list_size = type_list_size_impl<TypeList>::value;
//------------------------------------------------------------------------------
/// Size of a tatooine::type_list
template <typename... Types>
struct type_list_size_impl<type_list<Types...>>
    : std::integral_constant<std::size_t, sizeof...(Types)> {};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_at at
/// Using tatooine::type_list_at one can access the ith type of a
/// tatooine::type_list.
/// \ingroup type_list
/// \{
//==============================================================================
/// Access to the Ith element of TypeList
template <typename TypeList, std::size_t I>
struct type_list_at_impl;
//------------------------------------------------------------------------------
/// Access to the Ith element of TypeList
template <typename TypeList, std::size_t I>
using type_list_at = typename type_list_at_impl<TypeList, I>::type;
//------------------------------------------------------------------------------
/// Recursive Stepping through all types of a list.
template <typename Front, typename... Rest, std::size_t I>
struct type_list_at_impl<type_list<Front, Rest...>, I> {
  static_assert(sizeof...(Rest) >= I, "Index exceeds range.");
  using type = typename type_list_at_impl<type_list<Rest...>, I - 1>::type;
};
//------------------------------------------------------------------------------
/// Returns the front of a tatooine::type_list with I = 0.
template <typename Front, typename... Rest>
struct type_list_at_impl<type_list<Front, Rest...>, 0> {
  using type = Front;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_back back
/// tatooine::type_list_back returns the last element of a tatooine::type_list.
/// \ingroup type_list
/// \{
//==============================================================================
template <typename TypeList>
struct type_list_back_impl;
//------------------------------------------------------------------------------
template <typename TypeList>
using type_list_back =
    typename type_list_back_impl<TypeList>::type;
//------------------------------------------------------------------------------
template <typename... Types>
struct type_list_back_impl<type_list<Types...>> {
  using type = variadic::back<Types...>;
};
//------------------------------------------------------------------------------
template <>
struct type_list_back_impl<type_list<>> {
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_front front
/// tatooine::type_list_back returns the first element of a tatooine::type_list.
/// \ingroup type_list
/// \{
//==============================================================================
template <typename TypeList>
struct type_list_front_impl;
//------------------------------------------------------------------------------
template <typename TypeList>
using type_list_front =
    typename type_list_front_impl<TypeList>::type;
//------------------------------------------------------------------------------
template <typename... Types>
struct type_list_front_impl<type_list<Types...>> {
  using type = variadic::front<Types...>;
};
//------------------------------------------------------------------------------
template <>
struct type_list_front_impl<type_list<>> {
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_push_back push_back
/// Using tatooine::type_list_push_back one can append a new type at the back of
/// a tatooine::type_list.
/// \ingroup type_list
/// \{
//==============================================================================
template <typename TypeList, typename NewBack>
struct type_list_push_back_impl;
//------------------------------------------------------------------------------
template <typename TypeList, typename NewBack>
using type_list_push_back =
    typename type_list_push_back_impl<TypeList, NewBack>::type;
//------------------------------------------------------------------------------
template <typename... Types, typename NewBack>
struct type_list_push_back_impl<type_list<Types...>, NewBack> {
  using type = type_list<Types..., NewBack>;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_push_front push_front
/// Using tatooine::type_list_push_back one can append a new type at the front
/// of a tatooine::type_list.
/// \ingroup type_list
/// \{
//==============================================================================
template <typename TypeList, typename NewFront>
struct type_list_push_front_impl;
//------------------------------------------------------------------------------
template <typename TypeList, typename NewFront>
using type_list_push_front =
    typename type_list_push_front_impl<TypeList, NewFront>::type;
//------------------------------------------------------------------------------
template <typename... Types, typename NewFront>
struct type_list_push_front_impl<type_list<Types...>, NewFront> {
  using type = type_list<NewFront, Types...>;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_pop_back pop_back
/// Using tatooine::type_list_push_back one can remove the back
/// of a tatooine::type_list.
/// \ingroup type_list
/// \{
//==============================================================================
template <typename TypeList, typename... TypesAccumulator>
struct type_list_pop_back_impl;
//------------------------------------------------------------------------------
template <typename TypeList>
using type_list_pop_back = typename type_list_pop_back_impl<TypeList>::type;
//------------------------------------------------------------------------------
template <typename T0, typename T1, typename... Rest,
          typename... TypesAccumulator>
struct type_list_pop_back_impl<type_list<T0, T1, Rest...>, TypesAccumulator...> {
  using type = typename type_list_pop_back_impl<type_list<T1, Rest...>,
                                                TypesAccumulator..., T0>::type;
};
//------------------------------------------------------------------------------
template <typename T, typename... TypesAccumulator>
struct type_list_pop_back_impl<type_list<T>, TypesAccumulator...> {
  using type = type_list<TypesAccumulator...>;
};
//------------------------------------------------------------------------------
template <typename... TypesAccumulator>
struct type_list_pop_back_impl<type_list<>, TypesAccumulator...> {
  using type = type_list<TypesAccumulator...>;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_pop_front pop_front
/// Using tatooine::type_list_pop_front one can remove the front
/// of a tatooine::type_list.
/// \ingroup type_list
/// \{
//==============================================================================
template <typename TypeList>
struct type_list_pop_front_impl;
//------------------------------------------------------------------------------
template <typename TypeList>
using type_list_pop_front = typename type_list_pop_front_impl<TypeList>::type;
//------------------------------------------------------------------------------
template <typename Front, typename... Back>
struct type_list_pop_front_impl<type_list<Front, Back...>> {
  using type = type_list<Back...>;
};
//------------------------------------------------------------------------------
template <>
struct type_list_pop_front_impl<type_list<>> {
  using type = type_list<>;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_contains contains
/// \ingroup type_list
/// \{
//==============================================================================
template <typename TypeList, typename T>
struct type_list_contains_impl;
//------------------------------------------------------------------------------
template <typename TypeList, typename T>
static auto constexpr type_list_contains =
    type_list_contains_impl<TypeList, T>::value;
//------------------------------------------------------------------------------
template <typename SetHead, typename... SetRest, typename T>
struct type_list_contains_impl<type_list<SetHead, SetRest...>, T> {
  static auto constexpr value =
      type_list_contains_impl<type_list<SetRest...>, T>::value;
};
//------------------------------------------------------------------------------
template <typename SetHead, typename... SetRest>
struct type_list_contains_impl<type_list<SetHead, SetRest...>,
                                SetHead> : std::true_type {};
//------------------------------------------------------------------------------
template <typename T>
struct type_list_contains_impl<type_list<>, T> : std::false_type {};
//==============================================================================
/// \}
//==============================================================================
/// \brief An empty struct that holds types.
/// \tparam Ts Variadic list of types.
template <typename... Ts>
struct type_list {
  using this_t = type_list<Ts...>;

  using front = type_list_front<this_t>;
  using back = type_list_back<this_t>;
  template <typename T>
  using push_back = type_list_push_back<this_t, T>;
  template <typename T>
  using push_front = type_list_push_front<this_t, T>;

  using pop_back = type_list_pop_back<this_t>;
  using pop_front = type_list_pop_front<this_t>;

  template <std::size_t I>
  using at = type_list_at<this_t, I>;
  template <typename T>
  static auto constexpr contains = type_list_contains<this_t, T>;

  static auto constexpr size = type_list_size<this_t>;
};
/// \}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
