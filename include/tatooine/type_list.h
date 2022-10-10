#ifndef TATOOINE_TYPE_LIST_H
#define TATOOINE_TYPE_LIST_H
//==============================================================================
#include <tatooine/type_traits.h>
#include <tatooine/variadic_helpers.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \defgroup template_meta_programming_type_list List
/// \ingroup template_meta_programming
/// Use tatooine::type_list to construct a type list:
/// ```cpp
///  using list = tatooine::type_list<int, double, int, float>;
/// ```
/// `list` is of type tatooine::type_list_impl and holds the `int`, `double`,
/// `int` and `float`.
/// \{
//==============================================================================
template <typename... Ts>
struct type_list_impl;
//==============================================================================
/// \defgroup type_list_size size
/// Using tatooine::type_list_size one can access the size of a
/// tatooine::type_list_impl.
/// \ingroup template_meta_programming_type_list
/// \{
//==============================================================================
/// Size of a tatooine::type_list_impl
template <typename TypeList>
struct type_list_size_impl;
//------------------------------------------------------------------------------
/// Size of a tatooine::type_list_impl
template <typename TypeList>
static auto constexpr type_list_size = type_list_size_impl<TypeList>::value;
//------------------------------------------------------------------------------
/// Size of a tatooine::type_list_impl
template <typename... Types>
struct type_list_size_impl<type_list_impl<Types...>>
    : std::integral_constant<std::size_t, sizeof...(Types)> {};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_at at
/// Using tatooine::type_list_at one can access the ith type of a
/// tatooine::type_list_impl.
/// \ingroup template_meta_programming_type_list
/// \{
//==============================================================================
/// Access to the Ith element of TypeList
template <typename TypeList, std::size_t I>
struct type_list_at_impl;
struct type_list_out_of_bounds{};
//------------------------------------------------------------------------------
/// Access to the Ith element of TypeList
template <typename TypeList, std::size_t I>
using type_list_at = typename type_list_at_impl<TypeList, I>::type;
//------------------------------------------------------------------------------
/// Recursive Stepping through all types of a list.
template <typename Front, typename... Rest, std::size_t I>
struct type_list_at_impl<type_list_impl<Front, Rest...>, I> {
  using type = typename type_list_at_impl<type_list_impl<Rest...>, I - 1>::type;
};
//------------------------------------------------------------------------------
/// Recursive Stepping through all types of a list.
template <std::size_t I>
struct type_list_at_impl<type_list_impl<>, I> {
  using type = type_list_out_of_bounds;
};
//------------------------------------------------------------------------------
/// Returns the front of a tatooine::type_list_impl with I = 0.
template <typename Front, typename... Rest>
struct type_list_at_impl<type_list_impl<Front, Rest...>, 0> {
  using type = Front;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_back back
/// tatooine::type_list_back returns the last element of a tatooine::type_list_impl.
/// \ingroup template_meta_programming_type_list
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
struct type_list_back_impl<type_list_impl<Types...>> {
  using type = variadic::back_type<Types...>;
};
//------------------------------------------------------------------------------
template <>
struct type_list_back_impl<type_list_impl<>> {
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_front front
/// tatooine::type_list_back returns the first element of a tatooine::type_list_impl.
/// \ingroup template_meta_programming_type_list
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
struct type_list_front_impl<type_list_impl<Types...>> {
  using type = variadic::front_type<Types...>;
};
//------------------------------------------------------------------------------
template <>
struct type_list_front_impl<type_list_impl<>> {
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_push_back push_back
/// Using tatooine::type_list_push_back one can append a new type at the back of
/// a tatooine::type_list_impl.
/// \ingroup template_meta_programming_type_list
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
struct type_list_push_back_impl<type_list_impl<Types...>, NewBack> {
  using type = type_list_impl<Types..., NewBack>;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_push_front push_front
/// Using tatooine::type_list_push_back one can append a new type at the front
/// of a tatooine::type_list_impl.
/// \ingroup template_meta_programming_type_list
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
struct type_list_push_front_impl<type_list_impl<Types...>, NewFront> {
  using type = type_list_impl<NewFront, Types...>;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_pop_back pop_back
/// Using tatooine::type_list_push_back one can remove the back
/// of a tatooine::type_list_impl.
/// \ingroup template_meta_programming_type_list
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
struct type_list_pop_back_impl<type_list_impl<T0, T1, Rest...>, TypesAccumulator...> {
  using type = typename type_list_pop_back_impl<type_list_impl<T1, Rest...>,
                                                TypesAccumulator..., T0>::type;
};
//------------------------------------------------------------------------------
template <typename T, typename... TypesAccumulator>
struct type_list_pop_back_impl<type_list_impl<T>, TypesAccumulator...> {
  using type = type_list_impl<TypesAccumulator...>;
};
//------------------------------------------------------------------------------
template <typename... TypesAccumulator>
struct type_list_pop_back_impl<type_list_impl<>, TypesAccumulator...> {
  using type = type_list_impl<TypesAccumulator...>;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_pop_front pop_front
/// Using tatooine::type_list_pop_front one can remove the front
/// of a tatooine::type_list_impl.
/// \ingroup template_meta_programming_type_list
/// \{
//==============================================================================
template <typename TypeList>
struct type_list_pop_front_impl;
//------------------------------------------------------------------------------
template <typename TypeList>
using type_list_pop_front = typename type_list_pop_front_impl<TypeList>::type;
//------------------------------------------------------------------------------
template <typename Front, typename... Back>
struct type_list_pop_front_impl<type_list_impl<Front, Back...>> {
  using type = type_list_impl<Back...>;
};
//------------------------------------------------------------------------------
template <>
struct type_list_pop_front_impl<type_list_impl<>> {
  using type = type_list_impl<>;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_list_contains contains
/// \ingroup template_meta_programming_type_list
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
struct type_list_contains_impl<type_list_impl<SetHead, SetRest...>, T> {
  static auto constexpr value =
      type_list_contains_impl<type_list_impl<SetRest...>, T>::value;
};
//------------------------------------------------------------------------------
template <typename SetHead, typename... SetRest>
struct type_list_contains_impl<type_list_impl<SetHead, SetRest...>,
                                SetHead> : std::true_type {};
//------------------------------------------------------------------------------
template <typename T>
struct type_list_contains_impl<type_list_impl<>, T> : std::false_type {};
//==============================================================================
/// \}
//==============================================================================
/// \brief An empty struct that holds types.
/// \tparam Ts Variadic list of types.
template <typename... Ts>
struct type_list_impl {
  using this_type = type_list_impl<Ts...>;

  using front = type_list_front<this_type>;
  using back = type_list_back<this_type>;
  template <typename T>
  using push_back = type_list_push_back<this_type, T>;
  template <typename T>
  using push_front = type_list_push_front<this_type, T>;

  using pop_back = type_list_pop_back<this_type>;
  using pop_front = type_list_pop_front<this_type>;

  template <typename T>
  static bool constexpr contains = type_list_contains<this_type, T>;

  static auto constexpr size = type_list_size<this_type>;

  static bool constexpr empty = size == 0;

  template <std::size_t I>
  using at = type_list_at<this_type, I>;
};
//==============================================================================
/// \brief An empty struct that holds types.
/// \tparam Ts Variadic list of types.
/// 
/// En empty list cannot be popped nor has it a front or a back.
template <>
struct type_list_impl<> {
  using this_type = type_list_impl<>;

  template <typename T>
  using push_back = type_list_push_back<this_type, T>;

  template <typename T>
  using push_front = type_list_push_front<this_type, T>;

  template <typename T>
  static bool constexpr contains = type_list_contains<this_type, T>;

  static auto constexpr size = 0;

  static bool constexpr empty = true;

  template <std::size_t I>
  using at = type_list_at<this_type, I>;
};
//==============================================================================
/// Constructor for tatooine::type_list_impl
template <typename... Ts>
using type_list = type_list_impl<Ts...>;
/// \}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
