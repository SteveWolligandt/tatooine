#ifndef TATOOINE_STATIC_SET_H
#define TATOOINE_STATIC_SET_H
//==============================================================================
#include <tatooine/type_list.h>

#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \defgroup template_meta_programming_type_set Set
/// \ingroup template_meta_programming
/// A type set can be constructed with tatooine::type_set which constructs a
/// tatooine::type_set_impl with no redundant types:
/// ```cpp
///  using list = tatooine::type_set<int, double, int, float>;
/// ```
/// `list` is of type tatooine::type_set_impl and holds `int`, `double` and
/// `float`.
/// \{
template <typename... Ts>
struct type_set_impl;
//==============================================================================
/// \addtogroup type_list_size
/// \ingroup type_list
/// \{
//==============================================================================
/// Size of a tatooine::type_set_impl
template <typename... Types>
struct type_list_size_impl<type_set_impl<Types...>>
    : std::integral_constant<std::size_t, sizeof...(Types)> {};
//==============================================================================
/// \}
//==============================================================================
/// \addtogroup type_list_at
/// \ingroup type_list
/// \{
//==============================================================================
/// Access to the Ith element of TypeList
template <typename... Types, std::size_t I>
struct type_list_at_impl<type_set_impl<Types...>, I> {
  using type = typename type_list_at_impl<type_list<Types...>, I>::type;
};
//==============================================================================
/// \}
//==============================================================================
/// \addtogroup type_list_contains
/// \ingroup type_list
/// \{
//==============================================================================
template <typename... Ts, typename T>
struct type_list_contains_impl<type_set_impl<Ts...>, T> {
  static auto constexpr value =
      type_list_contains_impl<type_list<Ts...>, T>::value;
};
//==============================================================================
/// \}
//==============================================================================
/// \defgroup type_set_insert insert
/// \ingroup template_meta_programming_type_set
/// \{
//==============================================================================
template <typename TypeList, typename NewType, typename... TypesAccumulator>
struct type_set_insert_impl;
//------------------------------------------------------------------------------
/// Head and NewType are not equal -> continue iterating
template <typename SetHead, typename... SetRest, typename NewType,
          typename... TypesAccumulator>
struct type_set_insert_impl<type_list<SetHead, SetRest...>, NewType,
                            TypesAccumulator...> {
  using type =
      typename type_set_insert_impl<type_list<SetRest...>, NewType,
                                    TypesAccumulator..., SetHead>::type;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// Head and NewType are not equal -> continue iterating
template <typename SetHead, typename... SetRest, typename NewType,
          typename... TypesAccumulator>
struct type_set_insert_impl<type_set_impl<SetHead, SetRest...>, NewType,
                            TypesAccumulator...> {
  using type =
      typename type_set_insert_impl<type_set_impl<SetRest...>, NewType,
                                    TypesAccumulator..., SetHead>::type;
};
//------------------------------------------------------------------------------
/// Head and NewType are equal -> do not insert and stop
template <typename SetHead, typename... SetRest, typename... TypesAccumulator>
struct type_set_insert_impl<type_list<SetHead, SetRest...>, SetHead,
                            TypesAccumulator...> {
  using type = type_list<TypesAccumulator..., SetHead, SetRest...>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// Head and NewType are equal -> do not insert and stop
template <typename SetHead, typename... SetRest, typename... TypesAccumulator>
struct type_set_insert_impl<type_set_impl<SetHead, SetRest...>, SetHead,
                            TypesAccumulator...> {
  using type = type_set_impl<TypesAccumulator..., SetHead, SetRest...>;
};
//------------------------------------------------------------------------------
/// type_set is empty -> insert new type into set
template <typename NewType, typename... TypesAccumulator>
struct type_set_insert_impl<type_list<>, NewType, TypesAccumulator...> {
  using type = type_list<TypesAccumulator..., NewType>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename NewType, typename... TypesAccumulator>
struct type_set_insert_impl<type_set_impl<>, NewType, TypesAccumulator...> {
  using type = type_set_impl<TypesAccumulator..., NewType>;
};
//------------------------------------------------------------------------------
template <typename TypeList, typename NewType>
using type_set_insert = typename type_set_insert_impl<TypeList, NewType>::type;
/// \}
//==============================================================================
/// \defgroup type_set_constructor constructor
/// \ingroup template_meta_programming_type_set
/// \{
template <typename TypeList, typename... Ts>
struct type_set_constructor;
//------------------------------------------------------------------------------
template <typename TypeList, typename T, typename... Ts>
struct type_set_constructor<TypeList, T, Ts...> {
  using type =
      typename type_set_constructor<type_set_insert<TypeList, T>, Ts...>::type;
};
//------------------------------------------------------------------------------
template <typename ...Ts>
struct type_set_constructor<type_list<Ts...>> {
  using type = type_set_impl<Ts...>;
};
//------------------------------------------------------------------------------
template <typename... Ts>
using type_set = typename type_set_constructor<type_list<>, Ts...>::type;
//==============================================================================
/// \}
//==============================================================================
/// Inherits from a type_list with only unique types.
template <typename... Ts>
struct type_set_impl : type_list<Ts...> {
  using this_type   = type_set_impl<Ts...>;
  template <typename T>
  using insert = type_set_insert<this_type, T>;
};
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
