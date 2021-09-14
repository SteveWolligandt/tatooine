#ifndef TATOOINE_STATIC_SET_H
#define TATOOINE_STATIC_SET_H
//==============================================================================
#include <tatooine/type_list.h>

#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \defgroup type_set Set
/// \ingroup template_meta_programming
/// A type set can be constructed with tatooine::type_set which constructs a
/// tatooine::type_list with no redundant types.
/// \{
//==============================================================================
/// \defgroup type_set_insert insert
/// \ingroup type_set
/// \{
//==============================================================================
template <typename TypeList, typename NewType, typename... TypesAccumulator>
struct type_set_insert_impl;
/// Head and NewType are not equal -> continue iterating
template <typename SetHead, typename... SetRest, typename NewType,
          typename... TypesAccumulator>
struct type_set_insert_impl<type_list<SetHead, SetRest...>, NewType,
                            TypesAccumulator...> {
  using type =
      typename type_set_insert_impl<type_list<SetRest...>, NewType,
                                    TypesAccumulator..., SetHead>::type;
};
//------------------------------------------------------------------------------
/// Head and NewType are equal -> do not insert and stop
template <typename SetHead, typename... SetRest, typename... TypesAccumulator>
struct type_set_insert_impl<type_list<SetHead, SetRest...>, SetHead,
                            TypesAccumulator...> {
  using type = type_list<TypesAccumulator..., SetHead, SetRest...>;
};
//------------------------------------------------------------------------------
/// type_set is empty -> insert new type into set
template <typename NewType, typename... TypesAccumulator>
struct type_set_insert_impl<type_list<>, NewType, TypesAccumulator...> {
  using type = type_list<TypesAccumulator..., NewType>;
};

template <typename TypeList, typename NewType>
using type_set_insert = typename type_set_insert_impl<TypeList, NewType>::type;
/// \}
//==============================================================================
/// \defgroup type_set_constructor constructor
/// \ingroup type_set
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
template <typename TypeList>
struct type_set_constructor<TypeList> {
  using type = TypeList;
};
//------------------------------------------------------------------------------
/// Inherits from a type_list with only unique types.
template <typename... Ts>
struct type_set : type_set_constructor<type_list<>, Ts...>::type {
  using this_t   = type_set<Ts...>;
  using parent_t = typename type_set_constructor<type_list<>, Ts...>::type;
  template <typename T>
  using insert = type_set_insert<parent_t, T>;
};
//==============================================================================
/// \}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
