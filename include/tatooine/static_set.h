#ifndef TATOOINE_STATIC_SET_H
#define TATOOINE_STATIC_SET_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename... Ts>
struct static_type_set_impl {};
//==============================================================================
template <typename Set, std::size_t Cnt>
struct static_set_size_impl;
//------------------------------------------------------------------------------
template <typename Set>
static auto constexpr static_set_size = static_set_size_impl<Set, 0>::value;
//------------------------------------------------------------------------------
template <typename SetHead, typename... SetRest, std::size_t Cnt>
struct static_set_size_impl<static_type_set_impl<SetHead, SetRest...>, Cnt> {
  static auto constexpr value =
      static_set_size_impl<static_type_set_impl<SetRest...>, Cnt + 1>::value;
};
//------------------------------------------------------------------------------
template <std::size_t Cnt>
struct static_set_size_impl<static_type_set_impl<>, Cnt> {
  static auto constexpr value = Cnt;
};
//==============================================================================
template <typename Set, typename NewType, typename... TypesAccumulator>
struct insert_into_static_type_set_impl;
//==============================================================================
/// Head and NewType are not equal -> continue iterating
template <typename SetHead, typename... SetRest, typename NewType,
          typename... TypesAccumulator>
struct insert_into_static_type_set_impl<
    static_type_set_impl<SetHead, SetRest...>, NewType, TypesAccumulator...> {
  using type = typename insert_into_static_type_set_impl<
      static_type_set_impl<SetRest...>, NewType, TypesAccumulator...,
      SetHead>::type;
};
//------------------------------------------------------------------------------
/// Head and NewType are not equal -> continue iterating
template <typename SetHead, typename... SetRest, typename... TypesAccumulator>
struct insert_into_static_type_set_impl<
    static_type_set_impl<SetHead, SetRest...>, SetHead, TypesAccumulator...> {
  using type = static_type_set_impl<TypesAccumulator..., SetHead, SetRest...>;
};
//------------------------------------------------------------------------------
/// type_set is empty -> insert new type into set
template <typename NewType, typename... TypesAccumulator>
struct insert_into_static_type_set_impl<static_type_set_impl<>, NewType,
                                        TypesAccumulator...> {
  using type = static_type_set_impl<TypesAccumulator..., NewType>;
};

template <typename Set, typename NewType>
using insert_into_static_type_set =
    typename insert_into_static_type_set_impl<Set, NewType>::type;
//==============================================================================
template <typename Set, typename... Ts>
struct static_type_set_constructor;
//------------------------------------------------------------------------------
template <typename Set, typename T, typename... Ts>
struct static_type_set_constructor<Set, T, Ts...> {
  using type =
      typename static_type_set_constructor<insert_into_static_type_set<Set, T>,
                                           Ts...>::type;
};
//------------------------------------------------------------------------------
template <typename Set>
struct static_type_set_constructor<Set> {
  using type = Set;
};
//------------------------------------------------------------------------------
template <typename... Ts>
using static_type_set =
    typename static_type_set_constructor<static_type_set_impl<>, Ts...>::type;
//==============================================================================
template <typename Set, typename T>
struct static_set_includes_impl;
//------------------------------------------------------------------------------
template <typename Set, typename T>
static auto constexpr static_set_includes =
    static_set_includes_impl<Set, T>::value;
//------------------------------------------------------------------------------
template <typename SetHead, typename... SetRest, typename T>
struct static_set_includes_impl<static_type_set_impl<SetHead, SetRest...>, T> {
  static auto constexpr value =
      static_set_includes_impl<static_type_set_impl<SetRest...>, T>::value;
};
//------------------------------------------------------------------------------
template <typename SetHead, typename... SetRest>
struct static_set_includes_impl<static_type_set_impl<SetHead, SetRest...>,
                                SetHead> : std::true_type {};
//------------------------------------------------------------------------------
template <typename T>
struct static_set_includes_impl<static_type_set_impl<>, T> : std::false_type {};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
