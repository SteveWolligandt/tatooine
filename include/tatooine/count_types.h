#ifndef TATOOINE_COUNT_TYPES_H
#define TATOOINE_COUNT_TYPES_H
//==============================================================================
#include <tatooine/static_set.h>
#include <tatooine/type_number_pair.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename... Counters>
struct type_counter_impl {};
//==============================================================================
template <typename Counter, typename T>
struct get_count_impl;
//------------------------------------------------------------------------------
template <typename Counter, typename T>
static auto constexpr get_count = get_count_impl<Counter, T>::value;
//------------------------------------------------------------------------------
template <typename HeadCounter, typename... Counters, typename T>
struct get_count_impl<type_counter_impl<HeadCounter, Counters...>, T> {
  static auto constexpr value =
      get_count_impl<type_counter_impl<Counters...>, T>::value;
};
//------------------------------------------------------------------------------
template <std::size_t N, typename... Counters, typename T>
struct get_count_impl<type_counter_impl<type_number_pair<T, N>, Counters...>,
                      T> {
  static auto constexpr value = N;
};
//------------------------------------------------------------------------------
template <typename T>
struct get_count_impl<type_counter_impl<>, T> {
  // static_assert(false, "T not found in counter.");
  static auto constexpr value = 0;
};
//==============================================================================
template <typename Counter, typename T>
struct increase_type_counter_if_equal_impl;
//------------------------------------------------------------------------------
template <typename T, std::size_t N, typename OtherIndex>
struct increase_type_counter_if_equal_impl<type_number_pair<T, N>, OtherIndex> {
  using type = type_number_pair<T, N>;
};
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
struct increase_type_counter_if_equal_impl<type_number_pair<T, N>, T> {
  using type = type_number_pair<T, N + 1>;
};
//------------------------------------------------------------------------------
template <typename Counter, typename T>
using increase_if_equal =
    typename increase_type_counter_if_equal_impl<Counter, T>::type;
//==============================================================================
template <typename Counter, typename... Indices>
struct insert_types_into_counter_impl;

template <typename... Counters, typename HeadIndex, typename... Indices>
struct insert_types_into_counter_impl<type_counter_impl<Counters...>, HeadIndex,
                                      Indices...> {
  using type = typename insert_types_into_counter_impl<
      type_counter_impl<increase_if_equal<Counters, HeadIndex>...>,
      Indices...>::type;
};

template <typename... Counters>
struct insert_types_into_counter_impl<type_counter_impl<Counters...>> {
  using type = type_counter_impl<Counters...>;
};

template <typename Counter, typename... Indices>
using insert_types_into_counter =
    typename insert_types_into_counter_impl<Counter, Indices...>::type;
//==============================================================================
template <typename StaticTypeSet, typename... Indices>
struct count_types_impl;

template <typename... SetTypes, typename... Indices>
struct count_types_impl<static_type_set_impl<SetTypes...>, Indices...> {
  using type = insert_types_into_counter<
      type_counter_impl<type_number_pair<SetTypes, 0>...>, Indices...>;
};

template <typename... Indices>
using count_types =
    typename count_types_impl<static_type_set<Indices...>, Indices...>::type;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
