#ifndef TATOOINE_TYPE_COUNTER_H
#define TATOOINE_TYPE_COUNTER_H
//==============================================================================
#include <tatooine/type_number_pair.h>
#include <tatooine/type_set.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \addtogroup type_counting Type Counting
/// \ingroup template_meta_programming
///
/// This module is for counting types of a variadic list.
/// \{
//==============================================================================
template <typename Counter, typename T>
struct type_counter_get_count_impl;
//------------------------------------------------------------------------------
template <typename Counter, typename T>
static auto constexpr type_counter_get_count =
    type_counter_get_count_impl<Counter, T>::value;
//------------------------------------------------------------------------------
template <typename HeadCounter, typename... Counters, typename T>
struct type_counter_get_count_impl<type_list_impl<HeadCounter, Counters...>, T> {
  static auto constexpr value =
      type_counter_get_count_impl<type_list_impl<Counters...>, T>::value;
};
//------------------------------------------------------------------------------
template <typename HeadCounter, typename... Counters, typename T>
struct type_counter_get_count_impl<type_set_impl<HeadCounter, Counters...>, T> {
  static auto constexpr value =
      type_counter_get_count_impl<type_list_impl<Counters...>, T>::value;
};
//------------------------------------------------------------------------------
template <std::size_t N, typename... Counters, typename T>
struct type_counter_get_count_impl<
    type_list_impl<type_number_pair<T, N>, Counters...>, T> {
  static auto constexpr value = N;
};
//------------------------------------------------------------------------------
template <typename T>
struct type_counter_get_count_impl<type_list_impl<>, T> {
  // static_assert(false, "T not found in counter.");
  static auto constexpr value = 0;
};
//------------------------------------------------------------------------------
template <std::size_t N, typename... Counters, typename T>
struct type_counter_get_count_impl<
    type_set_impl<type_number_pair<T, N>, Counters...>, T> {
  static auto constexpr value = N;
};
//------------------------------------------------------------------------------
template <typename T>
struct type_counter_get_count_impl<type_set_impl<>, T> {
  // static_assert(false, "T not found in counter.");
  static auto constexpr value = 0;
};
//==============================================================================
template <typename Counter, typename T>
struct type_counter_increase_if_equal_impl;
//------------------------------------------------------------------------------
template <typename T, std::size_t N, typename OtherT>
struct type_counter_increase_if_equal_impl<type_number_pair<T, N>, OtherT> {
  using type = type_number_pair<T, N>;
};
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
struct type_counter_increase_if_equal_impl<type_number_pair<T, N>, T> {
  using type = type_number_pair<T, N + 1>;
};
//------------------------------------------------------------------------------
template <typename Counter, typename T>
using type_counter_increase_if_equal =
    typename type_counter_increase_if_equal_impl<Counter, T>::type;
//==============================================================================
template <typename Counter, typename... Ts>
struct type_counter_insert_impl;

template <typename... Counters, typename Head, typename... Rest>
struct type_counter_insert_impl<type_list_impl<Counters...>, Head, Rest...> {
  using type = typename type_counter_insert_impl<
      type_list_impl<type_counter_increase_if_equal<Counters, Head>...>,
      Rest...>::type;
};

template <typename... Counters>
struct type_counter_insert_impl<type_list_impl<Counters...>> {
  using type = type_list<Counters...>;
};
template <typename... Counters, typename Head, typename... Rest>
struct type_counter_insert_impl<type_set_impl<Counters...>, Head, Rest...> {
  using type = typename type_counter_insert_impl<
      type_set_impl<type_counter_increase_if_equal<Counters, Head>...>,
      Rest...>::type;
};

template <typename... Counters>
struct type_counter_insert_impl<type_set_impl<Counters...>> {
  using type = type_set<Counters...>;
};

template <typename Counter, typename... Ts>
using type_counter_insert =
    typename type_counter_insert_impl<Counter, Ts...>::type;
//==============================================================================
template <typename StaticTypeSet, typename... Ts>
struct count_types_impl;

template <typename... UniqueTypes, typename... Ts>
struct count_types_impl<type_list_impl<UniqueTypes...>, Ts...> {
  using type =
      type_counter_insert<type_list_impl<type_number_pair<UniqueTypes, 0>...>,
                          Ts...>;
};
template <typename... UniqueTypes, typename... Ts>
struct count_types_impl<type_set_impl<UniqueTypes...>, Ts...> {
  using type =
      type_counter_insert<type_set_impl<type_number_pair<UniqueTypes, 0>...>,
                          Ts...>;
};
template <typename... Ts>
using count_types = typename count_types_impl<type_list<Ts...>, Ts...>::type;
/// \}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
