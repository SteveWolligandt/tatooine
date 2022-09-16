#ifndef TATOOINE_VARIADIC_HELPERS_H
#define TATOOINE_VARIADIC_HELPERS_H
//==============================================================================
#include <cstdint>
#include <tatooine/tuple.h>
#include <tatooine/concepts.h>
#include <tatooine/variadic_helpers/ith_type.h>
//==============================================================================
namespace tatooine::variadic {
//==============================================================================
template <std::size_t I, std::size_t... Is>
struct front_number_impl {
  static auto constexpr value = I;
};
//------------------------------------------------------------------------------
template <std::size_t... Is>
static auto constexpr front_number = front_number_impl<Is...>::value;
//==============================================================================
template <std::size_t... I>
struct back_number_impl;
//------------------------------------------------------------------------------
template <std::size_t I>
struct back_number_impl<I> {
  static auto constexpr value = I;
};
//------------------------------------------------------------------------------
template <std::size_t I, std::size_t... Is>
struct back_number_impl<I, Is...> {
  static auto constexpr value = back_number_impl<Is...>::value;
};
//------------------------------------------------------------------------------
template <std::size_t... Is>
static auto constexpr back_number = back_number_impl<Is...>::value;
//==============================================================================
template <typename T, typename... Ts>
struct front_type_impl {
  using type = T;
};
//------------------------------------------------------------------------------
template <typename... Ts>
using front_type = typename front_type_impl<Ts...>::type;
//==============================================================================
template <typename... T>
struct back_type_impl;
//------------------------------------------------------------------------------
template <typename T>
struct back_type_impl<T> {
  using type = T;
};
//------------------------------------------------------------------------------
template <typename T, typename... Ts>
struct back_type_impl<T, Ts...> {
  using type = typename back_type_impl<Ts...>::type;
};
//------------------------------------------------------------------------------
template <typename... Ts>
using back_type = typename back_type_impl<Ts...>::type;
//==============================================================================
template <std::size_t I, std::size_t CurNum, std::size_t... RestNums>
struct ith_number_impl {
  static auto constexpr value = ith_number_impl<I - 1, RestNums...>::value;
};
template <std::size_t CurNum, std::size_t... RestNums>
struct ith_number_impl<0, CurNum, RestNums...> {
  static auto constexpr value = CurNum;
};
template <std::size_t I, std::size_t... Nums>
[[maybe_unused]] static auto constexpr ith_number =
    ith_number_impl<I, Nums...>::value;
//==============================================================================
template <std::size_t X, std::size_t... Rest>
struct contains_impl;
template <std::size_t X, std::size_t I, std::size_t... Rest>
struct contains_impl<X, I, Rest...>
    : std::integral_constant<bool, contains_impl<X, Rest...>::value> {};
template <std::size_t X, std::size_t... Rest>
struct contains_impl<X, X, Rest...> : std::true_type {};
template <std::size_t X>
struct contains_impl<X> : std::false_type {};
template <std::size_t X, std::size_t... Is>
static constexpr auto contains = contains_impl<X, Is...>::value;
//==============================================================================
/// Extracts variadic data into an array in the Range of Begin and End.
template <std::size_t I, std::size_t Begin, std::size_t End>
constexpr auto extract_to_array(auto& extracted_data) -> auto& {
    return extracted_data;
}
//==============================================================================
/// Extracts variadic data into an array in the Range of Begin and End.
template <std::size_t I, std::size_t Begin, std::size_t End,
          typename T, typename... Ts>
requires(Begin < End) && (I <= End)
constexpr auto extract_to_array(auto& extracted_data,
                       T&& t, [[maybe_unused]] Ts&&... ts) -> auto& {
  if constexpr (I == End) {
    return extracted_data;
  } else {
    if constexpr (I >= Begin) {
      extracted_data[I - Begin] = std::forward<T>(t);
    }
    return extract_to_array<I + 1, Begin, End>(extracted_data, std::forward<Ts>(ts)...);
  }
}
//==============================================================================
/// Extracts variadic data into an array in the Range of Begin and End.
template <std::size_t I, std::size_t Begin, std::size_t End>
constexpr auto extract_to_tuple(auto& extracted_data) -> auto& {
    return extracted_data;
}
//------------------------------------------------------------------------------
/// Extracts variadic data into an array in the Range of Begin and End.
template <std::size_t I, std::size_t Begin, std::size_t End,
          typename T, typename... Ts>
requires(Begin < End) && (I == End)
constexpr auto extract_to_tuple(auto& extracted_data) -> auto& {
    return extracted_data;
}
//------------------------------------------------------------------------------
/// Extracts variadic data into an array in the Range of Begin and End.
template <std::size_t I, std::size_t Begin, std::size_t End,
          typename T, typename... Ts>
requires(Begin < End) && (I <= End)
constexpr auto extract_to_tuple(auto& extracted_data,
                       T&& t, [[maybe_unused]] Ts&&... ts) -> auto& {
  if constexpr (I == End) {
    return extracted_data;
  } else {
    if constexpr (I >= Begin) {
      extracted_data.template at<I - Begin>() = std::forward<T>(t);
    }
    return extract_to_tuple<I + 1, Begin, End>(extracted_data, std::forward<Ts>(ts)...);
  }
}
//------------------------------------------------------------------------------
template <std::size_t I, std::size_t Begin, std::size_t End, typename... Ts>
struct extract_helper_tuple_impl;
//------------------------------------------------------------------------------
/// Iterate all the way until I == Begin
template <std::size_t I, std::size_t Begin, std::size_t End, typename T,
          typename... Ts>
requires(I < Begin)
struct extract_helper_tuple_impl<I, Begin, End, T, Ts...> {
  using type =
      typename extract_helper_tuple_impl<I + 1, Begin, End, Ts...>::type;
};
//------------------------------------------------------------------------------
/// Start concatenating types into tuple
template <std::size_t I, std::size_t Begin, std::size_t End, typename T0, typename T1,
          typename... Ts>
requires(I >= Begin && I < End-1)
struct extract_helper_tuple_impl<I, Begin, End, T0, T1, Ts...> {
  using type = tuple_concat_types<
      tuple<T0>,
      typename extract_helper_tuple_impl<I + 1, Begin, End, T1, Ts...>::type>;
};
//------------------------------------------------------------------------------
/// Start concatenating types into tuple
template <std::size_t I, std::size_t Begin, std::size_t End, typename T,
          typename... Ts>
requires(I == End - 1)
struct extract_helper_tuple_impl<I, Begin, End, T, Ts...> {
  using type = tuple<T>;
};
//------------------------------------------------------------------------------
template <std::size_t Begin, std::size_t End, typename... Ts>
using extract_helper_tuple =
    typename extract_helper_tuple_impl<0, Begin, End, Ts...>::type;
//------------------------------------------------------------------------------
/// Extracts variadic data into an array in the Range of Begin and End.
template <std::size_t Begin, std::size_t End, typename T, typename... Ts>
requires (Begin < End)
constexpr auto extract(T&&t, Ts&&... ts) {
  if constexpr ((std::is_same_v<std::decay_t<T>, std::decay_t<Ts>>&&...)) {
    auto extracted_data =
        std::array<std::decay_t<T>, End - Begin>{};
    return extract_to_array<0, Begin, End>(extracted_data,
                                           std::forward<T>(t),
                                           std::forward<Ts>(ts)...);
  } else {
    auto extracted_data = extract_helper_tuple<Begin, End, std::decay_t<T>, std::decay_t<Ts>...>{};
    return extract_to_tuple<0, Begin, End>(extracted_data,
                                           std::forward<T>(t),
                                           std::forward<Ts>(ts)...);
  }
}
//==============================================================================
}  // namespace tatooine::variadic
//==============================================================================
#endif
