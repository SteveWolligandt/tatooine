#ifndef __TATOOINE_TEMPLATE_HELPER_H__
#define __TATOOINE_TEMPLATE_HELPER_H__

#include <type_traits>

//==============================================================================
namespace tatooine {
namespace temp_helper {
//==============================================================================

template <typename T, typename... Ts>
struct front_t {
  using type = T;
};
template <typename... Ts>
using front = typename front_t<Ts...>::type;



//==============================================================================

template <typename... Ts>
struct back_t;

template <typename T>
struct back_t<T> {
  using type = T;
};

template <typename T0, typename T1, typename... Ts>
struct back_t<T0, T1, Ts...> {
  using type = typename back_t<T1, Ts...>::type;
};

template <typename... Ts>
using back = typename back_t<Ts...>::type;

//==============================================================================
template <std::size_t i, typename... Ts>
struct get_t;

template <typename T, typename... Ts>
struct get_t<0, T, Ts...> {
  using type = T;
};

template <std::size_t i, typename T, typename... Ts>
struct get_t<i, T, Ts...> {
  using type = typename get_t<i - 1, Ts...>::type;
};

template <std::size_t i, typename... Ts>
using get = typename get_t<i, Ts...>::type;

//==============================================================================
template <std::size_t i, typename T, T... Vs>
struct getval_t;

template <typename T, T V0, T... Vs>
struct getval_t<0, T, V0, Vs...> {
  static constexpr T value = V0;
};

template <std::size_t i, typename T, T V0, T... Vs>
struct getval_t<i, T,V0, Vs...> {
  static constexpr T value = getval_t<i - 1, T, Vs...>::value;
};

template <std::size_t i, typename T, T... Vs>
static constexpr auto getval = getval_t<i, T, Vs...>::value;

//==============================================================================

template <typename... Ts>
struct flipped {};

template <typename... Ts>
struct flip_t;

template <typename T0, typename T1, typename... Ts>
struct flip_t<T0, T1, Ts...> {
  template <typename... Flipped_Ts>
  static constexpr auto flip(flipped<Flipped_Ts...>&&) {
    return flipped<Flipped_Ts..., T0>{};
  }
  static constexpr auto flip() { return flip(flip_t<T1, Ts...>::flip()); }
};

template <typename T>
struct flip_t<T> {
  static constexpr auto flip() { return flipped<T>{}; }
};

template <typename... Ts>
inline auto flip() {
  return flip_t<Ts...>::flip();
}

//==============================================================================

template <typename... Ts>
struct all_same_t;

template <typename T>
struct all_same_t<T> {
  static constexpr bool value = true;
};

template <typename T0, typename T1, typename... Ts>
struct all_same_t<T0, T1, Ts...> {
  static constexpr bool value =
      std::is_same<T0, T1>::value && all_same_t<T1, Ts...>::value;
};

template <typename... Ts>
static constexpr bool all_same = all_same_t<Ts...>::value;

//==============================================================================
}  // namespace temp_helper
}  // namespace tatooine
//==============================================================================

#endif
