#ifndef TATOOINE_UTILITY_H
#define TATOOINE_UTILITY_H

#include <array>
#include <boost/core/demangle.hpp>

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename T>
struct internal_data_type;

template <>
struct internal_data_type<double> {
  using type = double;
};

template <>
struct internal_data_type<float> {
  using type = float;
};

//==============================================================================
template <typename T>
using internal_data_type_t = typename internal_data_type<T>::type;

/// creates an index_sequence and removes an element from it
template <size_t Omit, size_t... Is, size_t... Js>
constexpr auto sliced_indices(std::index_sequence<Is...>,
                              std::index_sequence<Js...>) {
  std::array indices{Is...};
  (++indices[Js + Omit], ...);
  return indices;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// creates an index sequence and removes an element from it
template <size_t N, size_t Omit>
constexpr auto sliced_indices() {
  return sliced_indices<Omit>(std::make_index_sequence<N - 1>{},
                              std::make_index_sequence<N - Omit - 1>{});
}
//==============================================================================
template <typename F, typename... Ts>
void for_each(F&& f, Ts&&... ts) {
  (f(std::forward<Ts>(ts)), ...);
}

//==============================================================================
template <typename T, typename... Ts>
struct front {
  using type = T;
};
template <typename... Ts>
using front_t = typename front<Ts...>::type;

//==============================================================================
template <typename... T>
struct back;
template <typename T>
struct back<T> {
  using type = T;
};
template <typename T, typename... Ts>
struct back<T, Ts...> {
  using type = typename back<Ts...>::type;
};
template <typename... Ts>
using back_t = typename back<Ts...>::type;

//==============================================================================
template <typename T, size_t... Is>
constexpr auto make_array(const T& t, std::index_sequence<Is...> /*is*/) {
  return std::array<T, sizeof...(Is)>{((void)Is, t)...};
}
//------------------------------------------------------------------------------
template <typename T, size_t N>
constexpr auto make_array() {
  return make_array<T>(T{}, std::make_index_sequence<N>{});
}
//------------------------------------------------------------------------------
template <typename T, size_t N>
constexpr auto make_array(const T& t) {
  return make_array<T>(t, std::make_index_sequence<N>{});
}

//==============================================================================
template <size_t I, size_t Begin, size_t End, typename Cont>
constexpr auto& extract(Cont& extracted_data) {
  return extracted_data;
}
//------------------------------------------------------------------------------
template <size_t I, size_t Begin, size_t End, typename Cont, typename T,
          typename... Ts>
constexpr auto& extract(Cont& extracted_data, T&& t, Ts&&... ts) {
  static_assert(Begin <= End);
  if constexpr (I > End) { return extracted_data; }
  if constexpr (Begin >= I) { extracted_data[I - Begin] = t; }
  return extract<I + 1, Begin, End>(extracted_data, std::forward<Ts>(ts)...);
}
//------------------------------------------------------------------------------
template <size_t Begin, size_t End, typename... Ts>
constexpr auto extract(Ts&&... ts) {
  static_assert(Begin <= End);
  auto extracted_data =
      make_array<std::decay_t<front_t<Ts...>>, End - Begin + 1>();
  return extract<0, Begin, End>(extracted_data, std::forward<Ts>(ts)...);
}

//==============================================================================
/// returns demangled typename
template <typename T>
inline std::string type_name(T&& /*t*/) {
  return boost::core::demangle(typeid(T).name());
}

//------------------------------------------------------------------------------
/// returns demangled typename
template <typename T>
inline std::string type_name() {
  return boost::core::demangle(typeid(T).name());
}

//==============================================================================
inline constexpr auto debug_mode() {
#ifndef NDEBUG
  return true;
#else
  return false;
#endif
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
inline constexpr auto release_mode() {
#ifdef NDEBUG
  return true;
#else
  return false;
#endif
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
