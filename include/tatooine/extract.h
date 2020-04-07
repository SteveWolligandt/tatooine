#ifndef TATOOINE_EXTRACT_H
#define TATOOINE_EXTRACT_H

#include "make_array.h"
#include "variadic_helpers.h"
//==============================================================================
namespace tatooine {
//==============================================================================
#if has_cxx17_support()
template <size_t I, size_t Begin, size_t End, typename Cont>
constexpr auto extract(Cont& extracted_data) -> auto& {
  return extracted_data;
}
//------------------------------------------------------------------------------
template <size_t I, size_t Begin, size_t End, typename Cont, typename T,
          typename... Ts>
constexpr auto extract(Cont& extracted_data, T&& t, Ts&&... ts) -> auto& {
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
#endif
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
