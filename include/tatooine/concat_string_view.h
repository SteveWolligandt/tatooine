#ifndef TATOOINE_CONCAT_STRING_VIEW_H
#define TATOOINE_CONCAT_STRING_VIEW_H
//==============================================================================
#include <string_view>
#include <utility>
//==============================================================================
namespace tatooine {
//==============================================================================
template <const char... cs>
struct c_str_assembler {
  static constexpr const char c_str[] = {cs..., '\0'};
  static constexpr std::string_view value {c_str};
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <const char... cs>
static constexpr std::string_view c_str_assembler_v = c_str_assembler<cs...>::value;
//==============================================================================
template <std::string_view const& S0, std::string_view const& S1,
          std::size_t... I0s, std::size_t... I1s>
constexpr std::string_view const& concat(std::index_sequence<I0s...>,
                      std::index_sequence<I1s...>) {
    return c_str_assembler_v<S0[I0s]..., S1[I1s]...>;
}
//------------------------------------------------------------------------------
template <std::string_view const& S>
constexpr std::string_view const& concat() {
    return S;
}
//------------------------------------------------------------------------------
template <std::string_view const& S0, std::string_view const& S1>
constexpr std::string_view const& concat() {
    return concat<S0, S1>(std::make_index_sequence<size(S0)>{},
                          std::make_index_sequence<size(S1)>{});
}
//------------------------------------------------------------------------------
template <std::string_view const& S0, std::string_view const& S1,
          std::string_view const& S2, std::string_view const&... Ss>
constexpr std::string_view const& concat() {
  return concat<S0, concat<S1, S2, Ss...>()>();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
