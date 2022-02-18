#ifndef TATOOINE_EINSTEIN_NOTATION_INDEX_H
#define TATOOINE_EINSTEIN_NOTATION_INDEX_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine::einstein_notation {
//==============================================================================
template <std::size_t I>
struct index_t {
  static auto constexpr get() { return I; }
};
//==============================================================================
[[maybe_unused]] static auto constexpr inline i = index_t<0>{};
[[maybe_unused]] static auto constexpr inline j = index_t<1>{};
[[maybe_unused]] static auto constexpr inline k = index_t<2>{};
[[maybe_unused]] static auto constexpr inline l = index_t<3>{};
[[maybe_unused]] static auto constexpr inline m = index_t<4>{};
[[maybe_unused]] static auto constexpr inline n = index_t<5>{};
[[maybe_unused]] static auto constexpr inline o = index_t<6>{};
[[maybe_unused]] static auto constexpr inline p = index_t<7>{};
[[maybe_unused]] static auto constexpr inline q = index_t<8>{};
[[maybe_unused]] static auto constexpr inline r = index_t<9>{};
[[maybe_unused]] static auto constexpr inline s = index_t<10>{};
[[maybe_unused]] static auto constexpr inline t = index_t<11>{};
[[maybe_unused]] static auto constexpr inline u = index_t<12>{};
[[maybe_unused]] static auto constexpr inline v = index_t<13>{};
[[maybe_unused]] static auto constexpr inline w = index_t<14>{};
//==============================================================================
template <typename T>
struct is_index_impl : std::false_type {};
//------------------------------------------------------------------------------
template <std::size_t N>
struct is_index_impl<index_t<N>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename... Ts>
static auto constexpr is_index = (is_index_impl<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename T>
concept index = is_index<T>;
//==============================================================================
}  // namespace tatooine::einstein_notation
//==============================================================================
#endif
