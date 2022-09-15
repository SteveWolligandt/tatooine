#ifndef TATOOINE_INVOCABLE_WITH_N_TYPES_H
#define TATOOINE_INVOCABLE_WITH_N_TYPES_H
//==============================================================================
#include <utility>
//==============================================================================
namespace tatooine {
//==============================================================================
/// For each type T in Ts it will be checked if F is invocable with n-times T.
/// E.g. is_invocable_with_n_types_impl<F, 2, int, unsigned int> checks if F is
/// invocable with (int, int) and (unsigned int, unsigned int)
template <typename F, std::size_t N, typename... Ts>
struct is_invocable_with_n_types_impl {
  //----------------------------------------------------------------------------
 private:
  template <typename T, std::size_t... NTimes>
  static constexpr bool check(std::index_sequence<NTimes...>) {
    return std::is_invocable_v<F, decltype(((void)NTimes, T{}))...>;
  }
  //----------------------------------------------------------------------------
 public:
  static constexpr bool value =
      (check<Ts>(std::make_index_sequence<N>{}) && ...);
};
//==============================================================================
template <typename F, std::size_t N>
struct is_invocable_with_n_integrals_impl
    : is_invocable_with_n_types_impl<F, N, bool, char, unsigned char, char16_t,
                             char32_t, wchar_t, short, unsigned short, int,
                             unsigned int, long, unsigned long, long long,
                             unsigned long long> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, std::size_t N>
static constexpr auto is_invocable_with_n_integrals =
    is_invocable_with_n_integrals_impl<F, N>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, std::size_t N>
concept invocable_with_n_integrals =
    is_invocable_with_n_integrals<F, N>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, std::size_t N>
struct is_invocable_with_n_floating_points_impl
    : is_invocable_with_n_types_impl<F, N, float, double, long double> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, std::size_t N>
static constexpr bool is_invocable_with_n_floating_points =
    is_invocable_with_n_floating_points_impl<F, N>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, std::size_t N>
concept invocable_with_n_floating_points =
    is_invocable_with_n_floating_points<F, N>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
