#ifndef TATOOINE_INVOCABLE_WITH_N_TYPES_H
#define TATOOINE_INVOCABLE_WITH_N_TYPES_H
//==============================================================================
namespace tatooine {
//==============================================================================
/// For each type T in Ts it will be checked if F is invocable with n-times T.
/// E.g. invocable_with_n_types<F, 2, int, unsigned int> checks if F is
/// invocable with (int, int) and (unsigned int, unsigned int)
template <typename F, std::size_t N, typename... Ts>
struct invocable_with_n_types {
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
struct invocable_with_n_integrals
    : invocable_with_n_types<
          F, N, bool, char, unsigned char, char8_t, unsigned char8_t, char16_t,
          unsigned char16_t, char32_t, unsigned char32_t, wchar_t,
          unsigned wchar_t, short, unsigned short, int, unsigned int, long,
          unsigned long, long long, unsigned long long> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, std::size_t N>
static constexpr bool invocable_with_n_integrals_v =
    invocable_with_n_integrals<F, N>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, std::size_t N>
struct invocable_with_n_floating_points
    : invocable_with_n_types<F, N, float, double, long double> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, std::size_t N>
static constexpr bool invocable_with_n_floating_points_v =
    invocable_with_n_floating_points<F, N>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
