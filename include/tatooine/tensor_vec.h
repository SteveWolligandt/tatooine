#ifndef TATOOINE_TENSOR_VEC_H
#define TATOOINE_TENSOR_VEC_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct vec : tensor<Real, N> {  // NOLINT
  using parent_t = tensor<Real, N>;
  using parent_t::at;
  using parent_t::dimension;
  using parent_t::rank;
  using parent_t::parent_t;
  using parent_t::operator();

  template <typename... Ts, size_t _Dim0 = parent_t::dimension(0),
            std::enable_if_t<_Dim0 == sizeof...(Ts), bool> = true>
  constexpr vec(const Ts&... ts) : parent_t{ts...} {}

  using iterator = typename parent_t::array_parent_t::container_t::iterator;
  using const_iterator =
      typename parent_t::array_parent_t::container_t::const_iterator;

  //----------------------------------------------------------------------------
  constexpr vec(const vec&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  explicit constexpr vec(vec&& other) noexcept : parent_t{std::move(other)} {}
  //----------------------------------------------------------------------------
  constexpr auto operator=(const vec&) -> vec& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  constexpr auto operator=(vec&& other) noexcept -> vec& {
    parent_t::operator=(std::move(other));
    return *this;
  }
  template <typename OtherTensor, typename OtherReal>
  constexpr vec(const base_tensor<OtherTensor, OtherReal, N>& other) {
    for (size_t i = 0; i < N; ++i) { at(i) = other(i); }
  }
  //----------------------------------------------------------------------------
  ~vec() = default;
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
vec(const Ts&...) -> vec<promote_t<Ts...>, sizeof...(Ts)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
using vec2 = vec<double, 2>;
using vec3 = vec<double, 3>;
using vec4 = vec<double, 4>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
