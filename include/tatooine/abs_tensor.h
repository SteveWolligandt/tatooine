#ifndef TATOOINE_ABS_TENSOR_H
#define TATOOINE_ABS_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, size_t... Dims>
struct abs_tensor : base_tensor<abs_tensor<Tensor, Dims...>,
                                typename Tensor::real_t, Dims...> {
  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  constexpr explicit abs_tensor(
      const base_tensor<Tensor, typename Tensor::real_t, Dims...>&
          internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto operator()(Is... is) const {
    static_assert(sizeof...(Is) == sizeof...(Dims));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto at(Is... is) const {
    static_assert(sizeof...(Is) == sizeof...(Dims));
    return std::abs(m_internal_tensor(is...));
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
constexpr auto abs(const base_tensor<Tensor, Real, Dims...>& t) {
  return abs_tensor<Tensor, Dims...>{t.as_derived()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
