#ifndef TATOOINE_CONCPETS_H
#define TATOOINE_CONCPETS_H
#include <concepts>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;
//-----------------------------------------------------------------------------
template <typename T>
concept has_defined_real_t = requires {
  typename T::real_t;
};
//-----------------------------------------------------------------------------
template <typename T>
concept has_defined_tensor_t = requires {
  typename T::tensor_t;
};
//-----------------------------------------------------------------------------
template <typename T>
concept has_static_rank = requires {
  { T::rank() } -> std::convertible_to<std::size_t>;
};
//-----------------------------------------------------------------------------
template <typename F, typename... Is>
concept invocable_with_integrals = std::regular_invocable<F, Is...> &&
                                   (std::is_integral_v<Is> && ...);
//-----------------------------------------------------------------------------
template <typename Tensor, typename... Is>
concept tensor_c =
    invocable_with_integrals<Tensor, Is...> &&
    has_static_rank<Tensor>                 &&
    has_defined_real_t<Tensor>&& requires(Tensor const t, Is const... is) {
      { t(is...) } -> std::convertible_to<typename Tensor::real_t>;
    }                                       &&
    sizeof...(Is) == Tensor::rank();
//-----------------------------------------------------------------------------
template <typename Tensor>
concept vec_c = tensor_c<Tensor, std::size_t>;
//-----------------------------------------------------------------------------
template <typename Tensor>
concept mat_c = tensor_c<Tensor, std::size_t, std::size_t>;
//-----------------------------------------------------------------------------
template <typename Tensor, typename... Is>
concept field_c =
    invocable_with_integrals<Tensor, Is...> &&
    has_static_rank<Tensor>                 &&
    has_defined_real_t<Tensor>&& requires(Tensor const t, Is const... is) {
      { t(is...) } -> std::convertible_to<typename Tensor::real_t>;
    }                                       &&
    sizeof...(Is) == Tensor::rank();
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
