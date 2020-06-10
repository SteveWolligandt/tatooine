#ifndef TATOOINE_CONCPETS_H
#define TATOOINE_CONCPETS_H
//==============================================================================
#include <concepts>
#include "invocable_with_n_types.h"
//==============================================================================
namespace tatooine {
//==============================================================================
// typedefs
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
concept has_defined_this_t = requires {
  typename T::this_t;
};
//-----------------------------------------------------------------------------
template <typename T>
concept has_defined_parent_t = requires {
  typename T::parent_t;
};
//-----------------------------------------------------------------------------
template <typename T>
concept has_defined_tensor_t = requires {
  typename T::tensor_t;
};
//-----------------------------------------------------------------------------
template <typename T>
concept has_defined_pos_t = requires {
  typename T::pos_t;
};
//==============================================================================
// methods
//==============================================================================
template <typename T>
concept has_static_num_dimensions_method = requires {
  { T::num_dimensions() } -> std::convertible_to<std::size_t>;
};
//-----------------------------------------------------------------------------
template <typename T>
concept has_static_rank_method = requires {
  { T::rank() } -> std::convertible_to<std::size_t>;
};
//-----------------------------------------------------------------------------
template <typename F, typename... Is>
concept invocable_with_integrals = std::regular_invocable<F, Is...> &&
                                   (std::is_integral_v<Is> && ...);
//==============================================================================
// types
//==============================================================================
template <typename Tensor, size_t... Dims>
concept tensor_c =
  has_static_rank_method<Tensor> &&
  invocable_with_n_integrals_v<Tensor, Tensor::rank()> &&
  has_defined_real_t<Tensor>;
//-----------------------------------------------------------------------------
template <typename Tensor, size_t N>
concept vec_c = tensor_c<Tensor, N>;
//-----------------------------------------------------------------------------
template <typename Tensor, size_t M, size_t N>
concept mat_c = tensor_c<Tensor, M, N>;
//-----------------------------------------------------------------------------
template <typename Tensor, typename... Is>
concept field_c =
    invocable_with_integrals<Tensor, Is...> &&
    has_static_rank_method<Tensor>                 &&
    has_defined_real_t<Tensor>&& requires(Tensor const t, Is const... is) {
      { t(is...) } -> std::convertible_to<typename Tensor::real_t>;
    }                                       &&
    sizeof...(Is) == Tensor::rank();
//-----------------------------------------------------------------------------
template <typename Flowmap, size_t i>
concept flowmap_c =
  has_defined_real_t<Flowmap> &&
  has_defined_pos_t<Flowmap>&&
  has_static_num_dimensions_method<Flowmap> &&
  Flowmap::num_dimensions() == i &&
  requires(Flowmap const flowmap,
           typename Flowmap::pos_t const& x,
           typename Flowmap::real_t const t,
           typename Flowmap::real_t const tau) {
    { flowmap(x, t, tau) }
      -> std::convertible_to<typename Flowmap::pos_t>;
    { flowmap.evaluate(x, t, tau) }
      -> std::convertible_to<typename Flowmap::pos_t>;
  };
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
