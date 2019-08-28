#ifndef TATOOINE_FIELD_H
#define TATOOINE_FIELD_H

#include "crtp.h"
#include "tensor.h"
#include "type_traits.h"

//==============================================================================
namespace tatooine {
//==============================================================================

struct out_of_domain : std::runtime_error {
  out_of_domain() : std::runtime_error{""} {}
};

template <typename derived_t, typename Real, size_t N, size_t... TensorDims>
struct field : crtp<derived_t> {
  using real_t   = Real;
  using this_t   = field<derived_t, real_t, N, TensorDims...>;
  using parent_t = crtp<derived_t>;
  using pos_t    = tensor<real_t, N>;
  using tensor_t = std::conditional_t<sizeof...(TensorDims) == 0, real_t,
                                      tensor<real_t, TensorDims...>>;
  static constexpr auto num_dimensions() { return N; }
  static constexpr auto has_in_domain() { return has_in_domain_v<derived_t>; }
  using parent_t::as_derived;

  //============================================================================
  constexpr decltype(auto) operator()(const pos_t &x, real_t t = 0) const {
    return evaluate(x, t);
  }

  //----------------------------------------------------------------------------
  constexpr decltype(auto) evaluate(const pos_t &x, real_t t = 0) const {
    if (!in_domain(x, t)) { throw out_of_domain{}; }
    return as_derived().evaluate(x, t);
  }

  //----------------------------------------------------------------------------
  constexpr decltype(auto) in_domain([[maybe_unused]] const pos_t &x,
                                     [[maybe_unused]] real_t t = 0) const {
    if constexpr (has_in_domain()) {
      return as_derived().in_domain(x, t);
    } else { 
      return true;
    }
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
