#ifndef TATOOINE_SPACETIME_H
#define TATOOINE_SPACETIME_H

#include "field.h"
#include "symbolic_field.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Field, typename real_t, size_t N, size_t VecDim>
struct spacetime
    : field<spacetime<Field, real_t, N, VecDim>, real_t, N, VecDim> {
  using field_t = Field;
  using this_t = spacetime<field_t, real_t, N, VecDim>;
  using parent_t =
      field<this_t, real_t, N, VecDim>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  static_assert(field_t::tensor_t::num_dimensions() == 1);
  static_assert(field_t::num_dimensions() == N - 1);
  static_assert(field_t::tensor_t::dimension(0) == VecDim - 1);

  //============================================================================
  private:
  field_t m_field;

  //============================================================================
  public:
   spacetime(const field<field_t, real_t, N - 1, VecDim - 1>& f) : m_field{f.as_derived()} {}

  //----------------------------------------------------------------------------
   constexpr tensor_t evaluate(const pos_t& x, real_t /*t*/) const {
     tensor<real_t, N - 1> spatial_position;
     real_t                temporal_position = x(N - 1);
     for (size_t i = 0; i < N - 1; ++i) { spatial_position(i) = x(i); }

     auto                  v = m_field(spatial_position, temporal_position);
     tensor_t t_out;
     for (size_t i = 0; i < N - 1; ++i) { t_out(i) = v(i); }
     t_out(N - 1) = 1;
     return t_out;
   }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename field_t, typename real_t, size_t N, size_t VecDim>
spacetime(const field<field_t, real_t, N, VecDim>&)
    ->spacetime<field_t, real_t, N + 1, VecDim + 1>;

//==============================================================================
template <typename real_t, size_t N, size_t VecDim>
struct spacetime<symbolic::field<real_t, N - 1, VecDim - 1>, real_t, N, VecDim>
    : symbolic::field<real_t, N, VecDim> {
  //============================================================================
  using field_t  = symbolic::field<real_t, N - 1, VecDim - 1>;
  using this_t   = spacetime<field_t, real_t, N, VecDim>;
  using parent_t = symbolic::field<real_t, N, VecDim>;
  using typename parent_t::pos_t;
  using typename parent_t::symtensor_t;
  using typename parent_t::tensor_t;

  //============================================================================
  spacetime(const symbolic::field<real_t, N - 1, VecDim - 1>& f) {
    symtensor_t ex;
    for (size_t i = 0; i < N - 1; ++i) {
      ex(i) =
          symbolic::ev(f.expr()(i), symbolic::symbol::t() == symbolic::symbol::x(N - 1));
    }
    ex(N - 1) = 1;
    this->set_expr(ex);
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename real_t, size_t N, size_t VecDim>
spacetime(const symbolic::field<real_t, N, VecDim>&)
    ->spacetime<symbolic::field<real_t, N, VecDim>, real_t, N + 1, VecDim + 1>;

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
