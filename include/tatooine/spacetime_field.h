#ifndef TATOOINE_SPACETIME_H
#define TATOOINE_SPACETIME_H

#include "field.h"
#include "symbolic_field.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Field, typename Real, size_t N, size_t VecDim>
struct spacetime_field
    : field<spacetime_field<Field, Real, N, VecDim>, Real, N, VecDim> {
  using field_t  = Field;
  using this_t   = spacetime_field<field_t, Real, N, VecDim>;
  using parent_t = field<this_t, Real, N, VecDim>;
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
  spacetime_field(const spacetime_field& other) = default;
  spacetime_field(spacetime_field&& other)      = default;
  spacetime_field& operator=(const spacetime_field& other) = default;
  spacetime_field& operator=(spacetime_field&& other) = default;
  spacetime_field(const field<field_t, Real, N - 1, VecDim - 1>& f)
      : m_field{f.as_derived()} {}

  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, Real /*t*/) const {
    tensor<Real, N - 1> spatial_position;
    Real                temporal_position = x(N - 1);
    for (size_t i = 0; i < N - 1; ++i) { spatial_position(i) = x(i); }

    auto     v = m_field(spatial_position, temporal_position);
    tensor_t t_out;
    for (size_t i = 0; i < N - 1; ++i) { t_out(i) = v(i); }
    t_out(N - 1) = 1;
    return t_out;
  }
  //----------------------------------------------------------------------------
  constexpr decltype(auto) in_domain(const pos_t& x, Real /*t*/) const {
    tensor<Real, N - 1> spatial_position;
    Real                temporal_position = x(N - 1);
    for (size_t i = 0; i < N - 1; ++i) { spatial_position(i) = x(i); }
    return m_field.in_domain(spatial_position, temporal_position);
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename field_t, typename Real, size_t N, size_t VecDim>
spacetime_field(const field<field_t, Real, N, VecDim>&)
    ->spacetime_field<field_t, Real, N + 1, VecDim + 1>;

//==============================================================================
template <typename Real, size_t N, size_t VecDim>
struct spacetime_field<symbolic::field<Real, N - 1, VecDim - 1>, Real, N,
                       VecDim> : symbolic::field<Real, N, VecDim> {
  //============================================================================
  using field_t  = symbolic::field<Real, N - 1, VecDim - 1>;
  using this_t   = spacetime_field<field_t, Real, N, VecDim>;
  using parent_t = symbolic::field<Real, N, VecDim>;
  using typename parent_t::pos_t;
  using typename parent_t::symtensor_t;
  using typename parent_t::tensor_t;

  //============================================================================
  spacetime_field(const field<symbolic::field<Real, N - 1, VecDim - 1>, Real,
                              N - 1, VecDim - 1>& f) {
    symtensor_t ex;
    for (size_t i = 0; i < N - 1; ++i) {
      ex(i) = symbolic::ev(f.as_derived().expr()(i),
                           symbolic::symbol::t() == symbolic::symbol::x(N - 1));
    }
    ex(N - 1) = 1;
    this->set_expr(ex);
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Real, size_t N, size_t VecDim>
spacetime_field(const symbolic::field<Real, N, VecDim>&)
    ->spacetime_field<symbolic::field<Real, N, VecDim>, Real, N + 1,
                      VecDim + 1>;

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
