#ifndef TATOOINE_DIFF_H
#define TATOOINE_DIFF_H

#include "field.h"
#include "utility.h"
#include "symbolic_field.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Field, size_t... TensorDims>
struct derived_field : field<derived_field<Field, TensorDims...>, typename Field::real_t,
                             Field::num_dimensions(), TensorDims...> {
  using this_t   = derived_field<Field, TensorDims...>;
  using parent_t = field<this_t, typename Field::real_t,
                         Field::num_dimensions(), TensorDims...>;
  using parent_t::num_dimensions;
  using typename parent_t::real_t;
  using typename parent_t::pos_t;
  using vec_t = vec<real_t, num_dimensions()>;
  using typename parent_t::tensor_t;

  static_assert(Field::tensor_t::num_dimensions() + 1 ==
                tensor_t::num_dimensions());

  //============================================================================
 private:
  Field m_internal_field;
  vec_t m_eps;

  //============================================================================
 public:
  template <typename Real, size_t N, size_t... FieldTensorDims>
  derived_field(const field<Field, Real, N, FieldTensorDims...>& f,
                Real                                             eps = 1e-6)
      : m_internal_field{f.as_derived()}, m_eps{fill{eps}} {}
  template <typename Real, size_t N, size_t... FieldTensorDims>
  derived_field(const field<Field, Real, N, FieldTensorDims...>& f,
                const vec<Real, num_dimensions()>&                    eps)
      : m_internal_field{f.as_derived()}, m_eps{eps} {}

  //----------------------------------------------------------------------------
  auto evaluate(const pos_t& x, const real_t t) const {
    tensor_t derivative;

    pos_t offset;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      offset(i) = m_eps(i);
      auto x0   = x - offset;
      auto x1   = x + offset;
      auto dx   = 2 * m_eps;
      if (!m_internal_field.in_domain(x0, t)) {
        x0 = x;
        dx = m_eps;
      }
      if (!m_internal_field.in_domain(x1, t)) {
        x1 = x;
        dx = m_eps;
      }
      derivative.template slice<sizeof...(TensorDims) - 1>(i) =
          (m_internal_field(x1, t) - m_internal_field(x0, t)) / (2 * m_eps);
      offset(i) = 0;
    }

    return derivative;
  }

  //----------------------------------------------------------------------------
  constexpr decltype(auto) in_domain(const pos_t& x, real_t t) const {
    return m_internal_field.in_domain(x, t);
  }

  //----------------------------------------------------------------------------
  auto&       internal_field() { return m_internal_field; }
  const auto& internal_field() const { return m_internal_field; }

  //----------------------------------------------------------------------------
  void        set_eps(const vec_t& eps) { m_eps = eps; }
  void        set_eps(vec_t&& eps) { m_eps = std::move(eps); }
  void        set_eps(real_t eps) { m_eps = vec_t{fill{eps}}; }
  auto&       eps() { return m_eps; }
  const auto& eps() const { return m_eps; }
  auto&       eps(size_t i) { return m_eps(i); }
  auto        eps(size_t i) const { return m_eps(i); }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff(const field<Field, Real, N, TensorDims...>& f) {
  return derived_field<Field, TensorDims..., N>{f};
}

//------------------------------------------------------------------------------
template <typename Real, size_t N, size_t... TensorDims>
auto diff(const symbolic::field<Real, N, TensorDims...>& f) {
  tensor<GiNaC::ex, TensorDims..., N> ex;
  for (size_t i = 0; i < N; ++i) {
    ex.template slice<sizeof...(TensorDims)>(i) =
        diff(f.expr(), symbolic::symbol::x(i));
  }
  return symbolic::field<Real, N, TensorDims..., N>{std::move(ex)};
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
