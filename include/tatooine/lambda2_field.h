#ifndef TATOOINE_LAMBDA2_FIELD_H
#define TATOOINE_LAMBDA2_FIELD_H

#include "field.h"
#include "diff.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V>
class lambda2_field : public field<lambda2_field<V>, typename V::real_t,
                                   V::num_dimensions()> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using real_t = typename V::real_t;
  using this_t = lambda2_field<V>;
  using parent_t =
      field<this_t, real_t, V::num_dimensions()>;
  using typename parent_t::tensor_t;
  using typename parent_t::pos_t;
  //============================================================================
  // fields
  //============================================================================
 private:
  V m_vf;

  //============================================================================
  // ctor
  //============================================================================
 public:
  template <typename Real, size_t N>
  lambda2_field(const field<V, Real, N, N>& v) : m_vf{v.as_derived()} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    auto J     = diff(m_vf)(x, t);
    auto S     = (J + transpose(J)) / 2;
    auto Omega = (J - transpose(J)) / 2;
    auto A     = S * S + Omega * Omega;
    return eigenvalues_sym(A)(1);
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t t) const {
    return m_vf.in_domain(x, t);
  }
};
//==============================================================================
template <typename V, typename Real, size_t N>
auto lambda2(const field<V, Real, N, N>& vf) {
  return lambda2_field<V>{vf.as_derived()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
