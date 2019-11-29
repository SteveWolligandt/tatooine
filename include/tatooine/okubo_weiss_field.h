#ifndef TATOOINE_LAMBDA2_FIELD_H
#define TATOOINE_LAMBDA2_FIELD_H

#include "field.h"
#include "diff.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V, size_t N>
class okubo_weiss_field
    : public field<okubo_weiss_field<V, N>, typename V::real_t, N> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using real_t = typename V::real_t;
  using this_t = okubo_weiss_field<V, N>;
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
  template <typename Real>
  okubo_weiss_field(const field<V, Real, N, N>& v) : m_vf{v.as_derived()} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    auto J      = diff(m_vf)(x, t);
    auto S      = (J + transpose(J)) / 2;
    auto Omega  = (J - transpose(J)) / 2;
    return (sqr_norm(Omega) - sqr_norm(S)) / 2;
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t t) const {
    return m_vf.in_domain(x, t);
  }
};
//==============================================================================
template <typename V>
class okubo_weiss_field<V, 3>
    : public field<okubo_weiss_field<V, 3>, typename V::real_t, 3> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using real_t = typename V::real_t;
  using this_t = okubo_weiss_field<V, 3>;
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
  template <typename Real>
  okubo_weiss_field(const field<V, Real, 3, 3>& v) : m_vf{v.as_derived()} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    auto J = diff(m_vf)(x, t);
    return -(J(0, 0) * J(0, 0) +
             J(1, 1) * J(1, 1) +
             J(2, 2) * J(2, 2) +
             2 * J(0, 1) * J(1, 0) +
             2 * J(0, 2) * J(2, 0) +
             2 * J(1, 2) * J(2, 1)) /
           2;
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t t) const {
    return m_vf.in_domain(x, t);
  }
};
//==============================================================================
template <typename V, typename Real, size_t N>
auto okubo_weiss(const field<V, Real, N, N>& vf) {
  return okubo_weiss_field<V, N>{vf.as_derived()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
