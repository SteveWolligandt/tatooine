#ifndef TATOOINE_CURVATURE_FIELD_H
#define TATOOINE_CURVATURE_FIELD_H

#include "field.h"
#include "diff.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V, size_t N>
class curvature_field;
//==============================================================================
template <typename V>
class curvature_field<V, 2>
    : public field<curvature_field<V>, typename V::real_t, 2> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using this_t   = curvature_field<V>;
  using real_t   = typename V::real_t;
  using parent_type = field<this_t, real_t, 2>;
  using typename parent_type::tensor_t;
  using typename parent_type::pos_t;

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
  curvature_field(const field<V, Real, 2, 2>& v) : m_vf{v.as_derived()} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    const auto Jf = diff(m_vf);
    const auto J  = Jf(x, t);
    const auto v  = m_vf(x, t);
    const auto a  = J * v;
    const auto lv = length(v);
    return (v(0) * a(1) - v(1) * a(0)) / (lv * lv * lv);
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t t) const {
    return m_vf.in_domain(x, t);
  }
};
//==============================================================================
template <typename V>
class curvature_field<V, 3>
    : public field<curvature_field<V>, typename V::real_t, 3> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using this_t   = curvature_field<V>;
  using real_t   = typename V::real_t;
  using parent_type = field<this_t, real_t, 3>;
  using typename parent_type::tensor_t;
  using typename parent_type::pos_t;

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
  curvature_field(const field<V, Real, 3, 3>& v) : m_vf{v.as_derived()} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    const auto Jf = diff(m_vf);
    const auto J  = Jf(x, t);
    const auto v  = m_vf(x, t);
    const auto a  = J * v;
    const auto lv = length(v);
    return cross(v,a) / (lv * lv * lv);
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t t) const {
    return m_vf.in_domain(x, t);
  }
};
//==============================================================================
template <typename V, typename Real>
auto curvature(const field<V, Real, 2, 2>& vf) {
  return curvature_field<V, 2>{vf.as_derived()};
}
//==============================================================================
template <typename V, typename Real>
auto curvature(const field<V, Real, 3, 3>& vf) {
  return curvature_field<V, 3>{vf.as_derived()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
