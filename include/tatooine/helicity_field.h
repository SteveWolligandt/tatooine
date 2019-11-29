#ifndef TATOOINE_HELICITY_FIELD_H
#define TATOOINE_HELICITY_FIELD_H

#include "field.h"
#include "vorticity_field.h"
#include "diff.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V>
class helicity_field
    : public field<helicity_field<V>, typename V::real_t, 3, 3> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using real_t = typename V::real_t;
  using this_t = helicity_field<V>;
  using parent_t =
      field<this_t, real_t, V::num_dimensions(), V::tensor_t::dimension(0)>;
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
  helicity_field(const field<V, Real, 3, 3>& v) : m_vf{v.as_derived()} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    const auto vort = vorticity(m_vf)(x, t);
    return cross(vort, m_vf(x, t));
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t t) const {
    return m_vf.in_domain(x, t);
  }
};
//==============================================================================
template <typename V, typename Real>
auto helicity(const field<V, Real, 3, 3>& vf) {
  return helicity_field<V>{vf.as_derived()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
