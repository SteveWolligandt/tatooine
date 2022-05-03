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
    : public field<helicity_field<V>, typename V::real_type, 3, 3> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using real_type = typename V::real_type;
  using this_type = helicity_field<V>;
  using parent_type =
      field<this_type, real_type, V::num_dimensions(), V::tensor_type::dimension(0)>;
  using typename parent_type::tensor_type;
  using typename parent_type::pos_type;
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
  constexpr tensor_type evaluate(const pos_type& x, real_type t) const {
    const auto vort = vorticity(m_vf)(x, t);
    return cross(vort, m_vf(x, t));
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_type& x, real_type t) const {
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
