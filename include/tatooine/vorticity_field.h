#ifndef TATOOINE_VORTICITY_FIELD_H
#define TATOOINE_VORTICITY_FIELD_H

#include "field.h"
#include "diff.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V>
class vorticity_field
    : public field<vorticity_field<V>, typename V::real_t, 3, 3> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using real_t = typename V::real_t;
  using this_t = vorticity_field<V>;
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
  vorticity_field(const field<V, Real, 3, 3>& v) : m_vf{v.as_derived()} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    const auto Jf = diff(m_vf);
    const auto J  = Jf(x, t);
    return {J(2, 1) - J(1, 2), J(0, 2) - J(2, 0), J(1, 0) - J(0, 1)};
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t t) const {
    return m_vf.in_domain(x, t);
  }
};
//==============================================================================
template <typename V, typename Real>
auto vorticity(const field<V, Real, 3, 3>& vf) {
  return vorticity_field<V>{vf.as_derived()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
