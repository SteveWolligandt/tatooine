#ifndef TATOOINE_SPACETIME_SPLITTED_VECTORFIELD_H
#define TATOOINE_SPACETIME_SPLITTED_VECTORFIELD_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V>
struct spacetime_splitted_vectorfield
    : vectorfield<spacetime_splitted_vectorfield<V>, typename V::real_t,
                  V::num_dimensions() - 1> {
  using this_t = spacetime_splitted_vectorfield<V>;
  using parent_t =
      vectorfield<this_t, typename V::real_t, V::num_dimensions() - 1>;
  using parent_t::num_dimensions;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  //============================================================================
  V const& m_v;
  //============================================================================
  template <typename VReal, size_t N>
  explicit spacetime_splitted_vectorfield(vectorfield<V, VReal, N> const& v)
      : m_v{v.as_derived()} {}
  //----------------------------------------------------------------------------
  spacetime_splitted_vectorfield(spacetime_splitted_vectorfield const&) =
      default;
  spacetime_splitted_vectorfield(spacetime_splitted_vectorfield&&) noexcept =
      default;
  //----------------------------------------------------------------------------
  ~spacetime_splitted_vectorfield() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    vec<real_t, num_dimensions() + 1> pt;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      pt(i) = x(i);
    }
    pt(num_dimensions()) = t;
    auto const vt        = m_v(pt, t);
    tensor_t   v;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      v(i) = vt(i);
    }
    return v;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto in_domain(pos_t const& x, real_t t) const -> bool final {
    vec<real_t, num_dimensions() + 1> pt;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      pt(i) = x(i);
    }
    pt(num_dimensions()) = t;
    return m_v.in_domain(pt, t);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
