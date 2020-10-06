#ifndef TATOOINE_SPACETIME_VECTORFIELD_H
#define TATOOINE_SPACETIME_VECTORFIELD_H
//==============================================================================
#include <tatooine/packages.h>

#include "field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Field, typename Real, size_t N>
struct spacetime_vectorfield
    : vectorfield<spacetime_vectorfield<Field, Real, N>, Real, N> {
  using field_t  = Field;
  using this_t   = spacetime_vectorfield<field_t, Real, N>;
  using parent_t = vectorfield<this_t, Real, N>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  static constexpr auto holds_field_pointer = std::is_pointer_v<field_t>;

  static_assert(std::remove_pointer_t<
                    std::decay_t<field_t>>::tensor_t::num_dimensions() == 1);
  static_assert(
      std::remove_pointer_t<std::decay_t<field_t>>::num_dimensions() == N - 1);

  //============================================================================
 private:
  field_t m_field;

  //============================================================================
 public:
  spacetime_vectorfield(const spacetime_vectorfield& other) = default;
  spacetime_vectorfield(spacetime_vectorfield&& other)      = default;
  spacetime_vectorfield& operator=(const spacetime_vectorfield& other) =
      default;
  spacetime_vectorfield& operator=(spacetime_vectorfield&& other) = default;
  template <typename _F = field_t, std::enable_if_t<!std::is_pointer_v<_F>, bool> = true>
  spacetime_vectorfield(const vectorfield<field_t, Real, N - 1>& f)
      : m_field{f.as_derived()} {}
  template <typename _F = field_t, std::enable_if_t<std::is_pointer_v<_F>, bool> = true>
  spacetime_vectorfield(parent::vectorfield<Real, N - 1> const* f)
      : m_field{f} {}
  template <typename _F                                   = field_t,
            std::enable_if_t<std::is_pointer_v<_F>, bool> = true>
  spacetime_vectorfield() : m_field{nullptr} {}

  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, Real /*t*/) const {
    tensor<Real, N - 1> spatial_position;
    Real                temporal_position = x(N - 1);
    for (size_t i = 0; i < N - 1; ++i) {
      spatial_position(i) = x(i);
    }

    auto v = [&] {
      if constexpr (holds_field_pointer) {
        return m_field->evaluate(spatial_position, temporal_position);
      } else {
        return m_field(spatial_position, temporal_position);
      }
    }();
    tensor_t t_out;
    for (size_t i = 0; i < N - 1; ++i) {
      t_out(i) = v(i);
    }
    t_out(N - 1) = 1;
    return t_out;
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real /*t*/) const {
    tensor<Real, N - 1> spatial_position;
    Real                temporal_position = x(N - 1);
    for (size_t i = 0; i < N - 1; ++i) {
      spatial_position(i) = x(i);
    }
    if constexpr (holds_field_pointer) {
      return m_field->in_domain(spatial_position, temporal_position);
    } else {
      return m_field.in_domain(spatial_position, temporal_position);
    }
  }
  //----------------------------------------------------------------------------
  template <typename V,typename _F = field_t, 
            std::enable_if_t<std::is_pointer_v<_F>, bool> = true>
  void set_field(vectorfield<V, Real, N - 1> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F = field_t, std::enable_if_t<std::is_pointer_v<_F>, bool> = true>
  void set_field(parent::vectorfield<Real, N - 1> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F = field_t, std::enable_if_t<std::is_pointer_v<_F>, bool> = true>
  void set_field(parent::vectorfield<Real, N - 1> const* v) {
    m_field = v;
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Field, typename Real, size_t N>
spacetime_vectorfield(const vectorfield<Field, Real, N>&)
    -> spacetime_vectorfield<Field, Real, N + 1>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
spacetime_vectorfield(parent::vectorfield<Real, N> const*)
    -> spacetime_vectorfield<parent::vectorfield<Real, N> const*, Real, N + 1>;

#if TATOOINE_GINAC_AVAILABLE
#include "symbolic_field.h"
//==============================================================================
template <typename Real, size_t N>
struct spacetime_vectorfield<symbolic::field<Real, N - 1>, Real, N>
    : symbolic::field<Real, N> {
  //============================================================================
  using field_t  = symbolic::field<Real, N - 1>;
  using this_t   = spacetime_vectorfield<field_t, Real, N>;
  using parent_t = symbolic::field<Real, N>;
  using typename parent_t::pos_t;
  using typename parent_t::symtensor_t;
  using typename parent_t::tensor_t;

  //============================================================================
  spacetime_vectorfield(
      const field<symbolic::field<Real, N - 1>, Real, N - 1>& f) {
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
template <typename Real, size_t N>
spacetime_vectorfield(const symbolic::field<Real, N>&)
    -> spacetime_vectorfield<symbolic::field<Real, N>, Real, N + 1>;
#endif
//==============================================================================
template <typename V, typename Real, size_t N>
auto spacetime(const vectorfield<V, Real, N>& vf) {
  return spacetime_vectorfield<V, Real, N + 1>{vf.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename Real, size_t N>
auto spacetime(parent::vectorfield<Real, N> const* vf) {
  return spacetime_vectorfield<parent::vectorfield<Real, N> const*, Real,
                               N + 1>{vf};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename Real, size_t N>
auto spacetime(parent::vectorfield<Real, N> const& vf) {
  return spacetime_vectorfield<parent::vectorfield<Real, N> const*, Real,
                               N + 1>{&vf};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
