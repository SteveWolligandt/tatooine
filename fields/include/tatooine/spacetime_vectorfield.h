#ifndef TATOOINE_SPACETIME_VECTORFIELD_H
#define TATOOINE_SPACETIME_VECTORFIELD_H
//==============================================================================
#include <tatooine/available_libraries.h>

#include "field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, typename Real, size_t N>
struct spacetime_vectorfield
    : vectorfield<spacetime_vectorfield<V, Real, N>, Real, N> {
  using this_type   = spacetime_vectorfield<V, Real, N>;
  using parent_type = vectorfield<this_type, Real, N>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  static constexpr auto holds_field_pointer = is_pointer<V>;

  static_assert(std::remove_pointer_t<std::decay_t<V>>::tensor_type::rank() == 1);
  static_assert(std::remove_pointer_t<std::decay_t<V>>::num_dimensions() ==
                N - 1);

 private:
  //============================================================================
  // members
  //============================================================================
  V m_v;

 public:
  //============================================================================
  // copy / move ctors
  //============================================================================
  spacetime_vectorfield(spacetime_vectorfield const& other)     = default;
  spacetime_vectorfield(spacetime_vectorfield&& other) noexcept = default;
  //============================================================================
  // assign ops
  //============================================================================
  auto operator                 =(spacetime_vectorfield const& other)
      -> spacetime_vectorfield& = default;
  auto operator                 =(spacetime_vectorfield&& other) noexcept
      -> spacetime_vectorfield& = default;
  //============================================================================
  // dtor
  //============================================================================
  virtual ~spacetime_vectorfield() = default;
  //============================================================================
  // ctors
  //============================================================================
  constexpr spacetime_vectorfield() requires is_pointer<V> : m_v{nullptr} {}
  //----------------------------------------------------------------------------
  constexpr spacetime_vectorfield(
      polymorphic::vectorfield<Real, N - 1> const* v) requires is_pointer<V>
      : m_v{v} {}
  //----------------------------------------------------------------------------
  template <std::convertible_to<V> W>
  constexpr spacetime_vectorfield(
      vectorfield<W, Real, N - 1> const& w) requires(!is_pointer<V>)
      : m_v{w.as_derived()} {}
  //============================================================================
  // methods
  //============================================================================
  [[nodiscard]] constexpr auto evaluate(pos_type const& x, Real /*t*/) const
      -> tensor_type final {
    tensor<Real, N - 1> spatial_position;
    Real                temporal_position = x(N - 1);
    for (size_t i = 0; i < N - 1; ++i) {
      spatial_position(i) = x(i);
    }

    auto const sample = v()(spatial_position, temporal_position);
    tensor_type   t_out;
    for (size_t i = 0; i < N - 1; ++i) {
      t_out(i) = sample(i);
    }
    t_out(N - 1) = 1;
    return t_out;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(pos_type const& x, Real /*t*/) const
      -> bool final {
    tensor<Real, N - 1> spatial_position;
    Real                temporal_position = x(N - 1);
    for (size_t i = 0; i < N - 1; ++i) {
      spatial_position(i) = x(i);
    }
    return v().in_domain(spatial_position, temporal_position);
  }
  //----------------------------------------------------------------------------
  template <typename W>
  void set_field(vectorfield<W, Real, N - 1> const& v) requires is_pointer<V> {
    m_v = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void set_field(polymorphic::vectorfield<Real, N - 1> const& v) requires
      is_pointer<V> {
    m_v = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void set_field(polymorphic::vectorfield<Real, N - 1> const* v) requires
      is_pointer<V> {
    m_v = v;
  }

 private : auto v() -> auto& {
    if constexpr (holds_field_pointer) {
      return *m_v;
    } else {
      return m_v;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto v() const -> auto const& {
    if constexpr (holds_field_pointer) {
      return *m_v;
    } else {
      return m_v;
    }
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename V, typename Real, size_t N>
spacetime_vectorfield(vectorfield<V, Real, N> const&)
    -> spacetime_vectorfield<V, Real, N + 1>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
spacetime_vectorfield(polymorphic::vectorfield<Real, N> const*)
    -> spacetime_vectorfield<polymorphic::vectorfield<Real, N> const*, Real,
                             N + 1>;

//==============================================================================
// SYMBOLIC
//==============================================================================
#if TATOOINE_GINAC_AVAILABLE
#include "symbolic_field.h"
//==============================================================================
template <typename Real, size_t N>
struct spacetime_vectorfield<symbolic::field<Real, N - 1>, Real, N>
    : symbolic::field<Real, N> {
  //============================================================================
  using V        = symbolic::field<Real, N - 1>;
  using this_type   = spacetime_vectorfield<V, Real, N>;
  using parent_type = symbolic::field<Real, N>;
  using typename parent_type::pos_type;
  using typename parent_type::symtensor_type;
  using typename parent_type::tensor_type;

  //============================================================================
  spacetime_vectorfield(
      field<symbolic::field<Real, N - 1> const, Real, N - 1>& v) {
    symtensor_type ex;
    for (size_t i = 0; i < N - 1; ++i) {
      ex(i) = symbolic::ev(v.as_derived().expr()(i),
                           symbolic::symbol::t() == symbolic::symbol::x(N - 1));
    }
    ex(N - 1) = 1;
    this->set_expr(ex);
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Real, size_t N>
spacetime_vectorfield(symbolic::field<Real, N> const&)
    -> spacetime_vectorfield<symbolic::field<Real, N>, Real, N + 1>;
#endif
//==============================================================================
template <typename V, typename Real, size_t N>
auto spacetime(vectorfield<V, Real, N> const& vf) {
  return spacetime_vectorfield<V, Real, N + 1>{vf.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename Real, size_t N>
auto spacetime(polymorphic::vectorfield<Real, N> const* vf) {
  return spacetime_vectorfield<polymorphic::vectorfield<Real, N> const*, Real,
                               N + 1>{vf};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename Real, size_t N>
auto spacetime(polymorphic::vectorfield<Real, N> const& vf) {
  return spacetime_vectorfield<polymorphic::vectorfield<Real, N> const*, Real,
                               N + 1>{&vf};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
