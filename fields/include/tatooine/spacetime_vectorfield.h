#ifndef TATOOINE_SPACETIME_VECTORFIELD_H
#define TATOOINE_SPACETIME_VECTORFIELD_H
//==============================================================================
#include <tatooine/available_libraries.h>

#include "field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V>
struct spacetime_vectorfield
    : vectorfield<spacetime_vectorfield<V>, field_real_type<V>,
                  field_num_dimensions<V> + 1> {
  using this_type = spacetime_vectorfield<V>;
  using parent_type =
      vectorfield<this_type, field_real_type<V>, field_num_dimensions<V> + 1>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  static constexpr auto holds_field_pointer = is_pointer<V>;
  static constexpr auto num_dimensions() -> std::size_t { return field_num_dimensions<V> + 1; }

  static_assert(field_tensor_type<V>::rank() == 1);

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
  auto operator=(spacetime_vectorfield const& other)
      -> spacetime_vectorfield& = default;
  auto operator=(spacetime_vectorfield&& other) noexcept
      -> spacetime_vectorfield& = default;
  //============================================================================
  // dtor
  //============================================================================
  virtual ~spacetime_vectorfield() = default;
  //============================================================================
  // ctors
  //============================================================================
  constexpr spacetime_vectorfield()
  requires is_pointer<V>
      : m_v{nullptr} {}
  //----------------------------------------------------------------------------
  explicit constexpr spacetime_vectorfield(
      polymorphic::vectorfield<real_type, num_dimensions() - 1> const* v)
  requires is_pointer<V>
      : m_v{v} {}
  //----------------------------------------------------------------------------
  template <std::convertible_to<V> W>
  constexpr spacetime_vectorfield(vectorfield<W, real_type, num_dimensions() - 1> const& w)
  requires(!is_pointer<V>)
      : m_v{w.as_derived()} {}
  //============================================================================
  // methods
  //============================================================================
  [[nodiscard]] constexpr auto evaluate(pos_type const& x, real_type const /*t*/) const
      -> tensor_type {
    auto spatial_position = vec<real_type, num_dimensions() - 1>{};
    for (std::size_t i = 0; i < num_dimensions() - 1; ++i) {
      spatial_position(i) = x(i);
    }
    auto temporal_position = x(num_dimensions() - 1);

    auto const  sample = v()(spatial_position, temporal_position);
    auto t_out = tensor_type{};
    for (std::size_t i = 0; i < num_dimensions() - 1; ++i) {
      t_out(i) = sample(i);
    }
    t_out(num_dimensions() - 1) = 1;
    return t_out;
  }
  //----------------------------------------------------------------------------
  template <typename W>
  auto set_field(vectorfield<W, real_type, num_dimensions() - 1> const& v)
  requires is_pointer<V>
  {
    m_v = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_field(polymorphic::vectorfield<real_type, num_dimensions() - 1> const& v)
  requires is_pointer<V>
  {
    m_v = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_field(polymorphic::vectorfield<real_type, num_dimensions() - 1> const* v)
  requires is_pointer<V>
  {
    m_v = v;
  }

 private:
  auto v() -> auto& {
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
template <typename V, typename Real, std::size_t N>
spacetime_vectorfield(vectorfield<V, Real, N> const&)
    -> spacetime_vectorfield<V>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t N>
spacetime_vectorfield(polymorphic::vectorfield<Real, N> const*)
    -> spacetime_vectorfield<polymorphic::vectorfield<Real, N> const*>;

//==============================================================================
// SYMBOLIC
//==============================================================================
#if TATOOINE_GINAC_AVAILABLE
#include "symbolic_field.h"
//==============================================================================
template <typename Real, std::size_t N>
struct spacetime_vectorfield<symbolic::field<Real, N - 1>, Real, N>
    : symbolic::field<Real, N> {
  //============================================================================
  using V           = symbolic::field<Real, N - 1>;
  using this_type   = spacetime_vectorfield<V, Real, N>;
  using parent_type = symbolic::field<Real, N>;
  using typename parent_type::pos_type;
  using typename parent_type::symtensor_type;
  using typename parent_type::tensor_type;

  //============================================================================
  spacetime_vectorfield(
      field<symbolic::field<Real, N - 1> const, Real, N - 1>& v) {
    symtensor_type ex;
    for (std::size_t i = 0; i < N - 1; ++i) {
      ex(i) = symbolic::ev(v.as_derived().expr()(i),
                           symbolic::symbol::t() == symbolic::symbol::x(N - 1));
    }
    ex(N - 1) = 1;
    this->set_expr(ex);
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Real, std::size_t N>
spacetime_vectorfield(symbolic::field<Real, N> const&)
    -> spacetime_vectorfield<symbolic::field<Real, N>>;
#endif
//==============================================================================
template <typename V, typename Real, std::size_t N>
auto spacetime(vectorfield<V, Real, N> const& vf) {
  return spacetime_vectorfield<V>{vf.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename Real, std::size_t N>
auto spacetime(polymorphic::vectorfield<Real, N> const* vf) {
  return spacetime_vectorfield<polymorphic::vectorfield<Real, N> const*>{vf};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename Real, std::size_t N>
auto spacetime(polymorphic::vectorfield<Real, N> const& vf) {
  return spacetime_vectorfield<polymorphic::vectorfield<Real, N> const*>{&vf};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
