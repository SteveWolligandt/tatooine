#ifndef TATOOINE_DIFFERENTIATED_FIELD_H
#define TATOOINE_DIFFERENTIATED_FIELD_H
//==============================================================================
#include <tatooine/packages.h>

#include "field.h"
#include "utility.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Field, size_t... TensorDims>
struct differentiated_field
    : field<differentiated_field<Field, TensorDims...>,
            typename std::decay_t<std::remove_pointer_t<Field>>::real_t,
            std::decay_t<std::remove_pointer_t<Field>>::num_dimensions(),
            TensorDims...> {
  using raw_field_t = std::decay_t<std::remove_pointer_t<Field>>;
  static constexpr auto holds_field_pointer = is_pointer<Field>;

  using this_t      = differentiated_field<Field, TensorDims...>;
  using parent_t    = field<this_t, typename raw_field_t::real_t,
                         raw_field_t::num_dimensions(), TensorDims...>;
  using parent_t::num_dimensions;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using vec_t = vec<real_t, num_dimensions()>;
  using typename parent_t::tensor_t;

  static_assert(raw_field_t::tensor_t::rank() == 1);
  static_assert(tensor_t::rank() == 2);
  static_assert(raw_field_t::tensor_t::rank() + 1 ==
                tensor_t::rank());
  //============================================================================
 private:
  Field m_internal_field;
  vec_t m_eps;
  //============================================================================
 public:
#ifdef __cpp_concpets
  template <typename Field_, arithmetic Eps>
#else
  template <typename Field_, typename Eps, enable_if<is_arithmetic<Eps>> = true>
#endif
  differentiated_field(Field_&& f, Eps const eps)
      : m_internal_field{std::forward<Field_>(f)}, m_eps{tag::fill{eps}} {
  }
  //----------------------------------------------------------------------------
  template <bool h = holds_field_pointer, enable_if<h> = true>
  explicit differentiated_field(vec_t const& eps = vec_t::ones() * 1e-7)
      : m_internal_field{nullptr}, m_eps{eps} {}
  //----------------------------------------------------------------------------
  template <typename Field_>
  differentiated_field(Field_&& f, vec_t const& eps)
      : m_internal_field{std::forward<Field_>(f)}, m_eps{eps} {}
  //----------------------------------------------------------------------------
#ifdef __cpp_concpets
  template <typename Field_, arithmetic Real>
#else
  template <typename Field_, typename Real, enable_if<is_arithmetic<Real>> = true>
#endif
  differentiated_field(Field_&& f, vec<Real, num_dimensions()> const& eps)
      : m_internal_field{std::forward<Field_>(f)}, m_eps{eps} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    tensor_t derivative;

    pos_t offset;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      offset(i) = m_eps(i);
      auto x0   = x - offset;
      auto x1   = x + offset;
      auto dx   = 2 * m_eps;
      if (!internal_field().in_domain(x0, t)) {
        x0 = x;
        dx = m_eps;
      }
      if (!internal_field().in_domain(x1, t)) {
        x1 = x;
        dx = m_eps;
      }
      derivative.template slice<sizeof...(TensorDims) - 1>(i) =
          (internal_field()(x1, t) - internal_field()(x0, t)) / dx;
      offset(i) = 0;
    }

    return derivative;
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& x, real_t t) const -> bool final {
    return internal_field().in_domain(x, t);
  }
  //----------------------------------------------------------------------------
  auto set_eps(vec_t const& eps) { m_eps = eps; }
  auto set_eps(vec_t&& eps) { m_eps = std::move(eps); }
  auto set_eps(real_t eps) { m_eps = vec_t{tag::fill{eps}}; }
  auto eps() -> auto& { return m_eps; }
  auto eps() const -> auto const& { return m_eps; }
  auto eps(size_t i) -> auto& { return m_eps(i); }
  auto eps(size_t i) const { return m_eps(i); }
  //----------------------------------------------------------------------------
  auto internal_field() const -> auto const& {
    if constexpr (holds_field_pointer) {
      return *m_internal_field;
    } else {
      return m_internal_field;
    }
  }
  //----------------------------------------------------------------------------
  template <bool h = holds_field_pointer, enable_if<h> = true>
  auto set_internal_field(Field f) -> void {
    m_internal_field = f;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff(field<Field, Real, N, TensorDims...> const& f, Real const eps) {
  return differentiated_field<Field const&, TensorDims..., N>{f.as_derived(),
                                                              eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff(field<Field, Real, N, TensorDims...>& f, Real const eps) {
  return differentiated_field<Field&, TensorDims..., N>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff(field<Field, Real, N, TensorDims...>&& f, Real const eps) {
  return differentiated_field<Field, TensorDims..., N>{
      std::move(f.as_derived()), eps};
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff(field<Field, Real, N, TensorDims...> const& f,
          vec<Real, N> const&                         eps) {
  return differentiated_field<Field const&, TensorDims..., N>{f.as_derived(),
                                                              eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff(field<Field, Real, N, TensorDims...>& f, vec<Real, N> const& eps) {
  return differentiated_field<Field&, TensorDims..., N>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff(field<Field, Real, N, TensorDims...>&& f, vec<Real, N> const& eps) {
  return differentiated_field<Field, TensorDims..., N>{
      std::move(f.as_derived()), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, size_t... TensorDims>
auto diff(parent::field<Real, N, TensorDims...>* f, vec<Real, N> const& eps) {
  return differentiated_field<parent::field<Real, N, TensorDims...>*,
                              TensorDims..., N>{f, eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, size_t... TensorDims>
auto diff(parent::field<Real, N, TensorDims...> const* f,
          vec<Real, N> const&                          eps) {
  return differentiated_field<parent::field<Real, N, TensorDims...> const*,
                              TensorDims..., N>{f, eps};
}
//==============================================================================
template <typename Field, size_t... TensorDims>
struct time_differentiated_field
    : field<time_differentiated_field<Field, TensorDims...>,
            typename std::decay_t<Field>::real_t,
            std::decay_t<Field>::num_dimensions(), TensorDims...> {
  using this_t   = time_differentiated_field<Field, TensorDims...>;
  using parent_t = field<this_t, typename std::decay_t<Field>::real_t,
                         std::decay_t<Field>::num_dimensions(), TensorDims...>;
  using parent_t::num_dimensions;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using vec_t = vec<real_t, num_dimensions()>;
  using typename parent_t::tensor_t;
  static constexpr auto holds_field_pointer = is_pointer<Field>;

  //============================================================================
 private:
  Field m_internal_field;
  real_t m_eps;
  //============================================================================
 public:
#ifdef __cpp_concpets
  template <typename Field_, arithmetic Eps>
#else
  template <typename Field_, typename Eps,
            enable_if<is_arithmetic<Eps> > = true>
#endif
  time_differentiated_field(Field_&& f, Eps const eps)
      : m_internal_field{std::forward<Field_>(f)},
        m_eps{static_cast<real_t>(eps)} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    real_t t0 = t - m_eps;
    real_t t1 = t + m_eps;
    auto   dt = 2 * m_eps;
    if (!internal_field().in_domain(x, t0)) {
      t0 = t;
      dt = m_eps;
    }
    if (!internal_field().in_domain(x, t1)) {
      t1 = t;
      dt = m_eps;
    }
    return (internal_field()(x, t1) - internal_field()(x, t0)) / dt;
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& x, real_t t) const -> bool final {
    return internal_field().in_domain(x, t);
  }
  //----------------------------------------------------------------------------
  auto set_eps(real_t eps) { m_eps = eps; }
  auto eps() -> auto& { return m_eps; }
  auto eps() const -> auto const& { return m_eps; }
  //----------------------------------------------------------------------------
  auto internal_field() const -> auto const& {
    if constexpr (holds_field_pointer) {
      return *m_internal_field;
    } else {
      return m_internal_field;
    }
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff_time(field<Field, Real, N, TensorDims...> const& f, Real const eps) {
  return time_differentiated_field<Field const&, TensorDims...>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff_time(field<Field, Real, N, TensorDims...>& f, Real const eps) {
  return time_differentiated_field<Field&, TensorDims...>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, size_t N, size_t... TensorDims>
auto diff_time(field<Field, Real, N, TensorDims...>&& f, Real const eps) {
  return time_differentiated_field<Field, TensorDims...>{
      std::move(f.as_derived()), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, size_t... TensorDims>
auto diff_time(parent::field<Real, N, TensorDims...>const* f, Real const eps) {
  return time_differentiated_field<parent::field<Real, N, TensorDims...> const*,
                                   TensorDims...>{f, eps};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
