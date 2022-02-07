#ifndef TATOOINE_DIFFERENTIATED_FIELD_H
#define TATOOINE_DIFFERENTIATED_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/field_type_traits.h>
#include <tatooine/packages.h>
#include <tatooine/tensor_type_operations.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename InternalField>
struct differentiated_field
    : field<differentiated_field<InternalField>, field_real_t<InternalField>,
            field_num_dimensions<InternalField>,
            tensor_add_dimension_right_t<field_num_dimensions<InternalField>,
                                         field_tensor_t<InternalField>>> {
  using raw_internal_field_t =
      std::decay_t<std::remove_pointer_t<InternalField>>;
  static constexpr auto holds_field_pointer = is_pointer<InternalField>;

  using this_t   = differentiated_field<InternalField>;
  using parent_type = field<
      this_t, typename raw_internal_field_t::real_t,
      raw_internal_field_t::num_dimensions(),
      tensor_add_dimension_right_t<raw_internal_field_t::num_dimensions(),
                                   typename raw_internal_field_t::tensor_t>>;
  using parent_type::num_dimensions;
  using typename parent_type::pos_t;
  using typename parent_type::real_t;
  using vec_t = vec<real_t, num_dimensions()>;
  using typename parent_type::tensor_t;

  // static_assert(raw_internal_field_t::tensor_rank() == 1);
  // static_assert(tensor_t::rank() == 2);
  // static_assert(raw_internal_field_t::tensor_rank() + 1 ==
  //              parent_type::tensor_rank());
  //============================================================================
 private:
  InternalField m_internal_field;
  vec_t         m_eps;
  //============================================================================
 public:
  template <typename Field_, arithmetic Eps>
  differentiated_field(Field_&& f, Eps const eps)
      : m_internal_field{std::forward<Field_>(f)}, m_eps{tag::fill{eps}} {
  }
  //----------------------------------------------------------------------------
  explicit differentiated_field(
      vec_t const& eps = vec_t::ones() * 1e-7) requires holds_field_pointer
      : m_internal_field{nullptr},
        m_eps{eps} {}
  //----------------------------------------------------------------------------
  template <typename Field_>
  differentiated_field(Field_&& f, vec_t const& eps)
      : m_internal_field{std::forward<Field_>(f)}, m_eps{eps} {}
  //----------------------------------------------------------------------------
  template <typename Field_, arithmetic Real>
  differentiated_field(Field_&& f, vec<Real, num_dimensions()> const& eps)
      : m_internal_field{std::forward<Field_>(f)}, m_eps{eps} {
  }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    auto derivative = tensor_t{};
    auto offset     = pos_t{};
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      offset(i) = m_eps(i);
      auto x0   = x - offset;
      auto x1   = x + offset;
      auto dx   = 2 * m_eps(i);
      if (!internal_field().in_domain(x0, t)) {
        x0 = x;
        dx = m_eps(i);
      }
      if (!internal_field().in_domain(x1, t)) {
        x1 = x;
        dx = m_eps(i);
      }
      constexpr std::size_t slice_dim = tensor_t::rank() - 1;
      derivative.template slice<slice_dim>(i) =
          (internal_field()(x1, t) - internal_field()(x0, t)) / dx;
      offset(i) = 0;
    }

    return derivative;
  }
  //----------------------------------------------------------------------------
  auto set_eps(vec_t const& eps) { m_eps = eps; }
  auto set_eps(vec_t&& eps) { m_eps = std::move(eps); }
  auto set_eps(real_t eps) { m_eps = vec_t{tag::fill{eps}}; }
  auto eps() -> auto& { return m_eps; }
  auto eps() const -> auto const& { return m_eps; }
  auto eps(std::size_t i) -> auto& { return m_eps(i); }
  auto eps(std::size_t i) const { return m_eps(i); }
  //----------------------------------------------------------------------------
  auto internal_field() const -> auto const& {
    if constexpr (holds_field_pointer) {
      return *m_internal_field;
    } else {
      return m_internal_field;
    }
  }
  //----------------------------------------------------------------------------
  auto set_internal_field(InternalField f)
      -> void requires(holds_field_pointer) {
    m_internal_field = f;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Field, typename Real, std::size_t NumDimensions,
          typename Tensor>
auto diff(field<Field, Real, NumDimensions, Tensor> const& f, Real const eps) {
  return differentiated_field<Field const&>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, std::size_t NumDimensions,
          typename Tensor>
auto diff(field<Field, Real, NumDimensions, Tensor>& f, Real const eps) {
  return differentiated_field<Field&>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, std::size_t NumDimensions,
          typename Tensor>
auto diff(field<Field, Real, NumDimensions, Tensor>&& f, Real const eps) {
  return differentiated_field<Field>{std::move(f.as_derived()), eps};
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, std::size_t NumDimensions,
          typename Tensor>
auto diff(field<Field, Real, NumDimensions, Tensor> const& f,
          vec<Real, NumDimensions> const&                  eps) {
  return differentiated_field<Field const&>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, std::size_t NumDimensions,
          typename Tensor>
auto diff(field<Field, Real, NumDimensions, Tensor>& f,
          vec<Real, NumDimensions> const&            eps) {
  return differentiated_field<Field&>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, std::size_t NumDimensions,
          typename Tensor>
auto diff(field<Field, Real, NumDimensions, Tensor>&& f,
          vec<Real, NumDimensions> const&             eps) {
  return differentiated_field<Field>{std::move(f.as_derived()), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t NumDimensions, typename Tensor>
auto diff(polymorphic::field<Real, NumDimensions, Tensor>* f,
          vec<Real, NumDimensions> const&                  eps) {
  return differentiated_field<polymorphic::field<Real, NumDimensions, Tensor>*>{
      f, eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t NumDimensions, typename Tensor>
auto diff(polymorphic::field<Real, NumDimensions, Tensor> const* f,
          vec<Real, NumDimensions> const&                        eps) {
  return differentiated_field<
      polymorphic::field<Real, NumDimensions, Tensor> const*>{f, eps};
}
//==============================================================================
template <typename InternalField>
struct time_differentiated_field
    : field<time_differentiated_field<InternalField>,
            typename std::decay_t<InternalField>::real_t,
            std::decay_t<InternalField>::num_dimensions(),
            typename std::decay_t<InternalField>::tensor_t> {
  using this_t   = time_differentiated_field<InternalField>;
  using parent_type = field<this_t, typename std::decay_t<InternalField>::real_t,
                         std::decay_t<InternalField>::num_dimensions(),
                         typename std::decay_t<InternalField>::tensor_t>;
  using parent_type::num_dimensions;
  using typename parent_type::pos_t;
  using typename parent_type::real_t;
  using vec_t = vec<real_t, num_dimensions()>;
  using typename parent_type::tensor_t;
  static constexpr auto holds_field_pointer = is_pointer<InternalField>;

  //============================================================================
 private:
  InternalField m_internal_field;
  real_t        m_eps;
  //============================================================================
 public:
  template <typename Field_, arithmetic Eps>
  time_differentiated_field(Field_&& f, Eps const eps)
      : m_internal_field{std::forward<Field_>(f)},
        m_eps{static_cast<real_t>(eps)} {
  }
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
template <typename Field, typename Real, std::size_t NumDimensions,
          typename Tensor>
auto diff_time(field<Field, Real, NumDimensions, Tensor> const& f,
               Real const                                       eps) {
  return time_differentiated_field<Field const&>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, std::size_t NumDimensions,
          typename Tensor>
auto diff_time(field<Field, Real, NumDimensions, Tensor>& f, Real const eps) {
  return time_differentiated_field<Field&>{f.as_derived(), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, std::size_t NumDimensions,
          typename Tensor>
auto diff_time(field<Field, Real, NumDimensions, Tensor>&& f, Real const eps) {
  return time_differentiated_field<Field>{std::move(f.as_derived()), eps};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t NumDimensions, typename Tensor>
auto diff_time(polymorphic::field<Real, NumDimensions, Tensor> const* f,
               Real const                                             eps) {
  return time_differentiated_field<
      polymorphic::field<Real, NumDimensions, Tensor> const*>{f, eps};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
