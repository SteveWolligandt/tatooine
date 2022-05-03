#ifndef TATOOINE_DIFFERENTIATED_FIELD_H
#define TATOOINE_DIFFERENTIATED_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/field_concept.h>
#include <tatooine/field_type_traits.h>
#include <tatooine/available_libraries.h>
#include <tatooine/tensor_type_operations.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <field_concept InternalField>
struct numerically_differentiated_field
    : field<numerically_differentiated_field<InternalField>,
            field_real_type<InternalField>, field_num_dimensions<InternalField>,
            tensor_add_dimension_right<field_num_dimensions<InternalField>,
                                       field_tensor_type<InternalField>>> {
  using raw_internal_field_t =
      std::decay_t<std::remove_pointer_t<InternalField>>;
  static constexpr auto holds_field_pointer = is_pointer<InternalField>;

  using this_type   = numerically_differentiated_field<InternalField>;
  using parent_type = field<
      this_type, typename raw_internal_field_t::real_type,
      raw_internal_field_t::num_dimensions(),
      tensor_add_dimension_right<raw_internal_field_t::num_dimensions(),
                                 typename raw_internal_field_t::tensor_type>>;
  using parent_type::num_dimensions;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using vec_type = vec<real_type, num_dimensions()>;
  using typename parent_type::tensor_type;

  // static_assert(raw_internal_field_t::tensor_rank() == 1);
  // static_assert(tensor_type::rank() == 2);
  // static_assert(raw_internal_field_t::tensor_rank() + 1 ==
  //              parent_type::tensor_rank());
  //============================================================================
 private:
  InternalField m_internal_field;
  vec_type      m_eps;
  //============================================================================
 public:
  template <typename Field_, arithmetic Eps>
  numerically_differentiated_field(Field_&& f, Eps const eps)
      : m_internal_field{std::forward<Field_>(f)}, m_eps{tag::fill{eps}} {}
  //----------------------------------------------------------------------------
  explicit numerically_differentiated_field(
      vec_type const& eps = vec_type::ones() *
                            1e-7) requires holds_field_pointer
      : m_internal_field{nullptr},
        m_eps{eps} {}
  //----------------------------------------------------------------------------
  template <typename Field_>
  numerically_differentiated_field(Field_&& f, vec_type const& eps)
      : m_internal_field{std::forward<Field_>(f)}, m_eps{eps} {}
  //----------------------------------------------------------------------------
  template <typename Field_, arithmetic Real>
  numerically_differentiated_field(Field_&&                           f,
                                   vec<Real, num_dimensions()> const& eps)
      : m_internal_field{std::forward<Field_>(f)}, m_eps{eps} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& x, real_type const t) const
      -> tensor_type {
    auto derivative = tensor_type{};
    auto offset     = pos_type{};
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
      constexpr std::size_t slice_dim = tensor_type::rank() - 1;
      derivative.template slice<slice_dim>(i) =
          (internal_field()(x1, t) - internal_field()(x0, t)) / dx;
      offset(i) = 0;
    }

    return derivative;
  }
  //----------------------------------------------------------------------------
  auto set_eps(vec_type const& eps) { m_eps = eps; }
  auto set_eps(vec_type&& eps) { m_eps = std::move(eps); }
  auto set_eps(real_type eps) { m_eps = vec_type{tag::fill{eps}}; }
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
//==============================================================================
auto diff(field_concept auto&& field, tag::numerical_t /*tag*/) {
  return numerically_differentiated_field{std::forward<decltype(field)>(field)};
}
//------------------------------------------------------------------------------
auto diff(field_concept auto&&  field, tag::numerical_t /*tag*/,
          arithmetic auto const epsilon) {
  return numerically_differentiated_field{std::forward<decltype(field)>(field),
                                          epsilon};
}
//------------------------------------------------------------------------------
auto diff(field_concept auto&& field, arithmetic auto const epsilon) {
  return diff(std::forward<decltype(field)>(field), tag::numerical, epsilon);
}
//------------------------------------------------------------------------------
auto diff(
    field_concept auto&& field, tag::numerical_t /*tag*/,
    fixed_size_real_vec<std::decay_t<decltype(field)>::num_dimensions()> auto&&
        epsilon) {
  return numerically_differentiated_field{
      std::forward<decltype(field)>(field),
      std::forward<decltype(epsilon)>(epsilon)};
}
//------------------------------------------------------------------------------
auto diff(
    field_concept auto&& field,
    fixed_size_real_vec<std::decay_t<decltype(field)>::num_dimensions()> auto&&
        epsilon) {
  return diff(std::forward<decltype(field)>(field), tag::numerical,
              std::forward<decltype(epsilon)>(epsilon));
}
//==============================================================================
template <typename InternalField>
struct differentiated_field : numerically_differentiated_field<InternalField> {
  using parent_type = numerically_differentiated_field<InternalField>;
  using parent_type::parent_type;
};
//==============================================================================
auto diff(field_concept auto&& field) {
  return differentiated_field{std::forward<decltype(field)>(field)};
}
//==============================================================================
template <typename InternalField>
struct time_differentiated_field
    : field<time_differentiated_field<InternalField>,
            typename std::decay_t<InternalField>::real_type,
            std::decay_t<InternalField>::num_dimensions(),
            typename std::decay_t<InternalField>::tensor_type> {
  using this_type = time_differentiated_field<InternalField>;
  using parent_type =
      field<this_type, typename std::decay_t<InternalField>::real_type,
            std::decay_t<InternalField>::num_dimensions(),
            typename std::decay_t<InternalField>::tensor_type>;
  using parent_type::num_dimensions;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using vec_type = vec<real_type, num_dimensions()>;
  using typename parent_type::tensor_type;
  static constexpr auto holds_field_pointer = is_pointer<InternalField>;

  //============================================================================
 private:
  InternalField m_internal_field;
  real_type     m_eps;
  //============================================================================
 public:
  template <typename Field_, arithmetic Eps>
  time_differentiated_field(Field_&& f, Eps const eps)
      : m_internal_field{std::forward<Field_>(f)},
        m_eps{static_cast<real_type>(eps)} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& x, real_type const t) const
      -> tensor_type final {
    real_type t0 = t - m_eps;
    real_type t1 = t + m_eps;
    auto      dt = 2 * m_eps;
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
  constexpr auto in_domain(pos_type const& x, real_type t) const -> bool final {
    return internal_field().in_domain(x, t);
  }
  //----------------------------------------------------------------------------
  auto set_eps(real_type eps) { m_eps = eps; }
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
