#ifndef TATOOINE_UNARY_OPERATION_FIELD_H
#define TATOOINE_UNARY_OPERATION_FIELD_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Field, typename Op, typename Real, size_t N,
          size_t... TensorDims>
struct unary_operation_field
    : field<unary_operation_field<Field, Op, Real, N, TensorDims...>, Real, N,
            TensorDims...> {
 public:
  using this_t   = unary_operation_field<Field, Op, Real, N, TensorDims...>;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using internal_field_t = Field;

 private:
  Field m_field;
  Op    m_op;

 public:
  constexpr unary_operation_field(unary_operation_field const&)     = default;
  constexpr unary_operation_field(unary_operation_field&&) noexcept = default;
  template <typename Field_, typename Op_>
  constexpr unary_operation_field(Field_&& field, Op_&& op)
      : m_field{std::forward<Field_>(field)}, m_op{std::forward<Op_>(op)} {}

 public:
  constexpr auto operator       =(unary_operation_field const&)
      -> unary_operation_field& = default;
  constexpr auto operator       =(unary_operation_field&&) noexcept
      -> unary_operation_field& = default;

 public:
  ~unary_operation_field() override = default;
  //============================================================================
  constexpr auto evaluate(pos_t const& x, Real t) const -> tensor_t final {
    if constexpr (std::is_pointer_v<std::decay_t<Field>>) {
      return m_op(m_field->evaluate(x, t));
    } else {
      return m_op(m_field(x, t));
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& x, Real t) const -> bool final {
    if constexpr (std::is_pointer_v<std::decay_t<Field>>) {
      return m_field->in_domain(x, t);
    } else {
      return m_field.in_domain(x, t);
    }
  }

  auto internal_field() const -> decltype(auto) { return m_field; }
  auto internal_field() -> decltype(auto) { return m_field; }
};
//==============================================================================
template <typename Field, typename Op>
struct unary_operation_field_builder {
  using transformed_tensor_t = std::invoke_result_t<
      Op, typename std::remove_pointer_t<std::decay_t<Field>>::tensor_t>;

 private:
  template <typename Field_, typename Op_, typename Tensor, typename Real,
            size_t... Dims>
  static auto build(Field_&& field, Op_&& op,
                    base_tensor<Tensor, Real, Dims...> const&) {
    return unary_operation_field<
        Field, Op, typename std::remove_pointer_t<std::decay_t<Field>>::real_t,
        std::remove_pointer_t<std::decay_t<Field>>::num_dimensions(), Dims...>{
        std::forward<Field_>(field), std::forward<Op_>(op)};
  }
#ifdef __cpp_concpets
  template <typename Field_, typename Op_, arithmetic T>
#else
  template <typename Field_, typename Op_, typename T,
            enable_if<is_arithmetic<T>> = true>
#endif
  static auto build(Field_&& field, Op_&& op, T const&) {
    return unary_operation_field<
        Field, Op, typename std::remove_pointer_t<std::decay_t<Field>>::real_t,
        std::remove_pointer_t<std::decay_t<Field>>::num_dimensions()>{
        std::forward<Field_>(field), std::forward<Op_>(op)};
  }

 public:
  using type = decltype(unary_operation_field_builder<Field, Op>::build(
      std::declval<Field>(), std::declval<Op>(),
      std::declval<transformed_tensor_t>()));
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Op>
using unary_operation_field_builder_t =
    typename unary_operation_field_builder<Field, Op>::type;
//==============================================================================
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto make_unary_operation_field(
    field<Field, Real, N, TensorDims...> const& field, Op const& op) {
  return unary_operation_field_builder_t<Field const&, Op const&>{
      field.as_derived(), op};
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto make_unary_operation_field(
    field<Field, Real, N, TensorDims...>&& field, Op const& op) {
  return unary_operation_field_builder_t<Field, Op const&>{
      std::move(field.as_derived()), op};
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto make_unary_operation_field(
    field<Field, Real, N, TensorDims...> const& field, Op&& op) {
  return unary_operation_field_builder_t<Field const&, std::decay_t<Op>>{
      field.as_derived(), std::move(op)};
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto make_unary_operation_field(
    field<Field, Real, N, TensorDims...>&& field, Op&& op) {
  return unary_operation_field_builder_t<Field, std::decay_t<Op>>{
      std::move(field.as_derived()), std::move(op)};
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(
    parent::field<Real, N, TensorDims...>* field, Op const& op) {
  return unary_operation_field_builder_t<parent::field<Real, N, TensorDims...>*,
                                         Op const&>{field, op};
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(
    parent::field<Real, N, TensorDims...>* field, Op&& op) {
  return unary_operation_field_builder_t<parent::field<Real, N, TensorDims...>*,
                                         std::decay_t<Op>>{field->as_derived(),
                                                           std::move(op)};
}
//==============================================================================
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto operator|(field<Field, Real, N, TensorDims...> const& field,
                         Op const&                                   op) {
  return make_unary_operation_field(field, op);
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto operator|(field<Field, Real, N, TensorDims...>&& field,
                         Op const&                              op) {
  return make_unary_operation_field(std::move(field.as_derived()), op);
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto operator|(field<Field, Real, N, TensorDims...> const& field,
                         Op&&                                        op) {
  return make_unary_operation_field(field.as_derived(), std::move(op));
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto operator|(field<Field, Real, N, TensorDims...>&& field,
                         Op&&                                   op) {
  return make_unary_operation_field(std::move(field.as_derived()),
                                    std::move(op));
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto operator|(parent::field<Real, N, TensorDims...>* field,
                         Op const&                              op) {
  return make_unary_operation_field(field, op);
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, size_t... TensorDims,
          typename Op>
constexpr auto operator|(parent::field<Real, N, TensorDims...>* field,
                         Op&&                                   op) {
  return make_unary_operation_field(field, std::move(op));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
