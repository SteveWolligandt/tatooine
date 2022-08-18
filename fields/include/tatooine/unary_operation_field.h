#ifndef TATOOINE_UNARY_OPERATION_FIELD_H
#define TATOOINE_UNARY_OPERATION_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/field_type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename InternalField, typename Op>
struct unary_operation_field
    : field<unary_operation_field<InternalField, Op>,
            field_real_type<InternalField>, field_num_dimensions<InternalField>,
            std::invoke_result_t<Op, field_tensor_type<InternalField>>> {
 public:
  using this_type           = unary_operation_field<InternalField, Op>;
  using internal_field_t = InternalField;
  using raw_internal_field_t =
      std::decay_t<std::remove_pointer_t<internal_field_t>>;
  using parent_t =
      field<this_type, typename raw_internal_field_t::real_type,
            raw_internal_field_t::num_dimensions(),
            std::invoke_result_t<Op, typename raw_internal_field_t::tensor_type>>;
  using typename parent_t::pos_type;
  using typename parent_t::real_type;
  using typename parent_t::tensor_type;

 private:
  internal_field_t m_field;
  Op               m_op;

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
  constexpr auto evaluate(pos_type const& x, real_type const t) const
      -> tensor_type final {
    if constexpr (std::is_pointer_v<std::decay_t<InternalField>>) {
      return m_op(m_field->evaluate(x, t));
    } else {
      return m_op(m_field(x, t));
    }
  }
  auto internal_field() const -> decltype(auto) { return m_field; }
  auto internal_field() -> decltype(auto) { return m_field; }
};
//==============================================================================
template <typename Field, typename Real, typename Tensor, size_t N, typename Op>
constexpr auto make_unary_operation_field(
    field<Field, Real, N, Tensor> const& field, Op const& op) {
  return unary_operation_field<Field const&, Op const&>{field.as_derived(), op};
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, typename Tensor, typename Op>
constexpr auto make_unary_operation_field(field<Field, Real, N, Tensor>&& field,
                                          Op const&                       op) {
  return unary_operation_field<Field, Op const&>{std::move(field.as_derived()),
                                                 op};
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, typename Tensor, typename Op>
constexpr auto make_unary_operation_field(
    field<Field, Real, N, Tensor> const& field, Op&& op) {
  return unary_operation_field<Field const&, std::decay_t<Op>>{
      field.as_derived(), std::move(op)};
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, size_t N, typename Tensor, typename Op>
constexpr auto make_unary_operation_field(field<Field, Real, N, Tensor>&& field,
                                          Op&&                            op) {
  return unary_operation_field<Field, std::decay_t<Op>>{
      std::move(field.as_derived()), std::move(op)};
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename Tensor, typename Op>
constexpr auto make_unary_operation_field(
    polymorphic::field<Real, N, Tensor>* field, Op const& op) {
  return unary_operation_field<polymorphic::field<Real, N, Tensor>*, Op const&>{
      field, op};
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename Tensor, typename Op>
constexpr auto make_unary_operation_field(
    polymorphic::field<Real, N, Tensor>* field, Op&& op) {
  return unary_operation_field<polymorphic::field<Real, N, Tensor>*,
                               std::decay_t<Op>>{field->as_derived(),
                                                 std::move(op)};
}
//==============================================================================
template <typename Field, typename Real, typename Tensor, size_t N, typename Op>
constexpr auto operator|(field<Field, Real, N, Tensor> const& field,
                         Op const&                            op) {
  return make_unary_operation_field(field, op);
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, typename Tensor, size_t N, typename Op>
constexpr auto operator|(field<Field, Real, N, Tensor>&& field, Op const& op) {
  return make_unary_operation_field(std::move(field.as_derived()), op);
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, typename Tensor, size_t N, typename Op>
constexpr auto operator|(field<Field, Real, N, Tensor> const& field, Op&& op) {
  return make_unary_operation_field(field.as_derived(), std::move(op));
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, typename Tensor, size_t N, typename Op>
constexpr auto operator|(field<Field, Real, N, Tensor>&& field, Op&& op) {
  return make_unary_operation_field(std::move(field.as_derived()),
                                    std::move(op));
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename Tensor, typename Op>
constexpr auto operator|(polymorphic::field<Real, N, Tensor>* field,
                         Op const&                            op) {
  return make_unary_operation_field(field, op);
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename Tensor, typename Op>
constexpr auto operator|(polymorphic::field<Real, N, Tensor>* field, Op&& op) {
  return make_unary_operation_field(field, std::move(op));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
