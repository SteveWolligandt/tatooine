#ifndef TATOOINE_UNARY_OPERATION_FIELD_H
#define TATOOINE_UNARY_OPERATION_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/field_type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Field, typename Op>
requires (is_field<Field>) &&
         (invocable<Op, field_tensor_type<Field>>)
struct unary_operation_field
    : field<unary_operation_field<Field, Op>, field_real_type<Field>,
            field_num_dimensions<Field>,
            std::invoke_result_t<Op, field_tensor_type<Field>>> {
 public:
  using this_type           = unary_operation_field<Field, Op>;
  using internal_field_type = Field;
  using parent_type =
      field<this_type, field_real_type<Field>, field_num_dimensions<Field>,
            std::invoke_result_t<Op, field_tensor_type<Field>>>;

  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;

 private:
  Field m_field;
  Op    m_op;

 public:
  template <convertible_to<Field> Field_, convertible_to<Op> Op_>
  constexpr unary_operation_field(Field_&& field, Op_&& op)
      : m_field{std::forward<Field_>(field)}, m_op{std::forward<Op_>(op)} {}
  //----------------------------------------------------------------------------
  template <convertible_to<Op> Op_>
  constexpr unary_operation_field(Op_&& op)
  requires std::is_pointer_v<std::decay_t<Field>>
    : m_field{nullptr}, m_op{std::forward<Op_>(op)} {}
  //----------------------------------------------------------------------------
  constexpr unary_operation_field(unary_operation_field const&) = default;
  //----------------------------------------------------------------------------
  constexpr unary_operation_field(unary_operation_field&&) noexcept = default;
  //----------------------------------------------------------------------------
  constexpr auto operator=(unary_operation_field const&)
      -> unary_operation_field& = default;
  //----------------------------------------------------------------------------
  constexpr auto operator=(unary_operation_field&&) noexcept
      -> unary_operation_field& = default;
  //----------------------------------------------------------------------------
  ~unary_operation_field() override = default;
  //============================================================================
  constexpr auto evaluate(pos_type const& x, real_type const t) const
      -> tensor_type {
    if constexpr (std::is_pointer_v<std::decay_t<Field>>) {
      return m_op(m_field->evaluate(x, t));
    } else {
      return m_op(m_field(x, t));
    }
  }
  //============================================================================
  auto internal_field() const -> decltype(auto) { return m_field; }
  auto internal_field() -> decltype(auto) { return m_field; }
};
//==============================================================================
template <typename Field, typename Op>
requires (!std::is_pointer_v<std::decay_t<Field>>)
unary_operation_field(Field&& field, Op&& op)
    -> unary_operation_field<Field, Op>;
//------------------------------------------------------------------------------
template <typename Field, typename Op>
requires (!std::is_pointer_v<Field>)
unary_operation_field(Field const& field, Op&& op)
    -> unary_operation_field<Field const&, Op>;
//------------------------------------------------------------------------------
template <typename Field, typename Op>
requires (!std::is_pointer_v<Field>)
unary_operation_field(Field& field, Op&& op)
    -> unary_operation_field<Field&, Op>;
//==============================================================================
template <typename Field, typename Op>
requires (!std::is_pointer_v<std::decay_t<Field>>)
unary_operation_field(Field&& field, Op const& op)
    -> unary_operation_field<Field, Op const&>;
//------------------------------------------------------------------------------
template <typename Field, typename Op>
requires (!std::is_pointer_v<Field>)
unary_operation_field(Field const& field, Op const& op)
    -> unary_operation_field<Field const&, Op const&>;
//------------------------------------------------------------------------------
template <typename Field, typename Op>
requires (!std::is_pointer_v<Field>)
unary_operation_field(Field& field, Op const& op)
    -> unary_operation_field<Field&, Op const&>;
//==============================================================================
template <typename Field, typename Op>
requires (!std::is_pointer_v<std::decay_t<Field>>)
unary_operation_field(Field&& field, Op& op)
    -> unary_operation_field<Field, Op&>;
//------------------------------------------------------------------------------
template <typename Field, typename Op>
requires (!std::is_pointer_v<Field>)
unary_operation_field(Field const& field, Op& op)
    -> unary_operation_field<Field const&, Op&>;
//------------------------------------------------------------------------------
template <typename Field, typename Op>
requires (!std::is_pointer_v<Field>)
unary_operation_field(Field& field, Op& op)
    -> unary_operation_field<Field&, Op&>;
//==============================================================================
template <typename Field, typename Op>
unary_operation_field(Field* field, Op&& op)
    -> unary_operation_field<Field*, Op>;
//------------------------------------------------------------------------------
template <typename Field, typename Op>
unary_operation_field(Field const* field, Op&& op)
    -> unary_operation_field<Field const*, Op>;
//==============================================================================
template <typename Field, typename Op>
requires (is_field<Field>) &&
         (invocable<Op, field_tensor_type<Field>>)
constexpr auto operator|(Field&& field, Op&& op) {
  return unary_operation_field{std::forward<Field>(field),
                               std::forward<Op>(op)};
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename Tensor, typename Op>
requires (invocable<Op, Tensor>)
constexpr auto operator|(polymorphic::field<Real, N, Tensor>* field, Op&& op) {
  return unary_operation_field{field, std::forward<Op>(op)};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
