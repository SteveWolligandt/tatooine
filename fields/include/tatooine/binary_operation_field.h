#ifndef TATOOINE_BINARY_OPERATION_FIELD_H
#define TATOOINE_BINARY_OPERATION_FIELD_H
//==============================================================================
#include "field.h"
#include "unary_operation_field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename LHSInternalField, typename RHSInternalField, typename Op>
struct binary_operation_field
    : field<binary_operation_field<LHSInternalField, RHSInternalField, Op>,
            common_type<field_real_type<LHSInternalField>,
                        field_real_type<RHSInternalField>>,
            field_num_dimensions<LHSInternalField>,
            std::invoke_result_t<Op,
                                 field_tensor_type<LHSInternalField>,
                                 field_tensor_type<RHSInternalField>>> {
  static_assert(field_num_dimensions<LHSInternalField> ==
                field_num_dimensions<RHSInternalField>);

 public:
  using this_type = binary_operation_field<LHSInternalField, RHSInternalField, Op>;
  using parent_type =
      field<this_type,
            common_type<field_real_type<LHSInternalField>,
                        field_real_type<RHSInternalField>>,
            field_num_dimensions<LHSInternalField>,
            std::invoke_result_t<Op, field_tensor_type<LHSInternalField>,
                                 field_tensor_type<RHSInternalField>>>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  LHSInternalField m_lhs;
  RHSInternalField m_rhs;
  Op               m_op;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  constexpr binary_operation_field(binary_operation_field const& other)
      : m_lhs{other.m_lhs}, m_rhs{other.m_rhs}, m_op{other.m_op} {}
  //----------------------------------------------------------------------------
  constexpr binary_operation_field(binary_operation_field&& other) noexcept
      : m_lhs{std::move(other.m_lhs)},
        m_rhs{std::move(other.m_rhs)},
        m_op{std::move(other.m_op)} {}
  //----------------------------------------------------------------------------
  template <typename LHS, typename Rhs, typename Op_>
  constexpr binary_operation_field(LHS&& lhs, Rhs&& rhs, Op_&& op)
      : m_lhs{std::forward<LHSInternalField>(lhs)},
        m_rhs{std::forward<RHSInternalField>(rhs)},
        m_op{std::forward<Op>(op)} {}
  //----------------------------------------------------------------------------
  // assignement operators
  //----------------------------------------------------------------------------
 public:
  constexpr auto operator=(binary_operation_field const& other)
      -> binary_operation_field& {
    m_lhs = other.m_lhs;
    m_rhs = other.m_rhs;
    return *this;
  }
  constexpr auto operator=(binary_operation_field&& other) noexcept
      -> binary_operation_field& {
    m_lhs = std::move(other.m_lhs);
    m_rhs = std::move(other.m_rhs);
    return *this;
  }

 public:
  //----------------------------------------------------------------------------
  // dtor
  //----------------------------------------------------------------------------
  ~binary_operation_field() override = default;
  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& x, real_type const t) const
      -> tensor_type final {
    return m_op(lhs()(x, t), rhs()(x, t));
  }
  auto lhs() const -> auto const& {
    if constexpr (is_pointer<LHSInternalField>) {
      return *m_lhs;
    } else {
      return m_lhs;
    }
  }
  auto rhs() const -> auto const& {
    if constexpr (is_pointer<RHSInternalField>) {
      return *m_rhs;
    } else {
      return m_rhs;
    }
  }
  //----------------------------------------------------------------------------
  auto set_v0(LHSInternalField lhs)
      -> void requires(is_pointer<LHSInternalField>) {
    m_lhs = lhs;
  }
  //----------------------------------------------------------------------------
  auto set_v1(RHSInternalField rhs)
      -> void requires(is_pointer<RHSInternalField>) {
    m_rhs = rhs;
  }
  //----------------------------------------------------------------------------
  auto fields_available() const -> bool requires(
      is_pointer<LHSInternalField>&& is_pointer<RHSInternalField>) {
    return m_lhs != nullptr && m_rhs != nullptr;
  }
};
//==============================================================================
template <typename LHSInternalField, typename LHSReal, size_t N,
          typename LHSTensor, typename RHSInternalField, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    const field<LHSInternalField, LHSReal, N, LHSTensor>& lhs,
    const field<RHSInternalField, RHSReal, N, RHSTensor>& rhs, const Op& op) {
  return binary_operation_field<
      const field<LHSInternalField, LHSReal, N, LHSTensor>&,
      const field<RHSInternalField, RHSReal, N, RHSTensor>&, const Op&>{
      lhs, rhs, op};
}
//------------------------------------------------------------------------------
template <typename LHSInternalField, typename LHSReal, size_t N,
          typename LHSTensor, typename RHSInternalField, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    field<LHSInternalField, LHSReal, N, LHSTensor>&&      lhs,
    const field<RHSInternalField, RHSReal, N, RHSTensor>& rhs, const Op& op) {
  return binary_operation_field<
      field<LHSInternalField, LHSReal, N, LHSTensor>,
      const field<RHSInternalField, RHSReal, N, RHSTensor>&, const Op&>{
      std::move(lhs), rhs, op};
}
//------------------------------------------------------------------------------
template <typename LHSInternalField, typename LHSReal, size_t N,
          typename LHSTensor, typename RHSInternalField, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    const field<LHSInternalField, LHSReal, N, LHSTensor>& lhs,
    field<RHSInternalField, RHSReal, N, RHSTensor>&& rhs, const Op& op) {
  return binary_operation_field<
      const field<LHSInternalField, LHSReal, N, LHSTensor>&,
      field<RHSInternalField, RHSReal, N, RHSTensor>, const Op&>{
      lhs, std::move(rhs), op};
}
//------------------------------------------------------------------------------
template <typename LHSInternalField, typename LHSReal, size_t N,
          typename LHSTensor, typename RHSInternalField, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    field<LHSInternalField, LHSReal, N, LHSTensor>&& lhs,
    field<RHSInternalField, RHSReal, N, RHSTensor>&& rhs, const Op& op) {
  return binary_operation_field<field<LHSInternalField, LHSReal, N, LHSTensor>,
                                field<RHSInternalField, RHSReal, N, RHSTensor>,
                                const Op&>{std::move(lhs), std::move(rhs), op};
}
//------------------------------------------------------------------------------
template <typename LHSInternalField, typename LHSReal, size_t N,
          typename LHSTensor, typename RHSInternalField, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    const field<LHSInternalField, LHSReal, N, LHSTensor>& lhs,
    const field<RHSInternalField, RHSReal, N, RHSTensor>& rhs, Op&& op) {
  return binary_operation_field<
      const field<LHSInternalField, LHSReal, N, LHSTensor>&,
      const field<RHSInternalField, RHSReal, N, RHSTensor>&, Op>{lhs, rhs,
                                                                 std::move(op)};
}
//------------------------------------------------------------------------------
template <typename LHSInternalField, typename LHSReal, size_t N,
          typename LHSTensor, typename RHSInternalField, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    field<LHSInternalField, LHSReal, N, LHSTensor>&&      lhs,
    const field<RHSInternalField, RHSReal, N, RHSTensor>& rhs, Op&& op) {
  return binary_operation_field<
      field<LHSInternalField, LHSReal, N, LHSTensor>,
      const field<RHSInternalField, RHSReal, N, RHSTensor>&, Op>{
      std::move(lhs), rhs, std::move(op)};
}
//------------------------------------------------------------------------------
template <typename LHSInternalField, typename LHSReal, size_t N,
          typename LHSTensor, typename RHSInternalField, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    const field<LHSInternalField, LHSReal, N, LHSTensor>& lhs,
    field<RHSInternalField, RHSReal, N, RHSTensor>&& rhs, Op op) {
  return binary_operation_field<
      const field<LHSInternalField, LHSReal, N, LHSTensor>&,
      field<RHSInternalField, RHSReal, N, RHSTensor>, Op>{lhs, std::move(rhs),
                                                          std::move(op)};
}
//------------------------------------------------------------------------------
template <typename LHSInternalField, typename LHSReal, size_t N,
          typename LHSTensor, typename RHSInternalField, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    field<LHSInternalField, LHSReal, N, LHSTensor>&& lhs,
    field<RHSInternalField, RHSReal, N, RHSTensor>&& rhs, Op op) {
  return binary_operation_field<field<LHSInternalField, LHSReal, N, LHSTensor>,
                                field<RHSInternalField, RHSReal, N, RHSTensor>,
                                Op>{std::move(lhs), std::move(rhs),
                                    std::move(op)};
}
//------------------------------------------------------------------------------
template <typename LHSReal, size_t N, typename LHSTensor, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    polymorphic::field<LHSReal, N, LHSTensor> const* lhs,
    polymorphic::field<RHSReal, N, RHSTensor> const* rhs, Op op) {
  return binary_operation_field<
      polymorphic::field<LHSReal, N, LHSTensor> const*,
      polymorphic::field<RHSReal, N, RHSTensor> const*, Op>{lhs, rhs,
                                                            std::move(op)};
}
//------------------------------------------------------------------------------
template <typename RealOut, size_t NOut, size_t... TensorDimsOut,
          typename LHSReal, size_t N, typename LHSTensor, typename RHSReal,
          typename RHSTensor, typename Op>
constexpr auto make_binary_operation_field(
    polymorphic::field<LHSReal, N, LHSTensor> const& lhs,
    polymorphic::field<RHSReal, N, RHSTensor> const& rhs, Op op) {
  return make_binary_operation_field(&lhs, &rhs, std::forward<Op>(op));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
