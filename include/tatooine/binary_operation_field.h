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
            common_type<field_real_t<LHSInternalField>,
                        field_real_t<RHSInternalField>>,
            field_num_dimensions<LHSInternalField>,
            std::invoke_result_t<Op,
                                 field_tensor_t<LHSInternalField>,
                                 field_tensor_t<RHSInternalField>>> {
  static_assert(field_num_dimensions<LHSInternalField> ==
                field_num_dimensions<RHSInternalField>);

 public:
  using this_t = binary_operation_field<LHSInternalField, RHSInternalField, Op>;
  using parent_type =
      field<this_t,
            common_type<field_real_t<LHSInternalField>,
                        field_real_t<RHSInternalField>>,
            field_num_dimensions<LHSInternalField>,
            std::invoke_result_t<Op, field_tensor_t<LHSInternalField>,
                                 field_tensor_t<RHSInternalField>>>;
  using typename parent_type::pos_t;
  using typename parent_type::real_t;
  using typename parent_type::tensor_t;
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
  constexpr auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
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
  template <bool Cond = is_pointer<LHSInternalField>, enable_if<Cond> = true>
  auto set_v0(LHSInternalField lhs) -> void {
    m_lhs = lhs;
  }
  //----------------------------------------------------------------------------
  template <bool Cond = is_pointer<RHSInternalField>, enable_if<Cond> = true>
  auto set_v1(RHSInternalField rhs) -> void {
    m_rhs = rhs;
  }
  //----------------------------------------------------------------------------
  template <bool Cond = is_pointer<LHSInternalField>, enable_if<Cond> = true>
  auto fields_available() const -> bool {
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
