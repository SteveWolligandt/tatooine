#ifndef TATOOINE_BINARY_OPERATION_FIELD_H
#define TATOOINE_BINARY_OPERATION_FIELD_H
//==============================================================================
#include "field.h"
#include "unary_operation_field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
//│ binary_operation_field                                                   │
template <typename V0, typename V1, typename Op, typename Real, size_t N,
          size_t... TensorDims>
struct binary_operation_field
    : field<binary_operation_field<V0, V1, Op, Real, N, TensorDims...>, Real, N,
            TensorDims...> {
 public:
  using this_t   = binary_operation_field<V0, V1, Op, Real, N, TensorDims...>;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ members                                                              │
  //├──────────────────────────────────────────────────────────────────────┤
 private:
  V0 m_v0;
  V1 m_v1;
  Op m_op;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ ctors                                                                │
  //├──────────────────────────────────────────────────────────────────────┤
 public:
  constexpr binary_operation_field(const binary_operation_field&) = default;
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  constexpr binary_operation_field(binary_operation_field&&) noexcept = default;
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  template <typename V0_, typename V1_, typename Op_>
  constexpr binary_operation_field(V0_&& v0, V1_&& v1, Op_&& op)
      : m_v0{std::forward<V0>(v0)},
        m_v1{std::forward<V1>(v1)},
        m_op{std::forward<Op>(op)} {}
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ assignment operators                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
 public:
  constexpr auto operator=(const binary_operation_field&)
    -> binary_operation_field& = default;
  constexpr auto operator=(binary_operation_field &&) noexcept
    ->binary_operation_field& = default;
 public:
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ dtor                                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  ~binary_operation_field() override = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ methods                                                              │
  //├──────────────────────────────────────────────────────────────────────┤
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  constexpr auto evaluate(const pos_t& x, Real t) const -> tensor_t final {
    return m_op(m_v0(x, t), m_v1(x, t));
  }
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  constexpr auto in_domain(const pos_t& x, Real t) const -> bool final {
    return m_v0.in_domain(x, t) && m_v1.in_domain(x, t);
  }
};
//╘══════════════════════════════════════════════════════════════════════════╛
//╒══════════════════════════════════════════════════════════════════════════╕
//│ binary field operations                                                  │
//╞══════════════════════════════════════════════════════════════════════════╡
template <typename RealOut, size_t NOut, size_t... TensorDimsOut, typename V0,
          typename Real0, size_t   N0, size_t... TensorDims0, typename V1,
          typename Real1, size_t   N1, size_t... TensorDims1, typename Op>
constexpr auto make_binary_operation_field(
    const field<V0, Real0, N0, TensorDims0...>& lhs,
    const field<V1, Real1, N1, TensorDims1...>& rhs, const Op& op) {
  return binary_operation_field<field<V0, Real0, N0, TensorDims0...>,
                                const field<V1, Real1, N1, TensorDims1...>&,
                                const Op&, RealOut, NOut, TensorDimsOut...>{
      lhs, rhs, op};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut,
          typename V0, typename Real0, size_t N0, size_t... TensorDims0,
          typename V1, typename Real1, size_t N1, size_t... TensorDims1,
          typename Op>
constexpr auto make_binary_operation_field(
    field<V0, Real0, N0, TensorDims0...>&& lhs,
    const field<V1, Real1, N1, TensorDims1...>& rhs,
    const Op& op) {
  return binary_operation_field<field<V0, Real0, N0, TensorDims0...>,
                                const field<V1, Real1, N1, TensorDims1...>&,
                                const Op&, RealOut, NOut, TensorDimsOut...>{
      std::move(lhs), rhs, op};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut,
          typename V0, typename Real0, size_t N0, size_t... TensorDims0,
          typename V1, typename Real1, size_t N1, size_t... TensorDims1,
          typename Op>
constexpr auto make_binary_operation_field(
    const field<V0, Real0, N0, TensorDims0...>& lhs,
    field<V1, Real1, N1, TensorDims1...>&& rhs,
    const Op& op) {
  return binary_operation_field<const field<V0, Real0, N0, TensorDims0...>&,
                                field<V1, Real1, N1, TensorDims1...>, const Op&,
                                RealOut, NOut, TensorDimsOut...>{
      lhs, std::move(rhs), op};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut,
          typename V0, typename Real0, size_t N0, size_t... TensorDims0,
          typename V1, typename Real1, size_t N1, size_t... TensorDims1,
          typename Op>
constexpr auto make_binary_operation_field(
    field<V0, Real0, N0, TensorDims0...>&& lhs,
    field<V1, Real1, N1, TensorDims1...>&& rhs,
    const Op& op) {
  return binary_operation_field<field<V0, Real0, N0, TensorDims0...>,
                                field<V1, Real1, N1, TensorDims1...>, const Op&,
                                RealOut, NOut, TensorDimsOut...>{
      std::move(lhs), std::move(rhs), op};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut,
          typename V0, typename Real0, size_t N0, size_t... TensorDims0,
          typename V1, typename Real1, size_t N1, size_t... TensorDims1,
          typename Op>
constexpr auto make_binary_operation_field(
    const field<V0, Real0, N0, TensorDims0...>& lhs,
    const field<V1, Real1, N1, TensorDims1...>& rhs,
    Op&& op) {
  return binary_operation_field<const field<V0, Real0, N0, TensorDims0...>&,
                                const field<V1, Real1, N1, TensorDims1...>&, Op,
                                RealOut, NOut, TensorDimsOut...>{lhs, rhs,
                                                                 std::move(op)};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut,
          typename V0, typename Real0, size_t N0, size_t... TensorDims0,
          typename V1, typename Real1, size_t N1, size_t... TensorDims1,
          typename Op>
constexpr auto make_binary_operation_field(
    field<V0, Real0, N0, TensorDims0...>&& lhs,
    const field<V1, Real1, N1, TensorDims1...>& rhs,
    Op&& op) {
  return binary_operation_field<field<V0, Real0, N0, TensorDims0...>,
                                const field<V1, Real1, N1, TensorDims1...>&, Op,
                                RealOut, NOut, TensorDimsOut...>{
      std::move(lhs), rhs, std::move(op)};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut,
          typename V0, typename Real0, size_t N0, size_t... TensorDims0,
          typename V1, typename Real1, size_t N1, size_t... TensorDims1,
          typename Op>
constexpr auto make_binary_operation_field(
    const field<V0, Real0, N0, TensorDims0...>& lhs,
    field<V1, Real1, N1, TensorDims1...>&& rhs,
    Op op) {
  return binary_operation_field<const field<V0, Real0, N0, TensorDims0...>&,
                                field<V1, Real1, N1, TensorDims1...>, Op,
                                RealOut, NOut, TensorDimsOut...>{
      lhs, std::move(rhs), std::move(op)};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut,
          typename V0, typename Real0, size_t N0, size_t... TensorDims0,
          typename V1, typename Real1, size_t N1, size_t... TensorDims1,
          typename Op>
constexpr auto make_binary_operation_field(
    field<V0, Real0, N0, TensorDims0...>&& lhs,
    field<V1, Real1, N1, TensorDims1...>&& rhs,
    Op op) {
  return binary_operation_field<field<V0, Real0, N0, TensorDims0...>,
                                field<V1, Real1, N1, TensorDims1...>, Op,
                                RealOut, NOut, TensorDimsOut...>{
      std::move(lhs), std::move(rhs), std::move(op)};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
