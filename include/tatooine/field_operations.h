#ifndef TATOOINE_FIELD_OPERATIONS_H
#define TATOOINE_FIELD_OPERATIONS_H
#include "field.h"
//╔════════════════════════════════════════════════════════════════════════════╗
namespace tatooine {
//╒══════════════════════════════════════════════════════════════════════════╕
//│ unary_operation_field                                                    │
template <typename V, typename Op, typename Real, size_t N,
          size_t... TensorDims>
struct unary_operation_field
    : field<unary_operation_field<V, Op, Real, N, TensorDims...>, Real, N,
            TensorDims...> {
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ typedefs                                                             │
  //├──────────────────────────────────────────────────────────────────────┤
  using this_t   = unary_operation_field<V, Op, Real, N, TensorDims...>;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ members                                                              │
  //├──────────────────────────────────────────────────────────────────────┤
 private:
  V  m_v;
  Op m_op;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ ctors                                                                │
  //├──────────────────────────────────────────────────────────────────────┤
 public:
  constexpr unary_operation_field(const unary_operation_field&) = default;
  constexpr unary_operation_field(unary_operation_field&&) = default;
  template <typename V_, typename Op_>
  constexpr unary_operation_field(V_&& v, Op_&& op)
      : m_v{std::forward<V_>(v)}, m_op{std::forward<Op_>(op)} {}
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ assignment operators                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
 public:
  constexpr auto operator=(const unary_operation_field&)
    -> unary_operation_field& = default;
  constexpr auto operator=(unary_operation_field &&)
    ->unary_operation_field& = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ dtor                                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
 public:
  ~unary_operation_field() final = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ methods                                                              │
  //├──────────────────────────────────────────────────────────────────────┤
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  auto evaluate(const pos_t& x, Real t) const -> tensor_t {
    return m_op(m_v(x, t));
  }
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  auto in_domain(const pos_t& x, Real t) const -> bool {
    return m_v.in_domain(x, t);
  }
};
//╞══════════════════════════════════════════════════════════════════════════╡
//│ unary field operations                                                   │
//╞══════════════════════════════════════════════════════════════════════════╡
template <typename RealOut, size_t NOut, size_t... TensorDimsOut, typename V,
          typename Real, size_t    N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(
    const field<V, Real, N, TensorDims...>& f, const Op& op) {
  return unary_operation_field<const field<V, Real, N, TensorDims...>&,
                               const Op&, RealOut, NOut, TensorDimsOut...>{f,
                                                                           op};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut, typename V,
          typename Real, size_t    N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(field<V, Real, N, TensorDims...>&& f,
                                          const Op& op) {
  return unary_operation_field<field<V, Real, N, TensorDims...>, const Op&,
                               RealOut, NOut, TensorDimsOut...>{std::move(f),
                                                                op};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut, typename V,
          typename Real, size_t    N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(
    const field<V, Real, N, TensorDims...>& f, Op&& op) {
  return unary_operation_field<const field<V, Real, N, TensorDims...>&, Op&&,
                               RealOut, NOut, TensorDimsOut...>{f,
                                                                std::move(op)};
}
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename RealOut, size_t NOut, size_t... TensorDimsOut, typename V,
          typename Real, size_t    N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(field<V, Real, N, TensorDims...>&& f,
                                          Op&& op) {
  return unary_operation_field<field<V, Real, N, TensorDims...>, Op, RealOut,
                               NOut, TensorDimsOut...>{std::move(f),
                                                       std::move(op)};
}
//╘══════════════════════════════════════════════════════════════════════════╛
//╒══════════════════════════════════════════════════════════════════════════╕
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
  constexpr binary_operation_field(binary_operation_field&&) = default;
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
  constexpr auto operator=(binary_operation_field &&)
    ->binary_operation_field& = default;
 public:
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ dtor                                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  ~binary_operation_field() final = default;
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
//╒══════════════════════════════════════════════════════════════════════════╕
//│ operations                                                               │
//╞══════════════════════════════════════════════════════════════════════════╡
template <typename V0, typename Real0,
          typename V1, typename Real1,
          size_t N, size_t TN>
constexpr auto dot(const field<V0, Real0, N, TN>& lhs,
                   const field<V1, Real1, N, TN>& rhs) {
  using RealOut = promote_t<Real0, Real1>;
  return make_binary_operation_field<RealOut, N>(
      lhs, rhs,
      [](const typename V0::tensor_t& lhs, const typename V1::tensor_t& rhs) {
        return dot(lhs, rhs);
      });
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename V, typename Real, size_t N, size_t... TensorDims>
constexpr auto operator-(const field<V, Real, N, TensorDims...>& f) {
  return make_unary_operation_field<Real, N, TensorDims...>(
      f, [](const auto& v) { return -v; });
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename V, typename Real, size_t N, size_t VecDim>
constexpr auto normalize(const field<V, Real, N, VecDim>& f) {
  return make_unary_operation_field<Real, N, VecDim>(
      f, [](const auto& v) { return normalize(v); });
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename V, typename Real, size_t N, size_t VecDim>
constexpr auto length(const field<V, Real, N, VecDim>& f) {
  return make_unary_operation_field<Real, N>(
      f, [](const auto& v) { return length(v); });
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename V0, typename Real0,
          typename V1, typename Real1,
          size_t N, size_t... TensorDims>
constexpr auto operator+(
    const field<V0, Real0, N, TensorDims...>& lhs,
    const field<V1, Real1, N, TensorDims...>& rhs) {
  return make_binary_operation_field<promote_t<Real0, Real1>, N,
                                     TensorDims...>(
      lhs, rhs, [](const auto& lhs, const auto& rhs) { return lhs + rhs; });
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename V0, typename Real0, typename V1,
          typename Real1, size_t N, size_t TM, size_t TN>
constexpr auto operator*(const field<V0, Real0, N, TM, TN>& lhs,
                         const field<V1, Real1, N, TN>&     rhs) {
  return make_binary_operation_field<promote_t<Real0, Real1>, N, TM>(
      lhs, rhs, [](const auto& lhs, const auto& rhs) { return lhs * rhs; });
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename V0, typename Real0,
          typename V1, typename Real1,
          size_t N, size_t... TensorDims>
constexpr auto operator*(const field<V0, Real0, N, TensorDims...>& lhs,
                         const field<V1, Real1, N, TensorDims...>& rhs) {
  return make_binary_operation_field<promote_t<Real0, Real1>, N, TensorDims...>(
      lhs, rhs, [](const auto& lhs, const auto& rhs) { return lhs * rhs; });
}
//├──────────────────────────────────────────────────────────────────────────┤
// template <typename tensor_t, typename tensor_Real, typename scalar_Real,
//          size_t... Dims,
//          std::enable_if_t<std::is_arithmetic_v<scalar_Real> ||
//                           is_complex_v<scalar_Real> ||
//                           std::is_same_v<scalar_Real, GiNaC::ex>>...>
// constexpr auto operator*(const base_tensor<tensor_t, tensor_Real, Dims...>&
// t,
//                         const scalar_Real scalar) {
//  return unary_operation(
//      [scalar](const auto& component) { return component * scalar; }, t);
//}
//├──────────────────────────────────────────────────────────────────────────┤
// template <typename tensor_t, typename tensor_Real, typename scalar_Real,
//          size_t... Dims,
//          std::enable_if_t<std::is_arithmetic_v<scalar_Real> ||
//                           is_complex_v<scalar_Real> ||
//                           std::is_same_v<scalar_Real, GiNaC::ex>>...>
// constexpr auto operator*(
//    const scalar_Real                                  scalar,
//    const base_tensor<tensor_t, tensor_Real, Dims...>& t) {
//  return unary_operation(
//      [scalar](const auto& component) { return component * scalar; }, t);
//}
//
//├──────────────────────────────────────────────────────────────────────────┤
// template <typename tensor_t, typename tensor_Real, typename scalar_Real,
//          size_t... Dims,
//          std::enable_if_t<std::is_arithmetic_v<scalar_Real> ||
//                           is_complex_v<scalar_Real> ||
//                           std::is_same_v<scalar_Real, GiNaC::ex>>...>
// constexpr auto operator/(const base_tensor<tensor_t, tensor_Real, Dims...>&
// t,
//                         const scalar_Real scalar) {
//  return unary_operation(
//      [scalar](const auto& component) { return component / scalar; }, t);
//}
//├──────────────────────────────────────────────────────────────────────────┤
// template <typename tensor_t, typename tensor_Real, typename scalar_Real,
//          size_t... Dims,
//          std::enable_if_t<std::is_arithmetic_v<scalar_Real> ||
//                           is_complex_v<scalar_Real> ||
//                           std::is_same_v<scalar_Real, GiNaC::ex>>...>
// constexpr auto operator/(
//    const scalar_Real                                  scalar,
//    const base_tensor<tensor_t, tensor_Real, Dims...>& t) {
//  return unary_operation(
//      [scalar](const auto& component) { return scalar / component; }, t);
//}
//
//├──────────────────────────────────────────────────────────────────────────┤
// template <typename lhs_tensor_t, typename Real0,
//          typename rhs_tensor_t, typename Real1,
//          size_t... Dims>
// constexpr auto operator-(
//    const base_tensor<lhs_tensor_t, Real0, Dims...>& lhs,
//    const base_tensor<rhs_tensor_t, Real1, Dims...>& rhs) {
//  return binary_operation(std::minus<promote_t<Real0, Real1>>{}, lhs,
//                          rhs);
//}
//
//├──────────────────────────────────────────────────────────────────────────┤
///// matrix-vector-multiplication
// template <typename lhs_tensor_t, typename Real0,
//          typename rhs_tensor_t, typename Real1, size_t M, size_t N>
// constexpr auto operator*(const base_tensor<lhs_tensor_t, Real0, M, N>& lhs,
//                         const base_tensor<rhs_tensor_t, Real1, N>& rhs) {
//  tensor<promote_t<Real0, Real1>, M> product;
//  for (size_t i = 0; i < M; ++i) {
//    product(i) = dot(lhs.template slice<0>(i), rhs);
//  }
//  return product;
//}
}
//╚════════════════════════════════════════════════════════════════════════════╝
#endif
