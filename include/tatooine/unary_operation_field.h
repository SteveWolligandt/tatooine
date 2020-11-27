#ifndef TATOOINE_UNARY_OPERATION_FIELD_H
#define TATOOINE_UNARY_OPERATION_FIELD_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, typename Op, typename Real, size_t N,
          size_t... TensorDims>
struct unary_operation_field
    : field<unary_operation_field<V, Op, Real, N, TensorDims...>, Real, N,
            TensorDims...> {
 public:
  using this_t   = unary_operation_field<V, Op, Real, N, TensorDims...>;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

 private:
  V  m_v;
  Op m_op;

 public:
  constexpr unary_operation_field(unary_operation_field const&)     = default;
  constexpr unary_operation_field(unary_operation_field&&) noexcept = default;
  template <typename V_, typename Op_>
  constexpr unary_operation_field(V_&& v, Op_&& op)
      : m_v{std::forward<V_>(v)}, m_op{std::forward<Op_>(op)} {}

 public:
  constexpr auto operator=(unary_operation_field const&)
      -> unary_operation_field& = default;
  constexpr auto operator=(unary_operation_field&&) noexcept
      -> unary_operation_field& = default;

 public:
  ~unary_operation_field() override = default;
  //============================================================================
  constexpr auto evaluate(pos_t const& x, Real t) const -> tensor_t final {
    return m_op(m_v(x, t));
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& x, Real t) const -> bool final {
    return m_v.in_domain(x, t);
  }
};
//==============================================================================
template <typename RealOut, size_t NOut, size_t... TensorDimsOut, typename V,
          typename Real, size_t    N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(
    field<V, Real, N, TensorDims...> const& v, Op const& op) {
  return unary_operation_field<V const&, Op const&, RealOut, NOut,
                               TensorDimsOut...>{v.as_derived(), op};
}
//------------------------------------------------------------------------------
template <typename RealOut, size_t NOut, size_t... TensorDimsOut, typename V,
          typename Real, size_t    N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(
    field<V, Real, N, TensorDims...>&& v, Op const& op) {
  return unary_operation_field<V, Op const&,
                               RealOut, NOut, TensorDimsOut...>{std::move(v.as_derived()),
                                                                op};
}
//------------------------------------------------------------------------------
template <typename RealOut, size_t NOut, size_t... TensorDimsOut, typename V,
          typename Real, size_t    N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(
    field<V, Real, N, TensorDims...> const& v, Op&& op) {
  return unary_operation_field<V const&, std::decay_t<Op>, RealOut, NOut,
                               TensorDimsOut...>{v.as_derived(), std::move(op)};
}
//------------------------------------------------------------------------------
template <typename RealOut, size_t NOut, size_t... TensorDimsOut, typename V,
          typename Real, size_t    N, size_t... TensorDims, typename Op>
constexpr auto make_unary_operation_field(
    field<V, Real, N, TensorDims...>&& v, Op&& op) {
  return unary_operation_field<V,
                               std::decay_t<Op>, RealOut, NOut,
                               TensorDimsOut...>{std::move(v.as_derived()), std::move(op)};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
