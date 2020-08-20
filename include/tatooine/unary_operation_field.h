#ifndef TATOOINE_UNARY_OPERATION_FIELD_H
#define TATOOINE_UNARY_OPERATION_FIELD_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename InternalField, typename PosOp, typename TimeOp,
          typename TensorOp, typename Real, size_t N, size_t... TensorDims>
struct unary_operation_field
    : field<unary_operation_field<InternalField, PosOp, TimeOp, TensorOp, Real,
                                  N, TensorDims...>,
            Real, N> {
  using this_t   = unary_operation_field<InternalField, PosOp, TimeOp, TensorOp,
                                       Real, N, TensorDims...>;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  static constexpr auto holds_field_pointer = std::is_pointer_v<InternalField>;
  //============================================================================
 private:
  InternalField m_field;
  PosOp         m_pos_op;
  TimeOp        m_time_op;
  TensorOp      m_tensor_op;

  //============================================================================
 public:
  unary_operation_field(unary_operation_field const& other)     = default;
  unary_operation_field(unary_operation_field&& other) noexcept = default;
  auto operator                 =(unary_operation_field const& other)
      -> unary_operation_field& = default;
  auto operator                 =(unary_operation_field&& other) noexcept
      -> unary_operation_field& = default;

  template <typename _F                                    = InternalField,
            std::enable_if_t<!std::is_pointer_v<_F>, bool> = true,
            std::enable_if_t<std::is_same_v<_F, InternalField>, bool> = true>
  unary_operation_field(field<InternalField, Real, N> const& f)
      : m_field{f.as_derived()} {}
  template <typename _F                                   = InternalField,
            std::enable_if_t<std::is_pointer_v<_F>, bool> = true,
            std::enable_if_t<std::is_same_v<_F, InternalField>, bool> = true>
  unary_operation_field(parent::field<Real, N> const* f) : m_field{f} {}
  template <typename _F                                   = InternalField,
            std::enable_if_t<std::is_pointer_v<_F>, bool> = true,
            std::enable_if_t<std::is_same_v<_F, InternalField>, bool> = true>
  unary_operation_field() : m_field{nullptr} {}

  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& x, Real const t) const
      -> tensor_t override {
    if constexpr (holds_field_pointer) {
      return m_tensor_op(m_field->evaluate(m_pos_op(x), m_time_op(t)));
    } else {
      return m_tensor_op(m_field.evaluate(m_pos_op(x), m_time_op(t)));
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(const pos_t& x, Real t) const -> bool override {
    if constexpr (holds_field_pointer) {
      return m_field->in_domain(m_pos_op(x), m_time_op(t));
    } else {
      return m_field.in_domain(m_pos_op(x), m_time_op(t));
    }
  }
  //----------------------------------------------------------------------------
  template <typename V, typename _F = InternalField,
            std::enable_if_t<std::is_pointer_v<_F>, bool>             = true,
            std::enable_if_t<std::is_same_v<_F, InternalField>, bool> = true>
  void set_field(field<V, Real, N, TensorDims...> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F                                   = InternalField,
            std::enable_if_t<std::is_pointer_v<_F>, bool> = true,
            std::enable_if_t<std::is_same_v<_F, InternalField>, bool> = true>
  void set_field(parent::field<Real, N, TensorDims...> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F                                   = InternalField,
            std::enable_if_t<std::is_pointer_v<_F>, bool> = true,
            std::enable_if_t<std::is_same_v<_F, InternalField>, bool> = true>
  void set_field(parent::field<Real, N, TensorDims...> const* v) {
    m_field = v;
  }
};
//==============================================================================
namespace detail {
//==============================================================================
template <typename Field, typename PosOp, typename TimeOp, typename TensorOp,
          typename PosReal, size_t PosN,
          typename TimeReal,
          typename TenTensor, typename TenReal, size_t TenN,
          size_t... TensorDims>
constexpr auto unary_operation_field_constructor_impl(
    Field&& field, PosOp&& pos_op, TimeOp&& time_op, TensorOp&& tensor_op,
    vec<PosReal, PosN> const&,
    TimeReal const,
    base_tensor<TenTensor, TenReal, TenN, TensorDims...> const&) {
  static_assert(std::is_same_v<PosReal, TimeReal>);
  static_assert(std::is_same_v<PosReal, TenReal>);
  return unary_operation_field<Field, PosOp, TimeOp, TensorOp, PosReal, PosN,
                               TensorDims...>{
         std::forward<Field>(field), std::forward<PosOp>(pos_op),
          std::forward<TimeOp>(time_op), std::forward<TensorOp>(tensor_op)};
}
template <typename Field, typename PosOp, typename TimeOp, typename TensorOp>
struct unary_operation_field_constructor {
  using type = decltype(unary_operation_field_constructor_impl(
      std::declval<Field>(), std::declval<PosOp>(), std::declval<TimeOp>(),
      std::declval<TensorOp>(),
      std::declval<std::invoke_result_t<PosOp, typename Field::pos_t>>(),
      std::declval<std::invoke_result_t<TimeOp, typename Field::real_t>>(),
      std::declval<
          std::invoke_result_t<TensorOp, typename Field::tensor_t>>()));
};
//==============================================================================
}  // namespace detail
//==============================================================================
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename PosOp, typename TimeOp, typename TensorOp>
constexpr auto make_unary_operation_field(
    field<Field, Real, N, TensorDims...> const& f, PosOp&& pos_op,
    TimeOp&& time_op, TensorOp&& tensor_op) {
  return
      typename detail::unary_operation_field_constructor<Field, PosOp, TimeOp,
                                                         TensorOp>::type{
          f.as_derived(), std::forward<PosOp>(pos_op),
          std::forward<TimeOp>(time_op), std::forward<TensorOp>(tensor_op)};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
