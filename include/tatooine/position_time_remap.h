#ifndef TATOOINE_UNARY_OPERATION_FIELD_H
#define TATOOINE_UNARY_OPERATION_FIELD_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename InternalField,
          typename PosOp, typename TimeOp, typename TensorOp,
          typename Real, size_t N, size_t... TensorDims>
struct unary_operation_field
    : field<unary_operation_field<InternalField,
                                  PosOp, TimeOp, TensorOp,
                                  Real, N, TensorDims...>,
            Real, N, TensorDims...> {
  using this_t   = unary_operation_field<InternalField,
                                         PosOp, TimeOp, TensorOp,
                                         Real, N, TensorDims...>;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::real_t;
  using typename parent_t::pos_t;
  using typename parent_t::time_t;
  using typename parent_t::tensor_t;
  static constexpr auto holds_field_pointer = std::is_pointer_v<InternalField>;

  static_assert(
      std::is_convertible_v<std::invoke_result_t<PosOp, pos_t, time_t>,
                            typename InternalField::pos_t>);
  static_assert(
      std::is_convertible_v<std::invoke_result_t<TimeOp, pos_t, time_t>,
                            typename InternalField::time_t>);
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
  //----------------------------------------------------------------------------
  auto operator=(unary_operation_field const& other)
      -> unary_operation_field& = default;
  auto operator=(unary_operation_field&& other) noexcept
      -> unary_operation_field& = default;
  //----------------------------------------------------------------------------
  virtual ~unary_operation_field() = default;
  //----------------------------------------------------------------------------
  template <typename _F = InternalField, typename _PosOp, typename _TimeOp,
            typename _TensorOp>
      requires(!std::is_pointer_v<_F>) &&
      std::is_same_v<_F, InternalField>
  unary_operation_field(
    field<InternalField, Real, N, TensorDims...> const& f,
    _PosOp&& pos_op, _TimeOp&& time_op, _TensorOp&& tensor_op)
      : m_field{f.as_derived()},
        m_pos_op{std::forward<_PosOp>(pos_op)},
        m_time_op{std::forward<_TimeOp>(time_op)},
        m_tensor_op{std::forward<_TensorOp>(tensor_op)} {}
  //----------------------------------------------------------------------------
  template <typename _PosOp, typename _TimeOp, typename _TensorOp>
  unary_operation_field(parent::field<Real, N> const* f, _PosOp&& pos_op,
                        _TimeOp&& time_op, _TensorOp&& tensor_op)
      : m_field{f},
        m_pos_op{std::forward<_PosOp>(pos_op)},
        m_time_op{std::forward<_TimeOp>(time_op)},
        m_tensor_op{std::forward<_TensorOp>(tensor_op)} {}
  //----------------------------------------------------------------------------
  template <typename _F = InternalField>
    requires std::is_pointer_v<_F> &&
             std::is_same_v<_F, InternalField>
  unary_operation_field() : m_field{nullptr} {}
  //============================================================================
  constexpr auto evaluate(pos_t const& x, Real const t) const
      -> tensor_t final {
    if constexpr (holds_field_pointer) {
      return m_tensor_op(m_field->evaluate(m_pos_op(x, t), m_time_op(x, t)));
    } else {
      return m_tensor_op(m_field.evaluate(m_pos_op(x, t), m_time_op(x, t)));
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(const pos_t& x, Real t) const -> bool final {
    if constexpr (holds_field_pointer) {
      return m_field->in_domain(m_pos_op(x, t), m_time_op(x, t));
    } else {
      return m_field.in_domain(m_pos_op(x, t), m_time_op(x, t));
    }
  }
  //----------------------------------------------------------------------------
  template <typename V, typename _F = InternalField>
    requires std::is_pointer_v<_F> &&
             std::is_same_v<_F, InternalField>
  void set_field(field<V, Real, N, TensorDims...> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F = InternalField>
    requires std::is_pointer_v<_F> &&
             std::is_same_v<_F, InternalField>
  void set_field(parent::field<Real, N, TensorDims...> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F = InternalField>
    requires std::is_pointer_v<_F> &&
             std::is_same_v<_F, InternalField>
  void set_field(parent::field<Real, N, TensorDims...> const* v) {
    m_field = v;
  }
};
//==============================================================================
namespace detail {
//==============================================================================
template <typename Field, typename PosOp, typename TimeOp, typename TensorOp,
          typename Pos, real_number PosReal, size_t PosN, real_number TimeReal,
          typename Tensor, real_number TensorReal, size_t... TensorDims>
constexpr auto unary_operation_field_constructor_impl(
    Field&& field, PosOp&& pos_op, TimeOp&& time_op, TensorOp&& tensor_op,
    base_tensor<Pos, PosReal, PosN> const& /*x*/, TimeReal const /*t*/,
    base_tensor<Tensor, TensorReal, TensorDims...> const& /*tensor*/) {
  using real_t = common_type<TimeReal, PosReal, TensorReal>;
  return unary_operation_field<Field, std::decay_t<PosOp>, std::decay_t<TimeOp>,
                               std::decay_t<TensorOp>, real_t, PosN,
                               TensorDims...>{
      std::forward<Field>(field), std::forward<PosOp>(pos_op),
      std::forward<TimeOp>(time_op), std::forward<TensorOp>(tensor_op)};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename PosOp, typename TimeOp, typename TensorOp,
          typename Pos, real_number PosReal, size_t PosN, real_number TimeReal,
          real_number Tensor>
constexpr auto unary_operation_field_constructor_impl(
    Field&& field, PosOp&& pos_op, TimeOp&& time_op, TensorOp&& tensor_op,
    base_tensor<Pos, PosReal, PosN> const& /*x*/, TimeReal const /*t*/,
    Tensor const& /*tensor*/) {
  using real_t = common_type<TimeReal, PosReal, Tensor>;
  return unary_operation_field<Field, std::decay_t<PosOp>, std::decay_t<TimeOp>,
                               std::decay_t<TensorOp>, real_t, PosN>{
      std::forward<Field>(field), std::forward<PosOp>(pos_op),
      std::forward<TimeOp>(time_op), std::forward<TensorOp>(tensor_op)};
}
//------------------------------------------------------------------------------
template <typename Field, typename PosOp, typename TimeOp, typename TensorOp>
struct unary_operation_field_constructor {
  using pos_t = std::invoke_result_t<PosOp, typename Field::pos_t,
                                     typename Field::time_t>;
  using time_t = std::invoke_result_t<TimeOp, typename Field::pos_t,
                                     typename Field::time_t>;
  using tensor_t = std::invoke_result_t<TensorOp, typename Field::tensor_t>;
  using type     = decltype(unary_operation_field_constructor_impl(
      std::declval<Field>(),
      std::declval<PosOp>(), std::declval<TimeOp>(), std::declval<TensorOp>(),
      std::declval<pos_t>(), std::declval<time_t>(), std::declval<tensor_t>()));
};
//==============================================================================
}  // namespace detail
//==============================================================================
template <size_t OutN, typename Field, typename Real, size_t N, size_t... TensorDims,
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
