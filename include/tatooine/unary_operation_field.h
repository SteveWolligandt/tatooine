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
  using parent_t = vectorfield<this_t, Real, N, TensorDims...>;
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
  constexpr tensor_t evaluate(pos_t const& x, Real const t) const override {
    if constexpr (holds_field_pointer) {
      return m_unary_op(m_field->evaluate(m_pos_op(x), m_time_op(t)));
    } else {
      return m_unary_op(m_field.evaluate(m_pos_op(x), m_time_op(t)));
    }
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real /*t*/) const override {
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
  void set_field(vectorfield<V, Real, N> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F                                   = InternalField,
            std::enable_if_t<std::is_pointer_v<_F>, bool> = true,
            std::enable_if_t<std::is_same_v<_F, InternalField>, bool> = true>
  void set_field(parent::vectorfield<Real, N> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F                                   = InternalField,
            std::enable_if_t<std::is_pointer_v<_F>, bool> = true,
            std::enable_if_t<std::is_same_v<_F, InternalField>, bool> = true>
  void set_field(parent::vectorfield<Real, N> const* v) {
    m_field = v;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename PosOp, typename TimeOp, typename TensorOp>
unary_operation_field(field<Field, Real, N, TensorDims...> const&,
                      PosOp&& pos_op, TimeOp&& time_op, TensorOp&& tensor_op)
    -> unary_operation_field<Field, PosOp, TimeOp, TensorOp, RealOut, NOut,
                             TensorDimsOut...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Real, size_t N, size_t... TensorDims,
          typename PosOp, typename TimeOp, typename TensorOp>
unary_operation_field(parent::field<Real, N, TensorDims...> const*)
    -> unary_operation_field<parent::field<Real, N, TensorDims...> const*,
                             PosOp, TimeOp, TensorOp, RealOut, NOut,
                             TensorDimsOut...>;
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
