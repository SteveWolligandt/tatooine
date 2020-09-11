#ifndef TATOOINE_UNARY_OPERATION_FIELD_H
#define TATOOINE_UNARY_OPERATION_FIELD_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename InternalField, typename Operation, size_t... TensorDims>
struct unary_operation_field
    : field<
          unary_operation_field<InternalField, Operation, TensorDims...>,
          typename std::decay_t<std::remove_pointer_t<InternalField>>::real_t,
          std::decay_t<std::remove_pointer_t<InternalField>>::num_dimensions(),
          TensorDims...> {
  using this_t = unary_operation_field<InternalField, Operation, TensorDims...>;
  using parent_t = field<
      this_t,
      typename std::decay_t<std::remove_pointer_t<InternalField>>::real_t,
      std::decay_t<std::remove_pointer_t<InternalField>>::num_dimensions(),
      TensorDims...>;
  using parent_t::num_dimensions;
  using internal_field_t = InternalField;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  static constexpr auto holds_field_pointer = std::is_pointer_v<InternalField>;

  //============================================================================
 private:
  InternalField m_field;
  Operation     m_operation;
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
  template <std::convertible_to<InternalField> _F, typename _Operation>
  requires std::is_same_v<std::decay_t<_F>, std::decay_t<InternalField>>
  unary_operation_field(_F&& f, _Operation&& operation)
      : m_field{std::forward<_F>(f)},
        m_operation{std::forward<_Operation>(operation)} {}
  //----------------------------------------------------------------------------
  template <typename _Operation, real_number Real, size_t N, typename _F = InternalField>
  requires std::is_pointer_v<InternalField> &&
           std::is_same_v<std::decay_t<_F>, std::decay_t<InternalField>>
  unary_operation_field(parent::field<Real, N> const* f, _Operation&& operation)
      : m_field{f}, m_operation{std::forward<_Operation>(operation)} {}
  //----------------------------------------------------------------------------
  template <typename _F = InternalField>
  requires std::is_pointer_v<_F> && std::is_same_v<_F, InternalField>
  unary_operation_field() : m_field{nullptr} {}
  //============================================================================
  constexpr auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    if constexpr (holds_field_pointer) {
      return m_operation(m_field->evaluate(x, t));
    } else {
      return m_operation(m_field.evaluate(x, t));
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(const pos_t& x, real_t t) const -> bool final {
    if constexpr (holds_field_pointer) {
      return m_field->in_domain(x, t);
    } else {
      return m_field.in_domain(x, t);
    }
  }
  //----------------------------------------------------------------------------
  template <typename V, typename _F = InternalField>
  requires std::is_pointer_v<_F>&& std::is_same_v<_F, InternalField> void
                                   set_field(field<V, real_t, num_dimensions(),
                  TensorDims...> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F = InternalField>
  requires std::is_pointer_v<_F> && std::is_same_v<_F, InternalField>
  void set_field(parent::field<real_t, num_dimensions(),
                               TensorDims...> const& v) {
    m_field = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _F = InternalField, real_number Real, size_t N>
  requires std::is_pointer_v<_F> && std::is_same_v<_F, InternalField>
  void set_field(parent::field<Real, N, TensorDims...> const* v) {
    m_field = v;
  }
  //----------------------------------------------------------------------------
  auto internal_field() const ->auto const& {
    return m_field;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto internal_field() -> auto& {
    return m_field;
  }
};
//==============================================================================
namespace detail {
//==============================================================================
template <typename Field, typename Operation, typename Tensor,
          real_number TensorReal, size_t... TensorDims>
constexpr auto unary_operation_field_constructor_impl(
    Field&& field, Operation&& operation,
    base_tensor<Tensor, TensorReal, TensorDims...> const& /*tensor*/) {
  return unary_operation_field<Field, std::decay_t<Operation>, TensorDims...>{
      std::forward<Field>(field), std::forward<Operation>(operation)};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, typename Operation, real_number Tensor>
constexpr auto unary_operation_field_constructor_impl(
    Field&& field, Operation&& operation, Tensor const& /*tensor*/) {
  return unary_operation_field<Field, std::decay_t<Operation>>{
      std::forward<Field>(field), std::forward<Operation>(operation)};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, typename Operation, real_number Tensor,
          size_t... TensorDims>
constexpr auto unary_operation_field_constructor_impl(
    parent::field<Real, N, TensorDims...> const* field, Operation&& operation,
    Tensor const& /*tensor*/) {
  return unary_operation_field<parent::field<Real, N, TensorDims...>,
                               std::decay_t<Operation>>{
      field, std::forward<Operation>(operation)};
}
//------------------------------------------------------------------------------
template <typename Field, typename Operation>
struct unary_operation_field_constructor {
  using tensor_t = std::invoke_result_t<
      Operation,
      typename std::decay_t<std::remove_pointer_t<Field>>::tensor_t>;
  using type     = decltype(unary_operation_field_constructor_impl(
      std::declval<Field>(),
      std::declval<Operation>(),
      std::declval<tensor_t>()));
};
//==============================================================================
}  // namespace detail
//==============================================================================
template <typename Field, real_number Real, size_t N,
          size_t... TensorDims, typename Operation>
constexpr auto make_unary_operation_field(
    field<Field, Real, N, TensorDims...> const& f, Operation&& operation) {
  return typename detail::unary_operation_field_constructor<Field const&,
                                                            Operation>::type{
      f.as_derived(), std::forward<Operation>(operation)};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, real_number Real, size_t N, size_t... TensorDims,
          typename Operation>
constexpr auto make_unary_operation_field(
    field<Field, Real, N, TensorDims...>&& f, Operation&& operation) {
  return typename detail::unary_operation_field_constructor<std::decay_t<Field>,
                                                            Operation>::type{
      std::move(f.as_derived()), std::forward<Operation>(operation)};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t... TensorDims, typename Operation>
constexpr auto make_unary_operation_field(
    parent::field<Real, N, TensorDims...> const* f, Operation&& operation) {
  return typename detail::unary_operation_field_constructor<
      parent::field<Real, N, TensorDims...> const*, Operation>::type{
      f, std::forward<Operation>(operation)};
}
//------------------------------------------------------------------------------
template <typename Field, real_number Real, size_t N,
          size_t... TensorDims, typename Operation>
constexpr auto operator|(
    field<Field, Real, N, TensorDims...> const& f, Operation&& operation) {
  return make_unary_operation_field(f, std::forward<Operation>(operation));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Field, real_number Real, size_t N, size_t... TensorDims,
          typename Operation>
constexpr auto operator|(field<Field, Real, N, TensorDims...>&& f,
                         Operation&&                            operation) {
  return make_unary_operation_field(std::move(f), std::forward<Operation>(operation));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t... TensorDims, typename Operation>
constexpr auto operator|(parent::field<Real, N, TensorDims...> const* f,
                         Operation&& operation) {
  return make_unary_operation_field(f, std::forward<Operation>(operation));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
