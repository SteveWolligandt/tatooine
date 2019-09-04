#ifndef TATOOINE_FIELD_H
#define TATOOINE_FIELD_H

#include "crtp.h"
#include "tensor.h"
#include "type_traits.h"

//==============================================================================
namespace tatooine {
//==============================================================================

struct out_of_domain : std::runtime_error {
  out_of_domain() : std::runtime_error{""} {}
};

template <typename Derived, typename Real, size_t N, size_t... TensorDims>
struct field : crtp<Derived> {
  using real_t   = Real;
  using this_t   = field<Derived, Real, N, TensorDims...>;
  using parent_t = crtp<Derived>;
  using pos_t    = tensor<Real, N>;
  using tensor_t = std::conditional_t<sizeof...(TensorDims) == 0, Real,
                                      tensor<Real, TensorDims...>>;
  static constexpr auto num_dimensions() { return N; }
  static constexpr auto num_tensor_dimensions() {
    return sizeof...(TensorDims);
  }
  template <size_t _num_tensor_dims = sizeof...(TensorDims),
            std::enable_if_t<(_num_tensor_dims > 0)>...>
  static constexpr auto tensor_dimension(size_t i) {
    return tensor_t::dimension(i);
  }
  static constexpr auto has_in_domain() { return has_in_domain_v<Derived>; }
  using parent_t::as_derived;

  //============================================================================
  constexpr decltype(auto) operator()(const pos_t& x, Real t = 0) const {
    return evaluate(x, t);
  }

  //----------------------------------------------------------------------------
  constexpr decltype(auto) evaluate(const pos_t& x, Real t = 0) const {
    // if (!in_domain(x, t)) { throw out_of_domain{}; }
    return as_derived().evaluate(x, t);
  }

  //----------------------------------------------------------------------------
  constexpr decltype(auto) in_domain([[maybe_unused]] const pos_t& x,
                                     [[maybe_unused]] Real t = 0) const {
    if constexpr (has_in_domain()) {
      return as_derived().in_domain(x, t);
    } else {
      return true;
    }
  }
};

//==============================================================================
// template <typename Op, typename field_t, typename Real, size_t N, size_t...
// TensorDims> struct unary_operation_field :
// field<unary_operation_field<field_t, Real, N, TensorDims...>> {
//  public:
//   using this_t   = unary_operation_field<field_t, Real, N, TensorDims...>;
//   using parent_t = field<this_t, Real, N, TensorDims...>;
//   using typename parent_t::pos_t;
//   using typename parent_t::tensor_t;
//
//  private:
//   field_t m_field;
//   Op      m_operator;
//
//  public:
//   template <typename _Real, size_t _N, size_t... _TensorDims>
//   unary_operation_field(field<field_t, _Real, _N, _TensorDims...>& _field,
//                         Op&& op)
//       : m_field{_field}, m_operator{op} {}
//
//   tensor_t evaluate(const pos_t& x, Real t) {
//     return m_operator(m_field(x, t));
//   }
//};

//==============================================================================
template <typename LhsField, typename RhsField, typename Op, typename Real,
          size_t N, size_t... TensorDims>
struct binary_operation_field
    : field<binary_operation_field<LhsField, RhsField, Op, Real, N,
                                   TensorDims...>,
            Real, N, TensorDims...> {
 public:
  using this_t =
      binary_operation_field<LhsField, RhsField, Op, Real, N, TensorDims...>;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

 private:
  LhsField m_lhs_field;
  RhsField m_rhs_field;
  Op       m_operator;

 public:
  constexpr binary_operation_field(const LhsField& lhs_field,
                                   const RhsField& rhs_field, const Op& op)
      : m_lhs_field{lhs_field}, m_rhs_field{rhs_field}, m_operator{op} {}

  constexpr binary_operation_field(const LhsField& lhs_field,
                                   const RhsField& rhs_field, Op&& op)
      : m_lhs_field{lhs_field},
        m_rhs_field{rhs_field},
        m_operator{std::move(op)} {}

  //----------------------------------------------------------------------------
  tensor_t evaluate(const pos_t& x, Real t) const {
    return m_operator(m_lhs_field(x, t), m_rhs_field(x, t));
  }
};

//==============================================================================
// operations
//==============================================================================

template <typename Real, size_t N, size_t... TensorDims, typename LhsField,
          typename LhsReal, typename RhsField, typename RhsReal,
          size_t... lhs_tensor_dims, size_t... rhs_tensor_dims, typename Op>
constexpr auto make_binary_operation_field(
    const field<LhsField, LhsReal, N, lhs_tensor_dims...>& lhs,
    const field<RhsField, RhsReal, N, rhs_tensor_dims...>& rhs, Op&& op) {
  return binary_operation_field<LhsField, RhsField, std::decay_t<Op>, Real, N,
                                TensorDims...>{
      lhs.as_derived(), rhs.as_derived(), std::forward<Op>(op)};
}

//------------------------------------------------------------------------------
template <typename LhsField, typename LhsReal, typename RhsField,
          typename RhsReal, size_t N, size_t D>
constexpr auto dot(const field<LhsField, LhsReal, N, D>& lhs,
                   const field<RhsField, RhsReal, N, D>& rhs) {
  return make_binary_operation_field<promote_t<LhsReal, RhsReal>, N>(
      lhs, rhs, [](const auto& lhs, const auto& rhs) { return dot(lhs, rhs); });
}

////------------------------------------------------------------------------------
// template <typename tensor_t, typename Real, size_t... Dims>
// constexpr auto operator-(const base_tensor<tensor_t, Real, Dims...>& t) {
//  tensor<Real, Dims...> negated = t;
//  return negated.unary_operation([](const auto& c){return  -c;});
//}

//------------------------------------------------------------------------------
template <typename LhsField, typename LhsReal, typename RhsField,
          typename RhsReal, size_t N, size_t... TensorDims>
constexpr auto operator+(
    const field<LhsField, LhsReal, N, TensorDims...>& lhs,
    const field<RhsField, RhsReal, N, TensorDims...>& rhs) {
  return make_binary_operation_field<promote_t<LhsReal, RhsReal>, N,
                                     TensorDims...>(
      lhs, rhs, [](const auto& lhs, const auto& rhs) { return lhs + rhs; });
}

//------------------------------------------------------------------------------
template <typename LhsField, typename LhsReal, typename RhsField,
          typename RhsReal, size_t N, size_t TM, size_t TN>
constexpr auto operator*(const field<LhsField, LhsReal, N, TM, TN>& lhs,
                         const field<RhsField, RhsReal, N, TN>&     rhs) {
  return make_binary_operation_field<promote_t<LhsReal, RhsReal>, N, TM>(
      lhs, rhs, [](const auto& lhs, const auto& rhs) { return lhs * rhs; });
}

//------------------------------------------------------------------------------
template <typename LhsField, typename LhsReal, typename RhsField,
          typename RhsReal, size_t... Dims>
constexpr auto operator*(const field<LhsField, LhsReal, Dims...>& lhs,
                         const field<RhsField, RhsReal, Dims...>& rhs) {
  return make_binary_operation_field<promote_t<LhsReal, RhsReal>, Dims...>(
      lhs, rhs, [](const auto& lhs, const auto& rhs) { return lhs * rhs; });
}

////------------------------------------------------------------------------------
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
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
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
////------------------------------------------------------------------------------
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
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
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
////------------------------------------------------------------------------------
// template <typename lhs_tensor_t, typename LhsReal,
//          typename rhs_tensor_t, typename RhsReal,
//          size_t... Dims>
// constexpr auto operator-(
//    const base_tensor<lhs_tensor_t, LhsReal, Dims...>& lhs,
//    const base_tensor<rhs_tensor_t, RhsReal, Dims...>& rhs) {
//  return binary_operation(std::minus<promote_t<LhsReal, RhsReal>>{}, lhs,
//                          rhs);
//}
//
////------------------------------------------------------------------------------
///// matrix-vector-multiplication
// template <typename lhs_tensor_t, typename LhsReal,
//          typename rhs_tensor_t, typename RhsReal, size_t M, size_t N>
// constexpr auto operator*(const base_tensor<lhs_tensor_t, LhsReal, M, N>& lhs,
//                         const base_tensor<rhs_tensor_t, RhsReal, N>& rhs) {
//  tensor<promote_t<LhsReal, RhsReal>, M> product;
//  for (size_t i = 0; i < M; ++i) {
//    product(i) = dot(lhs.template slice<0>(i), rhs);
//  }
//  return product;
//}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
