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

template <typename derived_t, typename Real, size_t N, size_t... TensorDims>
struct field : crtp<derived_t> {
  using real_t   = Real;
  using this_t   = field<derived_t, real_t, N, TensorDims...>;
  using parent_t = crtp<derived_t>;
  using pos_t    = tensor<real_t, N>;
  using tensor_t = std::conditional_t<sizeof...(TensorDims) == 0, real_t,
                                      tensor<real_t, TensorDims...>>;
  static constexpr auto num_dimensions() { return N; }
  static constexpr auto has_in_domain() { return has_in_domain_v<derived_t>; }
  using parent_t::as_derived;

  //============================================================================
  constexpr decltype(auto) operator()(const pos_t &x, real_t t = 0) const {
    return evaluate(x, t);
  }

  //----------------------------------------------------------------------------
  constexpr decltype(auto) evaluate(const pos_t &x, real_t t = 0) const {
    if (!in_domain(x, t)) { throw out_of_domain{}; }
    return as_derived().evaluate(x, t);
  }

  //----------------------------------------------------------------------------
  constexpr decltype(auto) in_domain([[maybe_unused]] const pos_t &x,
                                     [[maybe_unused]] real_t t = 0) const {
    if constexpr (has_in_domain()) {
      return as_derived().in_domain(x, t);
    } else { 
      return true;
    }
  }
};

//==============================================================================
//template <typename Op, typename field_t, typename real_t, size_t N, size_t... TensorDims>
//struct unary_operation_field : field<unary_operation_field<field_t, real_t, N, TensorDims...>> {
//  public:
//   using this_t   = unary_operation_field<field_t, real_t, N, TensorDims...>;
//   using parent_t = field<this_t, real_t, N, TensorDims...>;
//   using typename parent_t::pos_t;
//   using typename parent_t::tensor_t;
//
//  private:
//   field_t m_field;
//   Op      m_operator;
//
//  public:
//   template <typename _real_t, size_t _N, size_t... _TensorDims>
//   unary_operation_field(field<field_t, _real_t, _N, _TensorDims...>& _field,
//                         Op&& op)
//       : m_field{_field}, m_operator{op} {}
//
//   tensor_t evaluate(const pos_t& x, real_t t) {
//     return m_operator(m_field(x, t));
//   }
//};

//==============================================================================
template <typename lhs_field_t, typename rhs_field_t, typename Op,
          typename real_t, size_t N, size_t... TensorDims>
struct binary_operation_field
    : field<binary_operation_field<lhs_field_t, rhs_field_t, Op, real_t, N,
                                   TensorDims...>,
            real_t, N, TensorDims...> {
 public:
  using this_t = binary_operation_field<lhs_field_t, rhs_field_t, Op, real_t, N,
                                        TensorDims...>;
  using parent_t = field<this_t, real_t, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

 private:
  lhs_field_t m_lhs_field;
  rhs_field_t m_rhs_field;
  Op          m_operator;

 public:
  constexpr binary_operation_field(const lhs_field_t& lhs_field,
                         const rhs_field_t& rhs_field, const Op& op)
      : m_lhs_field{lhs_field},
        m_rhs_field{rhs_field},
        m_operator{op} {}

  constexpr binary_operation_field(const lhs_field_t& lhs_field,
                         const rhs_field_t& rhs_field, Op&& op)
      : m_lhs_field{lhs_field},
        m_rhs_field{rhs_field},
        m_operator{std::move(op)} {}

  //----------------------------------------------------------------------------
  tensor_t evaluate(const pos_t& x, real_t t) const {
    return m_operator(m_lhs_field(x, t), m_rhs_field(x, t));
  }
};

//==============================================================================
// operations
//==============================================================================

template <typename real_t, size_t N, size_t... TensorDims, typename lhs_field_t,
          typename lhs_real_t, typename rhs_field_t, typename rhs_real_t,
          size_t... lhs_tensor_dims, size_t... rhs_tensor_dims, typename Op>
constexpr auto make_binary_operation_field(
    const field<lhs_field_t, lhs_real_t, N, lhs_tensor_dims...>& lhs,
    const field<rhs_field_t, rhs_real_t, N, rhs_tensor_dims...>& rhs, Op&& op) {
  return binary_operation_field<lhs_field_t, rhs_field_t, std::decay_t<Op>,
                                real_t, N, TensorDims...>{
      lhs.as_derived(), rhs.as_derived(), std::forward<Op>(op)};
}

//------------------------------------------------------------------------------
template <typename lhs_field_t, typename lhs_real_t, typename rhs_field_t,
          typename rhs_real_t, size_t N, size_t D>
constexpr auto dot(const field<lhs_field_t, lhs_real_t, N, D>& lhs,
                   const field<rhs_field_t, rhs_real_t, N, D>& rhs) {
  return make_binary_operation_field<promote_t<lhs_real_t, rhs_real_t>, N>(
      lhs, rhs, [](const auto& lhs, const auto& rhs) { return dot(lhs, rhs); });
}

////------------------------------------------------------------------------------
//template <typename tensor_t, typename real_t, size_t... Dims>
//constexpr auto operator-(const base_tensor<tensor_t, real_t, Dims...>& t) {
//  tensor<real_t, Dims...> negated = t;
//  return negated.unary_operation([](const auto& c){return  -c;});
//}

//------------------------------------------------------------------------------
template <typename lhs_field_t, typename lhs_real_t, typename rhs_field_t,
          typename rhs_real_t, size_t N, size_t... TensorDims>
constexpr auto operator+(const field<lhs_field_t, lhs_real_t, N, TensorDims...>& lhs,
                         const field<rhs_field_t, rhs_real_t, N, TensorDims...>& rhs) {
  return make_binary_operation_field<promote_t<lhs_real_t, rhs_real_t>, N,
                                     TensorDims...>(
      lhs, rhs, [](const auto& lhs, const auto& rhs) { return lhs + rhs; });
}

////------------------------------------------------------------------------------
//template <typename tensor_t, typename tensor_real_t, typename scalar_real_t,
//          size_t... Dims,
//          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
//                           is_complex_v<scalar_real_t> ||
//                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
//constexpr auto operator*(const base_tensor<tensor_t, tensor_real_t, Dims...>& t,
//                         const scalar_real_t scalar) {
//  return unary_operation(
//      [scalar](const auto& component) { return component * scalar; }, t);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename tensor_t, typename tensor_real_t, typename scalar_real_t,
//          size_t... Dims,
//          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
//                           is_complex_v<scalar_real_t> ||
//                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
//constexpr auto operator*(
//    const scalar_real_t                                  scalar,
//    const base_tensor<tensor_t, tensor_real_t, Dims...>& t) {
//  return unary_operation(
//      [scalar](const auto& component) { return component * scalar; }, t);
//}
//
////------------------------------------------------------------------------------
//template <typename tensor_t, typename tensor_real_t, typename scalar_real_t,
//          size_t... Dims,
//          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
//                           is_complex_v<scalar_real_t> ||
//                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
//constexpr auto operator/(const base_tensor<tensor_t, tensor_real_t, Dims...>& t,
//                         const scalar_real_t scalar) {
//  return unary_operation(
//      [scalar](const auto& component) { return component / scalar; }, t);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename tensor_t, typename tensor_real_t, typename scalar_real_t,
//          size_t... Dims,
//          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
//                           is_complex_v<scalar_real_t> ||
//                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
//constexpr auto operator/(
//    const scalar_real_t                                  scalar,
//    const base_tensor<tensor_t, tensor_real_t, Dims...>& t) {
//  return unary_operation(
//      [scalar](const auto& component) { return scalar / component; }, t);
//}
//
////------------------------------------------------------------------------------
//template <typename lhs_tensor_t, typename lhs_real_t,
//          typename rhs_tensor_t, typename rhs_real_t,
//          size_t... Dims>
//constexpr auto operator-(
//    const base_tensor<lhs_tensor_t, lhs_real_t, Dims...>& lhs,
//    const base_tensor<rhs_tensor_t, rhs_real_t, Dims...>& rhs) {
//  return binary_operation(std::minus<promote_t<lhs_real_t, rhs_real_t>>{}, lhs,
//                          rhs);
//}
//
////------------------------------------------------------------------------------
///// matrix-vector-multiplication
//template <typename lhs_tensor_t, typename lhs_real_t,
//          typename rhs_tensor_t, typename rhs_real_t, size_t M, size_t N>
//constexpr auto operator*(const base_tensor<lhs_tensor_t, lhs_real_t, M, N>& lhs,
//                         const base_tensor<rhs_tensor_t, rhs_real_t, N>& rhs) {
//  tensor<promote_t<lhs_real_t, rhs_real_t>, M> product;
//  for (size_t i = 0; i < M; ++i) {
//    product(i) = dot(lhs.template slice<0>(i), rhs);
//  }
//  return product;
//}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
