#ifndef TATOOINE_SYMBOLIC_FIELD_H
#define TATOOINE_SYMBOLIC_FIELD_H

#include "symbolic.h"
#include "crtp.h"
#include "tensor.h"
#include "field.h"

//==============================================================================
namespace tatooine::symbolic {
//==============================================================================

template <typename real_t, size_t N, size_t... TensorDims>
struct field : tatooine::field<field<real_t, N, TensorDims...>, real_t, N,
                               TensorDims...> {
  using this_t   = field<real_t, N, TensorDims...>;
  using parent_t = tatooine::field<this_t, real_t, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using parent_t::num_dimensions;
  using symtensor_t = tensor<GiNaC::ex, TensorDims...>;

  static auto& x(size_t i) { return symbol::x(i); }
  static auto& t() { return symbol::t(); }

 private:
  symtensor_t m_expr;

 protected:
  void set_expr(const symtensor_t& ex) { m_expr = ex; }
  void set_expr(symtensor_t&& ex) { m_expr = std::move(ex); }

 public:
  constexpr field() = default;
  constexpr field(const symtensor_t& ex) : m_expr{ex} {}
  constexpr field(symtensor_t&& ex) : m_expr{std::move(ex)} {}

  //----------------------------------------------------------------------------
  [[nodiscard]] const auto& expr() const { return m_expr; }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  tensor_t evaluate(const pos_t& _x, double _t,
                    std::index_sequence<Is...> /*is*/) const {
    return evtod<real_t>(m_expr, (x(Is) == _x(Is))..., t() == _t);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  tensor_t evaluate(const pos_t& _x, double _t) const {
    return evaluate(_x, _t, std::make_index_sequence<num_dimensions()>{});
  }
};

//==============================================================================
// operations
//==============================================================================
template <typename lhs_real_t, typename rhs_real_t, size_t N, size_t D>
constexpr auto dot(const field<lhs_real_t, N, D>& lhs,
                   const field<rhs_real_t, N, D>& rhs) {
  return field<promote_t<lhs_real_t, rhs_real_t>, N>{
      dot(lhs.expr(), rhs.expr())};
}

//------------------------------------------------------------------------------
template <typename lhs_real_t, typename rhs_real_t, size_t N,
          size_t... TensorDims>
constexpr auto operator+(const field<lhs_real_t, N, TensorDims...>& lhs,
                         const field<rhs_real_t, N, TensorDims...>& rhs) {
  return field<promote_t<lhs_real_t, rhs_real_t>, N, TensorDims...>{lhs.expr() +
                                                                    rhs.expr()};
}

//------------------------------------------------------------------------------
template <typename lhs_real_t, typename rhs_real_t, size_t N,
          size_t D0, size_t D1>
constexpr auto operator*(const field<lhs_real_t, N, D0, D1>& lhs,
                         const field<rhs_real_t, N, D1>& rhs) {
  return field<promote_t<lhs_real_t, rhs_real_t>, N, D0>{lhs.expr() *
                                                         rhs.expr()};
}

//------------------------------------------------------------------------------
// template <typename field_t, typename real_t, size_t N>
//[[nodiscard]] auto newton_raphson(const field<field_t, real_t, N, N>& vf,
//                                  typename field_t::pos_t x, double t, size_t
//                                  n, double precision = 1e-10) {
// jacobian      j{vf};
// GiNaC::matrix step = GiNaC::matrix{
//    {vf.x(0)}, {vf.x(1)}}.sub(j.expr().inverse().mul(vf.expr()));
// step.evalm();
//
// for (size_t i = 0; i < n; ++i) {
//  auto x_expr = GiNaC::evalm(
//      ev(vf.expr(), vf.x(0) == x[0], vf.x(1) == x[1], vf.t() == t));
//  typename vf_t::pos_t y{evtod(step(0, 0), vf.x(0) == x[0],
//                                           vf.x(1) == x[1], vf.t() == t),
//                         evtod(step(1, 0), vf.x(0) == x[0],
//                                           vf.x(1) == x[1], vf.t() == t)};
//  if (length(x - y) < precision) {
//    x = std::move(y);
//    break;
//  };
//  x = std::move(y);
//}
// return x;
//}

//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================

#endif
