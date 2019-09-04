#ifndef TATOOINE_SYMBOLIC_FIELD_H
#define TATOOINE_SYMBOLIC_FIELD_H

#include "crtp.h"
#include "field.h"
#include "symbolic.h"
#include "tensor.h"

//==============================================================================
namespace tatooine::symbolic {
//==============================================================================

template <typename real_t, size_t N, size_t... TensorDims>
struct field : tatooine::field<field<real_t, N, TensorDims...>, real_t, N,
                               TensorDims...> {
  using this_t   = field<real_t, N, TensorDims...>;
  using parent_t = tatooine::field<this_t, real_t, N, TensorDims...>;
  using parent_t::num_dimensions;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
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
template <typename LhsReal, typename RhsReal, size_t N, size_t D>
constexpr auto dot(const field<LhsReal, N, D>& lhs,
                   const field<RhsReal, N, D>& rhs) {
  return field<promote_t<LhsReal, RhsReal>, N>{dot(lhs.expr(), rhs.expr())};
}

//------------------------------------------------------------------------------
template <typename LhsReal, typename RhsReal, size_t N, size_t... TensorDims>
constexpr auto operator+(const field<LhsReal, N, TensorDims...>& lhs,
                         const field<RhsReal, N, TensorDims...>& rhs) {
  return field<promote_t<LhsReal, RhsReal>, N, TensorDims...>{lhs.expr() +
                                                              rhs.expr()};
}
//------------------------------------------------------------------------------
template <typename LhsReal, typename RhsReal, size_t N, size_t... TensorDims>
constexpr auto operator-(const field<LhsReal, N, TensorDims...>& lhs,
                         const field<RhsReal, N, TensorDims...>& rhs) {
  return field<promote_t<LhsReal, RhsReal>, N, TensorDims...>{lhs.expr() -
                                                              rhs.expr()};
}

//------------------------------------------------------------------------------
template <typename LhsReal, typename RhsReal, size_t... TensorDims>
constexpr auto operator*(const field<LhsReal, TensorDims...>& lhs,
                         const field<RhsReal, TensorDims...>& rhs) {
  return field<promote_t<LhsReal, RhsReal>, TensorDims...>{lhs.expr() *
                                                           rhs.expr()};
}

//------------------------------------------------------------------------------
template <typename LhsReal, typename RhsReal, size_t... TensorDims>
constexpr auto operator/(const field<LhsReal, TensorDims...>& lhs,
                         const field<RhsReal, TensorDims...>& rhs) {
  return field<promote_t<LhsReal, RhsReal>, TensorDims...>{lhs.expr() /
                                                           rhs.expr()};
}

//------------------------------------------------------------------------------
template <typename LhsReal, typename RhsReal, size_t N, size_t D0, size_t D1>
constexpr auto operator*(const field<LhsReal, N, D0, D1>& lhs,
                         const field<RhsReal, N, D1>&     rhs) {
  return field<promote_t<LhsReal, RhsReal>, N, D0>{lhs.expr() * rhs.expr()};
}

//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================

#endif
