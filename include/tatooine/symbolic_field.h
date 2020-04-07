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

  static auto x(size_t i) -> auto& { return symbol::x(i); }
  static auto t() -> auto& { return symbol::t(); }

 private:
  symtensor_t m_expr;

 protected:
  void set_expr(const symtensor_t& ex) { m_expr = ex; }
  void set_expr(symtensor_t&& ex) { m_expr = std::move(ex); }

 public:
  constexpr field() = default;
  explicit constexpr field(const symtensor_t& ex) : m_expr{ex} {}
  explicit constexpr field(symtensor_t&& ex) : m_expr{std::move(ex)} {}
  //----------------------------------------------------------------------------
  [[nodiscard]] auto expr() const -> const auto& { return m_expr; }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  auto evaluate(const pos_t& _x, double _t,
                    std::index_sequence<Is...> /*is*/) const -> tensor_t {
    return evtod<real_t>(m_expr, (x(Is) == _x(Is))..., t() == _t);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto evaluate(const pos_t& _x, double _t) const -> tensor_t {
    return evaluate(_x, _t, std::make_index_sequence<num_dimensions()>{});
  }
  constexpr auto in_domain(const pos_t& /*x*/, double /*t*/) const { return true; }
};

//==============================================================================
// operations
//==============================================================================
template <typename Real0, typename Real1, size_t N, size_t D>
constexpr auto dot(const field<Real0, N, D>& lhs,
                   const field<Real1, N, D>& rhs) {
  return field<promote_t<Real0, Real1>, N>{dot(lhs.expr(), rhs.expr())};
}

//------------------------------------------------------------------------------
template <typename Real0, typename Real1, size_t N, size_t... TensorDims>
constexpr auto operator+(const field<Real0, N, TensorDims...>& lhs,
                         const field<Real1, N, TensorDims...>& rhs) {
  return field<promote_t<Real0, Real1>, N, TensorDims...>{lhs.expr() +
                                                              rhs.expr()};
}
//------------------------------------------------------------------------------
template <typename Real0, typename Real1, size_t N, size_t... TensorDims>
constexpr auto operator-(const field<Real0, N, TensorDims...>& lhs,
                         const field<Real1, N, TensorDims...>& rhs) {
  return field<promote_t<Real0, Real1>, N, TensorDims...>{lhs.expr() -
                                                              rhs.expr()};
}

//------------------------------------------------------------------------------
template <typename Real0, typename Real1, size_t... TensorDims>
constexpr auto operator*(const field<Real0, TensorDims...>& lhs,
                         const field<Real1, TensorDims...>& rhs) {
  return field<promote_t<Real0, Real1>, TensorDims...>{lhs.expr() *
                                                           rhs.expr()};
}

//------------------------------------------------------------------------------
template <typename Real0, typename Real1, size_t... TensorDims>
constexpr auto operator/(const field<Real0, TensorDims...>& lhs,
                         const field<Real1, TensorDims...>& rhs) {
  return field<promote_t<Real0, Real1>, TensorDims...>{lhs.expr() /
                                                           rhs.expr()};
}

//------------------------------------------------------------------------------
template <typename Real0, typename Real1, size_t N, size_t D0, size_t D1>
constexpr auto operator*(const field<Real0, N, D0, D1>& lhs,
                         const field<Real1, N, D1>&     rhs) {
  return field<promote_t<Real0, Real1>, N, D0>{lhs.expr() * rhs.expr()};
}

//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================
namespace tatooine{
//==============================================================================
template <typename T>
struct is_symbolic_field_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, size_t... TensorDims>
struct is_symbolic_field_impl<
    field<symbolic::field<Real, N, TensorDims...>, Real, N, TensorDims...>> 
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, size_t... TensorDims>
struct is_symbolic_field_impl<symbolic::field<Real, N, TensorDims...>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_symbolic_field_v = is_symbolic<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
constexpr auto is_symbolic_field(T&&) noexcept {
  return false;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, size_t... TensorDims>
constexpr auto is_symbolic_field(
    const field<symbolic::field<Real, N, TensorDims...>, Real, N,
                TensorDims...>&) noexcept {
  return true;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
