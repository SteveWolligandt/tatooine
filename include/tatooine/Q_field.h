#ifndef TATOOINE_Q_FIELD_H
#define TATOOINE_Q_FIELD_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, size_t N>
class Q_field
    : public scalarfield<Q_field<V, N>, typename V::real_t, N> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using real_t = typename V::real_t;
  using this_t = Q_field<V, N>;
  using parent_t =
      field<this_t, real_t, V::num_dimensions()>;
  using typename parent_t::tensor_t;
  using typename parent_t::pos_t;
  //============================================================================
  // fields
  //============================================================================
 private:
  V const& m_v;

  //============================================================================
  // ctor
  //============================================================================
 public:
  template <typename Real>
  Q_field(field<V, Real, N, N> const& v) : m_v{v.as_derived()} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_t evaluate(pos_t const& x, real_t t) const {
    auto J      = diff(m_v, 1e-7)(x, t);
    auto S      = (J + transposed(J)) / 2;
    auto Omega  = (J - transposed(J)) / 2;
    return (sqr_norm(Omega) - sqr_norm(S)) / 2;
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(pos_t const& x, real_t t) const {
    return m_v.in_domain(x, t);
  }
};
//==============================================================================
template <typename V>
class Q_field<V, 3>
    : public field<Q_field<V, 3>, typename V::real_t, 3> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using real_t = typename V::real_t;
  using this_t = Q_field<V, 3>;
  using parent_t =
      field<this_t, real_t, V::num_dimensions()>;
  using typename parent_t::tensor_t;
  using typename parent_t::pos_t;
  //============================================================================
  // fields
  //============================================================================
 private:
  V const& m_v;

  //============================================================================
  // ctor
  //============================================================================
 public:
  template <typename Real>
  Q_field(const field<V, Real, 3, 3>& v) : m_v{v.as_derived()} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    auto J = diff(m_v, 1e-7)(x, t);
    //auto const& a =  J(0,0);
    //auto const& b =  J(0,1);
    //auto const& c =  J(0,2);
    //auto const& d =  J(1,0);
    //auto const& e =  J(1,1);
    //auto const& f =  J(1,2);
    //auto const& g =  J(2,0);
    //auto const& h =  J(2,1);
    //auto const& i =  J(2,2);
    //auto const sqrt2 = std::sqrt(real_t(2));
    //return -(sqrt2 * sqrt(2 * i * i + h * h + 2 * f * h + g * g + 2 * c * g +
    //                      f * f + 2 * e * e + d * d + 2 * b * d + c * c +
    //                      b * b + 2 * a * a) -
    //         sqrt2 * std::sqrt(h * h - 2 * f * h + g * g - 2 * c * g + f * f +
    //                           d * d - 2 * b * d + c * c + b * b)) /
    //       4.0;
    return -(J(0, 0) * J(0, 0) +
             J(1, 1) * J(1, 1) +
             J(2, 2) * J(2, 2) +
             2 * J(0, 1) * J(1, 0) +
             2 * J(0, 2) * J(2, 0) +
             2 * J(1, 2) * J(2, 1)) /
           2;
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t t) const {
    return m_v.in_domain(x, t);
  }
};
//==============================================================================
template <typename V, typename Real, size_t N>
auto Q(field<V, Real, N, N> const& vf) {
  return Q_field<V, N>{vf.as_derived()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
