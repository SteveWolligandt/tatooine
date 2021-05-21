#ifndef TATOOINE_Q_FIELD_H
#define TATOOINE_Q_FIELD_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V>
class Q_field
    : public scalarfield<Q_field<V>, field_real_t<V>, field_num_dimensions<V>> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using this_t = Q_field<V>;
  using parent_t =
      scalarfield<this_t, field_real_t<V>, field_num_dimensions<V>>;
  using typename parent_t::real_t;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  // fields
  //============================================================================
 private:
  V m_v;

  //============================================================================
  // ctor
  //============================================================================
 public:
  template <typename V_>
  Q_field(V_ v) : m_v{std::forward<V_>(v)} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    if constexpr (V::num_dimensions() == 3) {
      auto J = diff(m_v, 1e-7)(x, t);
      // auto const& a =  J(0,0);
      // auto const& b =  J(0,1);
      // auto const& c =  J(0,2);
      // auto const& d =  J(1,0);
      // auto const& e =  J(1,1);
      // auto const& f =  J(1,2);
      // auto const& g =  J(2,0);
      // auto const& h =  J(2,1);
      // auto const& i =  J(2,2);
      // auto const sqrt2 = std::sqrt(real_t(2));
      // return -(sqrt2 * sqrt(2 * i * i + h * h + 2 * f * h + g * g + 2 * c * g
      // +
      //                      f * f + 2 * e * e + d * d + 2 * b * d + c * c +
      //                      b * b + 2 * a * a) -
      //         sqrt2 * std::sqrt(h * h - 2 * f * h + g * g - 2 * c * g + f * f
      //         +
      //                           d * d - 2 * b * d + c * c + b * b)) /
      //       4.0;
      return -(J(0, 0) * J(0, 0) + J(1, 1) * J(1, 1) + J(2, 2) * J(2, 2) +
               2 * J(0, 1) * J(1, 0) + 2 * J(0, 2) * J(2, 0) +
               2 * J(1, 2) * J(2, 1)) /
             2;
    } else {
      auto J     = diff(m_v, 1e-7)(x, t);
      auto S     = (J + transposed(J)) / 2;
      auto Omega = (J - transposed(J)) / 2;
      return (sqr_norm(Omega) - sqr_norm(S)) / 2;
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& x, real_t const t) const -> bool final {
    return m_v.in_domain(x, t);
  }
};
//==============================================================================
template <typename V,
          enable_if<is_vectorfield_v<std::decay_t<std::remove_pointer_t<V>>>>>
auto Q(V&& v) {
  return Q_field<std::decay_t<V>>{std::forward<V>(v)};
}
//------------------------------------------------------------------------------
template <typename V,
          enable_if<is_vectorfield_v<std::decay_t<std::remove_pointer_t<V>>>>>
auto Q(V const& v) {
  return Q_field<V const&>{v};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
