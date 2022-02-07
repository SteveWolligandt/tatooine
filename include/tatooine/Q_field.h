#ifndef TATOOINE_Q_FIELD_H
#define TATOOINE_Q_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/differentiated_field.h>
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
  using parent_type =
      scalarfield<this_t, field_real_t<V>, field_num_dimensions<V>>;
  using typename parent_type::real_t;
  using typename parent_type::pos_t;
  using typename parent_type::tensor_t;
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
  explicit Q_field(V_&& v) : m_v{std::forward<V_>(v)} {}

  Q_field(Q_field const&)     = default;
  Q_field(Q_field&&) noexcept = default;
  auto operator=(Q_field const&) -> Q_field& = default;
  auto operator=(Q_field&&) noexcept -> Q_field& = default;
  ~Q_field()                  = default;

  //============================================================================
  // methods
  //============================================================================
  constexpr auto evaluate(pos_t const& x, real_t const t) const -> tensor_t {
    auto J     = diff(m_v, 1e-7)(x, t);
    auto S     = (J + transposed(J)) / 2;
    auto Omega = (J - transposed(J)) / 2;
    return (sqr_norm(Omega, 2) - sqr_norm(S, 2)) / 2;
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& x, real_t const t) const -> bool {
    return m_v.in_domain(x, t);
  }
};
//==============================================================================
template <typename V>
auto Q(
    V&& v) requires is_vectorfield<std::decay_t<std::remove_pointer_t<V>>> {
  return Q_field<std::decay_t<V>>{std::forward<V>(v)};
}
//------------------------------------------------------------------------------
template <typename V>
auto Q(V const& v) requires is_vectorfield<
    std::decay_t<std::remove_pointer_t<V>>> {
  return Q_field<V const&>{v};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
