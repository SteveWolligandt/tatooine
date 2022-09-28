#ifndef TATOOINE_ANALYTICAL_NUMERICAL_SADDLE_H
#define TATOOINE_ANALYTICAL_NUMERICAL_SADDLE_H
//==============================================================================
#include <tatooine/differentiated_field.h>
#include <tatooine/differentiated_flowmap.h>
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
template <floating_point Real>
struct saddle : vectorfield<saddle<Real>, Real, 2> {
  using this_type   = saddle<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //============================================================================
  constexpr saddle() noexcept {}
  constexpr saddle(saddle const&)     = default;
  constexpr saddle(saddle&&) noexcept = default;
  auto constexpr operator=(saddle const&) -> saddle& = default;
  auto constexpr operator=(saddle&&) noexcept -> saddle& = default;
  //----------------------------------------------------------------------------
  virtual ~saddle() = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto constexpr evaluate(pos_type const& x,
                                        Real const /*t*/) const -> tensor_type {
    return tensor_type{-x(0), x(1)};
  }
};
//==============================================================================
saddle()->saddle<real_number>;
//==============================================================================
template <floating_point Real>
struct rotated_saddle : vectorfield<rotated_saddle<Real>, Real, 2> {
  using this_type   = rotated_saddle<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  Real m_angle_in_radians;
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  explicit constexpr rotated_saddle(Real angle_in_radians = M_PI / 4) noexcept
      : m_angle_in_radians{angle_in_radians} {}
  constexpr rotated_saddle(rotated_saddle const&)     = default;
  constexpr rotated_saddle(rotated_saddle&&) noexcept = default;
  auto constexpr operator=(rotated_saddle const&) -> rotated_saddle& = default;
  auto constexpr operator=(rotated_saddle&&) noexcept
      -> rotated_saddle& = default;
  //----------------------------------------------------------------------------
  virtual ~rotated_saddle() = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto constexpr evaluate(pos_type const& x,
                                        Real const /*t*/) const -> tensor_type {
    auto const R = Mat2<Real>{
        {gcem::cos(m_angle_in_radians), -gcem::sin(m_angle_in_radians)},
        {gcem::sin(m_angle_in_radians), gcem::cos(m_angle_in_radians)}};
    return R * tensor_type{-x(0), x(1)} * transposed(R);
  }
};
//==============================================================================
rotated_saddle()->rotated_saddle<real_number>;
//==============================================================================
template <typename Real>
struct saddle_flowmap {
  using real_type  = Real;
  using vec_type   = vec<Real, 2>;
  using pos_type   = vec_type;
  saddle_flowmap() = default;
  saddle_flowmap(saddle<Real> const&) {}
  static auto constexpr num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto constexpr evaluate(pos_type const& x, Real const t, Real const tau) const
      -> pos_type {
    return {std::exp(-tau) * x(0), std::exp(tau) * x(1)};
  }
  //----------------------------------------------------------------------------
  auto constexpr operator()(pos_type const& x, Real const t,
                            Real const tau) const -> pos_type {
    return evaluate(x, t, tau);
  }
};
template <floating_point Real>
auto constexpr flowmap(saddle<Real> const& /*v*/, tag::analytical_t /*tag*/) {
  return analytical::numerical::saddle_flowmap<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <floating_point Real>
auto constexpr flowmap(saddle<Real> const& v) {
  return flowmap(v, tag::analytical);
}
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
namespace tatooine {
//==============================================================================
template <floating_point Real>
struct differentiated_field<analytical::numerical::saddle<Real>>
    : matrixfield<analytical::numerical::saddle<Real>, Real, 2> {
  using this_type =
      differentiated_field<analytical::numerical::saddle<Real>>;
  using parent_type = matrixfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;

  //============================================================================
 public:
  auto constexpr evaluate(pos_type const& x, Real const t) const
      -> tensor_type {
    return {{-1, 0}, {0, 1}};
  }
};
//==============================================================================
template <floating_point Real>
struct differentiated_flowmap<
    analytical::numerical::saddle_flowmap<Real>> {
  using real_type     = Real;
  using vec_type      = vec<Real, 2>;
  using pos_type      = vec_type;
  using mat_type      = mat<real_type, 2, 2>;
  using gradient_type = mat_type;
  static auto constexpr num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto constexpr evaluate(pos_type const& x, Real const t, Real const tau) const
      -> gradient_type {
    return {{std::exp(-tau), real_type(0)}, {real_type(0), std::exp(tau)}};
  }
  //----------------------------------------------------------------------------
  auto constexpr operator()(pos_type const& x, Real const t,
                            Real const tau) const {
    return evaluate(x, t, tau);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
