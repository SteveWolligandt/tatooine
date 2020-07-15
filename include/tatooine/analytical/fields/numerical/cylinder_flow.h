#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_CYLINDER_FLOW_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_CYLINDER_FLOW_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::numerical {
//==============================================================================
template <typename Real>
struct cylinder_flow : vectorfield<cylinder_flow<Real>, Real, 2> {
  using this_t = cylinder_flow<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  //============================================================================
  using typename parent_t::real_t;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  real_t a = 0;
  real_t Tc = 1;
  real_t L = 2;
  real_t R0 = 1;
  real_t y0 = 1;
  real_t alpha = 1;
  //============================================================================
  [[no_discard]] auto evaluate(pos_t const& p, real_t t) const -> tensor_t {
    real_t const eps = 1e-7;
    auto const&  x   = p(0);
    auto const&  y   = p(1);
    auto const   xn  = x - eps;
    auto const   xp  = x + eps;
    auto const   yn  = y - eps;
    auto const   yp  = y + eps;

    auto const b_xn = std::sqrt(xn * xn + y * y) - 1;
    auto const b_xp = std::sqrt(xp * xp + y * y) - 1;
    auto const b_yn = std::sqrt(x * x + yn * yn) - 1;
    auto const b_yp = std::sqrt(x * x + yp * yp) - 1;

    auto const f_xn  = 1 - std::exp(-a * b_xn * b_xn);
    auto const f_xp  = 1 - std::exp(-a * b_xp * b_xp);
    auto const f_yn  = 1 - std::exp(-a * b_yn * b_yn);
    auto const f_yp  = 1 - std::exp(-a * b_yp * b_yp);

    auto const h1 = std::abs(std::sin(M_PI * t / Tc));
    auto const h2 = std::abs(std::sin(M_PI * (t - Tc / 2) / Tc));

    auto const x1 = 1 + L * ((t / Tc) % 1);
    auto const x2 = 1 + L * (((t - Tc / 2) / Tc) % 1);
    auto const g1 = std::exp(
        -R0((p - x1) * (p - x1) + alpha * alpha * (y - y0) * (y - y0)));
    auto const g2 = std::exp(
        -R0((p - x2) * (p - x2) + alpha * alpha * (y + y0) * (y + y0)));

    auto const s_xn =
        1 - std::exp(-(xn - 1) * (xn - 1) / (alpha * alpha - y * y));
    auto const s_xp =
        1 - std::exp(-(xp - 1) * (xp - 1) / (alpha * alpha - y * y));
    auto const s_yn =
        1 - std::exp(-(x - 1) * (x - 1) / (alpha * alpha - yn * yn));
    auto const s_yp =
        1 - std::exp(-(x - 1) * (x - 1) / (alpha * alpha - yp * yp));

    auto const g_xn   = -w * h1 * g1 + w * h2 * h2 * g2 + u0 * y * s_xn;
    auto const g_xp   = -w * h1 * g1 + w * h2 * h2 * g2 + u0 * y * s_xp;
    auto const g_yn   = -w * h1 * g1 + w * h2 * h2 * g2 + u0 * yn * s_yn;
    auto const g_yp   = -w * h1 * g1 + w * h2 * h2 * g2 + u0 * yp * s_yp;
    auto const psi_xn = f_xn * g_xn;
    auto const psi_xp = f_xp * g_xp;
    auto const psi_yn = f_yn * g_yn;
    auto const psi_yp = f_yp * g_yp;

    return tensor_t{ (psi_yp - psi_yn) / (2 * eps),
                    -(psi_xp - psi_xn) / (2 * eps)};
  }
  //----------------------------------------------------------------------------
  [[no_discard]] auto in_domain(pos_t const& x, real_t t) const -> bool {
    if (sqr_length(x) <= 1) { return false; }
    return true;
  }
};
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif
