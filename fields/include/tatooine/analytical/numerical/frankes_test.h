#ifndef TATOOINE_ANALYTICAL_NUMERICAL_FRANKES_TEST_H
#define TATOOINE_ANALYTICAL_NUMERICAL_FRANKES_TEST_H
//==============================================================================
#include <tatooine/field.h>

#include <cmath>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
/// \brief Franke's Test Function.
/// <a href = "https://www.sfu.ca/~ssurjano/franke2d.html">See Here</a>
template <floating_point Real>
struct frankes_test : scalarfield<frankes_test<Real>, Real, 2> {
  using this_type   = frankes_test<Real>;
  using parent_type = scalarfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  //============================================================================
  explicit constexpr frankes_test() noexcept = default;
  //------------------------------------------------------------------------------
  constexpr frankes_test(frankes_test const&)     = default;
  constexpr frankes_test(frankes_test&&) noexcept = default;
  //------------------------------------------------------------------------------
  auto constexpr operator=(frankes_test const&) -> frankes_test& = default;
  auto constexpr operator=(frankes_test&&) noexcept -> frankes_test& = default;
  //------------------------------------------------------------------------------
  ~frankes_test() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto constexpr evaluate(fixed_size_vec<2> auto const& q,
                                        Real const t) const -> tensor_type {
    auto const a     = 9 * q(0) - 2;
    auto const b     = 9 * q(1) - 2;
    auto const c     = 9 * q(0) + 1;
    auto const d     = 9 * q(1) + 1;
    auto const e     = 9 * q(0) - 7;
    auto const f     = 9 * q(1) - 3;
    auto const g     = 9 * q(0) - 4;
    auto const h     = 9 * q(1) - 7;
    auto const term1 = 0.75 * std::exp(-a * a / 4 - b * b / 4);
    auto const term2 = 0.75 * std::exp(-c * c / 49 - d / 10);
    auto const term3 = 0.5 * std::exp(-e * e / 4 - f * f / 4);
    auto const term4 = -0.2 * std::exp(-g * g - h * h);

    return term1 + term2 + term3 + term4;
  }
};
//==============================================================================
frankes_test()->frankes_test<real_number>;
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#include <tatooine/differentiated_field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <floating_point Real>
struct differentiated_field<analytical::numerical::frankes_test<Real>>
    : vectorfield<differentiated_field<
                      analytical::numerical::frankes_test<Real>>,
                  Real, 2> {
  using internal_field_type = analytical::numerical::frankes_test<Real>;
  using this_type           = differentiated_field<internal_field_type>;
  using parent_type         = vectorfield<
      differentiated_field<analytical::numerical::frankes_test<Real>>,
      Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  //============================================================================
  explicit constexpr differentiated_field() noexcept = default;
  //------------------------------------------------------------------------------
  constexpr differentiated_field(differentiated_field const&)     = default;
  constexpr differentiated_field(differentiated_field&&) noexcept = default;
  //------------------------------------------------------------------------------
  auto constexpr operator      =(differentiated_field const&)
      -> differentiated_field& = default;
  auto constexpr operator      =(differentiated_field&&) noexcept
      -> differentiated_field& = default;
  //------------------------------------------------------------------------------
  ~differentiated_field() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto constexpr evaluate(fixed_size_vec<2> auto const& q,
                                        Real const t) const -> tensor_type {
    auto constexpr exp51 = gcem::exp(Real(51));
    auto constexpr exp63 = gcem::exp(Real(63));
    auto constexpr exp65 = gcem::exp(Real(65));
    auto x               = q.x();
    auto y               = q.y();
    auto xx              = x * x;
    auto yy              = y * y;
    auto a = gcem::exp(-(Real(405) * yy) / Real(4) - (Real(9) * y) / Real(10) -
                       (Real(20169) * xx) / Real(196) -
                       (Real(18) * x) / Real(49) - Real(66));
    auto constexpr b = gcem::exp(Real(431) / Real(490));
    auto constexpr c = gcem::exp(Real(59) / Real(490));
    auto d           = gcem::exp((Real(4050) * xx) / Real(49) +
                                 (Real(459) * x) / Real(49) + Real(63));
    auto e =
        gcem::exp((Real(4293) * xx) / Real(196) + (Real(3546) * x) / Real(49));
    auto f = gcem::exp((Real(4050) * xx) / Real(49) +
                       (Real(3123) * x) / Real(98) + Real(51));

    return {
        -(a *
          (b * ((Real(4860) * exp65 * x + Real(540) * exp65) *
                    gcem::exp((Real(405) * yy) / Real(4) +
                              (Real(405) * xx) / Real(4)) +
                c * ((Real(59535) * exp63 * x - Real(13230) * exp63) *
                         gcem::exp(Real(81) * yy + (Real(99) * y) / Real(10) +
                                   (Real(4050) * xx) / Real(49) +
                                   (Real(459) * x) / Real(49)) +
                     (Real(28224) - Real(63504) * x) *
                         gcem::exp((Real(81) * yy) / Real(4) +
                                   (Real(1269) * y) / Real(10) +
                                   (Real(4293) * xx) / Real(196) +
                                   (Real(3546) * x) / Real(49)))) +
           (Real(39690) * exp51 * x - Real(30870) * exp51) *
               gcem::exp(Real(81) * yy + (Real(72) * y) / Real(5) +
                         (Real(4050) * xx) / Real(49) +
                         (Real(3123) * x) / Real(98) + Real(1) / Real(2)))) /
            Real(1960),
        -(a *
          (b * (Real(27) * gcem::exp((Real(405) * yy) / Real(4) +
                                     (Real(405) * xx) / Real(4) + Real(65)) +
                c * ((Real(1215) * d * y - Real(270) * d) *
                         gcem::exp(Real(81) * yy + (Real(99) * y) / Real(10)) +
                     (Real(1008) * e - Real(1296) * e * y) *
                         gcem::exp((Real(81) * yy) / Real(4) +
                                   (Real(1269) * y) / Real(10)))) +
           (Real(810) * f * y - Real(270) * f) *
               gcem::exp(Real(81) * yy + (Real(72) * y) / Real(5) +
                         Real(1) / Real(2)))) /
            Real(40)};
  }
};
//------------------------------------------------------------------------------
template <floating_point Real>
auto diff(analytical::numerical::frankes_test<Real> const& f) {
  return differentiated_field<
      analytical::numerical::frankes_test<Real>>{};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
