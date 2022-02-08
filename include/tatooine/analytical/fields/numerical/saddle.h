#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_SADDLE_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_SADDLE_H
//==============================================================================
#include <tatooine/differentiated_field.h>
#include <tatooine/field.h>
#include <tatooine/flowmap_gradient_central_differences.h>
#include <tatooine/numerical_flowmap.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
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
  [[nodiscard]] auto constexpr evaluate(pos_type const& x, Real const /*t*/) const
      -> tensor_type {
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
   [[nodiscard]] auto constexpr evaluate(pos_type const& x, Real const /*t*/) const
       -> tensor_type {
     auto const R = Mat2<Real>{
         {gcem::cos(m_angle_in_radians), -gcem::sin(m_angle_in_radians)},
         {gcem::sin(m_angle_in_radians),  gcem::cos(m_angle_in_radians)}};
     return R * tensor_type{-x(0), x(1)} * transposed(R);
  }
};
//==============================================================================
rotated_saddle()->rotated_saddle<real_number>;
//==============================================================================
// template <typename Real>
// struct saddle_flowmap {
//  using real_type = Real;
//  using vec_t  = vec<Real, 2>;
//  using pos_type  = vec_t;
//  saddle_flowmap() = default;
//  saddle_flowmap(saddle<Real> const&) {}
//  static auto constexpr num_dimensions() { return 2; }
//  //----------------------------------------------------------------------------
//  auto constexpr evaluate(pos_type const& x, Real const [>t<],
//                          Real const   tau) const -> pos_type {
//    return {std::exp(-tau) * x(0), std::exp(tau) * x(1)};
//  }
//  //----------------------------------------------------------------------------
//  auto constexpr operator()(pos_type const& x, Real const t, Real const tau)
//  const
//      -> pos_type {
//    return evaluate(x, t, tau);
//  }
//};
////------------------------------------------------------------------------------
// template <
//     template <typename, size_t> typename ODESolver =
//     ode::vclibs::rungekutta43, template <typename> typename
//     InterpolationKernel = interpolation::cubic, floating_point Real>
// auto constexpr flowmap(vectorfield<saddle<Real>, Real, 2> const& v,
//                        tag::numerical_t [>tag<]) {
//   return numerical_flowmap<saddle<Real>, ODESolver, InterpolationKernel>{v};
// }
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <floating_point Real>
// auto constexpr flowmap(vectorfield<saddle<Real>, Real, 2> const& [>v<],
//                        tag::analytical_t [>tag<]) {
//   return analytical::fields::numerical::saddle_flowmap<Real>{};
// }
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <floating_point Real>
// auto constexpr flowmap(vectorfield<saddle<Real>, Real, 2> const& v) {
//   return flowmap(v, tag::analytical);
// }
////==============================================================================
// template <floating_point Real>
// struct saddle_flowmap_gradient {
//   using real_type     = Real;
//   using vec_t      = vec<Real, 2>;
//   using pos_type      = vec_t;
//   using mat_t      = mat<real_type, 2, 2>;
//   using gradient_t = mat_t;
//   static auto constexpr num_dimensions() { return 2; }
//   //----------------------------------------------------------------------------
//   auto constexpr evaluate(pos_type const& [>x*/, Real const /*t<],
//                           Real const tau) const -> gradient_t {
//     return {{std::exp(-tau), real_type(0)},
//             {real_type(0), std::exp(tau)}};
//   }
//   //----------------------------------------------------------------------------
//   auto constexpr operator()(pos_type const& x, Real const t,
//                             Real const tau) const {
//     return evaluate(x, t, tau);
//   }
// };
////------------------------------------------------------------------------------
// template <floating_point Real>
// auto diff(saddle_flowmap<Real> const&, tag::analytical_t [>tag<]) {
//   return saddle_flowmap_gradient<Real>{};
// }
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <floating_point Real>
// auto diff(saddle_flowmap<Real> const& flowmap, tag::central_t [>tag<],
//           Real const                  epsilon) {
//   return flowmap_gradient_central_differences<saddle_flowmap<Real>>{flowmap,
//                                                                     epsilon};
// }
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <floating_point Real>
// auto constexpr diff(saddle_flowmap<Real> const& flowmap, tag::central_t
// [>tag<],
//                     vec<Real, 2>                epsilon) {
//   return flowmap_gradient_central_differences<saddle_flowmap<Real>>{flowmap,
//                                                                     epsilon};
// }
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <floating_point Real>
// auto diff(saddle_flowmap<Real> const& flowmap) {
//   return diff(flowmap, tag::analytical);
// }
////==============================================================================
//}  // namespace tatooine::analytical::fields::numerical
////==============================================================================
// namespace tatooine {
////==============================================================================
// template <floating_point Real>
// struct differentiated_field<analytical::fields::numerical::saddle<Real>>
//     : matrixfield<analytical::fields::numerical::saddle<Real>, Real, 2> {
//   using this_type =
//       differentiated_field<analytical::fields::numerical::saddle<Real>>;
//   using parent_type = matrixfield<this_type, Real, 2>;
//   using typename parent_type::pos_type;
//   using typename parent_type::tensor_type;
//
//   //============================================================================
//  public:
//   auto constexpr evaluate(pos_type const& [>x*/, Real const /*t<]) const
//       -> tensor_type {
//     return {{-1, 0}, {0, 1}};
//   }
//   //----------------------------------------------------------------------------
//   auto constexpr in_domain(pos_type const& [>x*/, Real const /*t<]) const ->
//   bool {
//     return true;
//   }
// };
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
#endif
