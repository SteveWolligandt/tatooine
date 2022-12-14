#ifndef TATOOINE_STEADIFICATION_SETTINGS_H
#define TATOOINE_STEADIFICATION_SETTINGS_H

#include "datasets.h"

//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename Real>
struct settings;
//==============================================================================
template <typename Real> struct settings<numerical::doublegyre<Real>> {
  static constexpr std::string_view name = "doublegyre";
  static constexpr Real             eps  = 1e-3;
  static constexpr boundingbox<Real, 2> domain{vec{eps, eps},
                                               vec{2 - eps, 1 - eps}};
  static constexpr vec<size_t, 2>   render_resolution{1000, 500};
};
//==============================================================================
template <typename Real>
struct settings<fixed_time_field<numerical::doublegyre<Real>, Real, 2, 2>> {
  static constexpr std::string_view     name = "fixed_time_doublegyre";
  static constexpr Real                 eps  = 1e-4;
  static constexpr boundingbox<Real, 2> domain{vec{eps, eps},
                                               vec{2 - eps, 1 - eps}};
  static constexpr vec<size_t, 2>       render_resolution{500, 250};
};
//==============================================================================
template <typename Real> struct settings<numerical::sinuscosinus<Real>> {
  static constexpr std::string_view name = "sinuscosinus";
  static constexpr boundingbox<Real, 2> domain{vec{-2, -2}, vec{2, 2}};
  static constexpr vec<size_t, 2>   render_resolution{1000, 1000};
};
//==============================================================================
template <typename Real> struct settings<laminar<Real>> {
  static constexpr std::string_view     name = "laminar";
  static constexpr boundingbox<Real, 2> domain{vec{0, 0}, vec{2, 2}};
  static constexpr vec<size_t, 2> render_resolution{500, 500};
  static constexpr size_t         num_edges = 5;
};
//==============================================================================
template <> struct settings<boussinesq> {
  static constexpr std::string_view name = "boussinesq";
  static constexpr vec<size_t, 2>   render_resolution{500, 1500};
  static constexpr double           eps       = 1e-4;
  //----------------------------------------------------------------------------
  static constexpr boundingbox<double, 2> domain{
      vec{boussinesq::domain.front(0) + eps,
          boussinesq::domain.front(1) + eps},
      vec{boussinesq::domain.back(0) - eps,
          boussinesq::domain.back(1) - eps}};
};
//==============================================================================
template <> struct settings<rbc> {
  using real_type = typename rbc::real_type;
  static constexpr std::string_view name      = "rbc";
  static constexpr real_type           eps       = 1e-2;
  static constexpr size_t           num_edges = 5;
  //----------------------------------------------------------------------------
  static constexpr boundingbox<double, 2> domain{
      vec{rbc::domain.dimension(0).front() + eps,
          rbc::domain.dimension(1).front() + eps},
      vec{rbc::domain.dimension(0).back() - eps,
          rbc::domain.dimension(1).back() - eps}};
  //----------------------------------------------------------------------------
  static constexpr vec<size_t, 2> render_resolution{
      rbc::domain.dimension(0).size() * 4,
      rbc::domain.dimension(1).size() * 4};
};
//==============================================================================
//template <> struct settings<cylinder> {
//  static constexpr std::string_view name = "cylinder";
//  static constexpr grid domain{linspace{0.0, 559.0, cylinder::res[0] / 20},
//                               linspace{0.0, 159.0, cylinder::res[1] / 20}};
//  static constexpr vec<size_t, 2> render_resolution{560 * 2, 160 * 2};
//  static constexpr size_t         num_edges = 5;
//};
//==============================================================================
template <> struct settings<cavity> {
  using real_type = typename cavity::real_type;
  static constexpr std::string_view name = "cavity";
  static constexpr vec<size_t, 2>   render_resolution{
      cavity::domain.dimension(0).size() * 4,
      cavity::domain.dimension(1).size() * 4};
  static constexpr size_t num_edges = 5;
  static constexpr real_type eps       = 1e-4;
  //----------------------------------------------------------------------------
  static constexpr boundingbox<real_type, 2> domain{
      vec{cavity::domain.front(0) + eps, cavity::domain.front(1) + eps},
      vec{cavity::domain.back(0) - eps, cavity::domain.back(1) - eps}};
};
  //==============================================================================
  // template <typename real_type> struct settings<movinggyre<real_type>> {
  //  static constexpr std::string_view name = "movinggyre";
  //  static constexpr real_type           eps  = 1e-4;
  //  static constexpr grid domain{linspace{0, 1, 51}, linspace{-0.5, 0.5, 51}};
  //  static constexpr vec<size_t, 2> render_resolution{1000, 1000};
  //  static constexpr size_t         num_edges = 5;
  //};
  //==============================================================================
  // template <> struct settings<FlappingWing> {
  //  static constexpr std::string_view name = "FlappingWing";
  //  static constexpr real_type           eps  = 1e-5;
  //  static constexpr grid             domain{linspace{0.0 + eps, 24.0 - eps,
  //  26},
  //                                           linspace{0.0 + eps, 24.0 - eps,
  //                                           26}};
  //  static constexpr vec<size_t, 2>   render_resolution{1000, 1000};
  //  static constexpr size_t           num_edges = 5;
  //};
  //==============================================================================
}  // namespace tatooine::steadification
//==============================================================================

#endif
