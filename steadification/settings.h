#ifndef SETTINGS_H
#define SETTINGS_H

#include "datasets.h"

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct settings_t;

//==============================================================================
template <> struct settings_t<rbc> {
  using real_t = typename rbc::real_t;
  static constexpr std::string_view name      = "rbc";
  static constexpr real_t           eps       = 1e-2;
  static constexpr size_t           num_edges = 5;
  //----------------------------------------------------------------------------
  static constexpr grid domain{
      linspace{rbc::domain.dimension(0).front() + eps,
               rbc::domain.dimension(0).back() - eps,
               rbc::domain.dimension(0).size() / 8},
      linspace{rbc::domain.dimension(1).front() + eps,
               rbc::domain.dimension(1).back() - eps,
               rbc::domain.dimension(1).size() / 8}};
  //----------------------------------------------------------------------------
  static constexpr vec<size_t, 2> render_resolution{
      rbc::domain.dimension(0).size() * 4,
      rbc::domain.dimension(1).size() * 4};
};

//==============================================================================
template <typename Real> struct settings_t<numerical::doublegyre<Real>> {
  static constexpr std::string_view name = "doublegyre";
  static constexpr Real             eps  = 1e-4;
  static constexpr boundingbox      domain{vec{eps, eps, 0},
                                      vec{2 - eps, 1 - eps, 10}};
  static constexpr vec<size_t, 2>   render_resolution{1000, 500};
};

//==============================================================================
template <typename Real>
struct settings_t<fixed_time_field<numerical::doublegyre<Real>, Real, 2, 2>> {
  static constexpr std::string_view name = "fixed_time_doublegyre";
  static constexpr Real           eps  = 1e-4;
  static constexpr grid             domain{linspace{eps, 2 - eps, 41},
                               linspace{eps, 1 - eps, 21}};
  static constexpr vec<size_t, 2>   render_resolution{1600, 800};
  static constexpr size_t           num_edges = 5;
};

//==============================================================================
template <typename Real> struct settings_t<numerical::sinuscosinus<Real>> {
  static constexpr std::string_view name = "sinuscosinus";
  static constexpr grid             domain{linspace{-2.0, 2.0, 30},
                               linspace{-2.0, 2.0, 30}};
  static constexpr vec<size_t, 2>   render_resolution{1000, 1000};
  static constexpr size_t           num_edges = 5;
};

//==============================================================================
template <typename Real> struct settings_t<laminar<Real>> {
  static constexpr std::string_view name = "laminar";
  static constexpr grid domain{linspace{0.0, 2.0, 20}, linspace{0, 2, 20}};
  static constexpr vec<size_t, 2> render_resolution{500, 500};
  static constexpr size_t         num_edges = 5;
};

//==============================================================================
//template <> struct settings_t<cylinder> {
//  static constexpr std::string_view name = "cylinder";
//  static constexpr grid domain{linspace{0.0, 559.0, cylinder::res[0] / 20},
//                               linspace{0.0, 159.0, cylinder::res[1] / 20}};
//  static constexpr vec<size_t, 2> render_resolution{560 * 2, 160 * 2};
//  static constexpr size_t         num_edges = 5;
//};

//==============================================================================
template <> struct settings_t<boussinesq> {
  static constexpr std::string_view name = "boussinesq";
  static constexpr vec<size_t, 2>   render_resolution{500, 1500};
  static constexpr size_t           num_edges = 5;
  static constexpr double           eps       = 1e-4;
  //----------------------------------------------------------------------------
  static constexpr grid domain{
      linspace{
          boussinesq::domain.dimension(0).front() + 1.0 / boussinesq::res(0),
          boussinesq::domain.dimension(0).back() - 1.0 / boussinesq::res(0),
          18},
      linspace{
          boussinesq::domain.dimension(1).front() + 1.0 / boussinesq::res(1),
          boussinesq::domain.dimension(1).back() - 1.0 / boussinesq::res(1),
          18 * 3}};
};

//==============================================================================
template <> struct settings_t<cavity> {
  using real_t = typename rbc::real_t;
  static constexpr std::string_view name = "cavity";
  static constexpr vec<size_t, 2>   render_resolution{
      cavity::domain.dimension(0).size() * 5,
      cavity::domain.dimension(1).size() * 5};
  static constexpr size_t num_edges = 5;
  static constexpr real_t eps       = 1e-4;
  //----------------------------------------------------------------------------
  static constexpr grid domain{
      linspace{cavity::domain.dimension(0).front() + eps,
               cavity::domain.dimension(0).back() - eps,
               cavity::domain.dimension(0).size() / 5},
      linspace{cavity::domain.dimension(1).front() + eps,
               cavity::domain.dimension(1).back() - eps,
               cavity::domain.dimension(1).size() / 5}};
};

//==============================================================================
//template <typename real_t> struct settings_t<movinggyre<real_t>> {
//  static constexpr std::string_view name = "movinggyre";
//  static constexpr real_t           eps  = 1e-4;
//  static constexpr grid domain{linspace{0, 1, 51}, linspace{-0.5, 0.5, 51}};
//  static constexpr vec<size_t, 2> render_resolution{1000, 1000};
//  static constexpr size_t         num_edges = 5;
//};

//==============================================================================
// template <> struct settings_t<FlappingWing> {
//  static constexpr std::string_view name = "FlappingWing";
//  static constexpr real_t           eps  = 1e-5;
//  static constexpr grid             domain{linspace{0.0 + eps, 24.0 - eps,
//  26},
//                                           linspace{0.0 + eps, 24.0 - eps,
//                                           26}};
//  static constexpr vec<size_t, 2>   render_resolution{1000, 1000};
//  static constexpr size_t           num_edges = 5;
//};
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
