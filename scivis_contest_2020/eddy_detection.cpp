#include "ensemble_member.h"

#include <filesystem>
//==============================================================================
using namespace tatooine;
using namespace tatooine::scivis_contest_2020;
using V = ensemble_member<interpolation::hermite>;
//==============================================================================
void print_usage(char** argv);
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // check arguments
  if (argc < 2) {
    print_usage(argv);
    throw std::runtime_error{"specify path to ensemble member!"};
  }
  if (argc < 3) {
    print_usage(argv);
    throw std::runtime_error{"specify time!"};
  }
  // setup grid and properties
  V        v{argv[1]};
  auto     dim0 = v.xc_axis;
  auto     dim1 = v.yc_axis;
  linspace dim2{v.z_axis.front(), v.z_axis.back(), 100};
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};
  // grid g{linspace{v.xc_axis.front(), v.xc_axis.back(), v.xc_axis.size() / 4},
  //       linspace{v.yc_axis.front(), v.yc_axis.back(), v.yc_axis.size() / 4},
  //       linspace{v.z_axis.front(), v.z_axis.back(), 100}};
  g.dimension<0>().pop_front();
  g.dimension<0>().pop_back();
  g.dimension<0>().pop_back();

  g.dimension<1>().pop_front();
  g.dimension<1>().pop_back();
  g.dimension<1>().pop_back();

  g.dimension<2>().pop_front();
  g.dimension<2>().pop_back();
  g.dimension<2>().pop_back();

  auto   Jf = diff(v, 1e-8);
  double t = std::stod(argv[2]);

  auto& imag_prop = g.add_contiguous_vertex_property<double, x_fastest>("imag");
  auto& lambda2_prop =
      g.add_contiguous_vertex_property<double, x_fastest>("lambda2");
  auto& Q_prop = g.add_contiguous_vertex_property<double, x_fastest>("Q");
  auto& vorticity_magnitude_prop =
      g.add_contiguous_vertex_property<double, x_fastest>(
          "vorticity_magnitude");
  auto& divergence_prop =
      g.add_contiguous_vertex_property<double, x_fastest>("divergence");
  auto& lagrangian_vorticity_magnitude_prop =
      g.add_contiguous_vertex_property<double, x_fastest>(
          "lagrangian_vorticity_magnitude");
  auto& lagrangian_lambda2_prop =
      g.add_contiguous_vertex_property<double, x_fastest>("lagrangian_lambda2");
  auto& lagrangian_Q_prop =
      g.add_contiguous_vertex_property<double, x_fastest>("lagrangian_Q");

  auto apply_properties = [&](auto const... is) {
    auto const x = g.vertex_at(is...);
    if (v.in_domain(x, t)) {
      // eulerian properties
      auto const J        = Jf(x, t);
      auto const lambda_J = eigenvalues(J);
      auto const S        = (J + transposed(J)) / 2;
      auto const Omega    = (J - transposed(J)) / 2;
      auto const SS       = S * S;
      auto const OO       = Omega * Omega;
      auto const SSOO     = SS + OO;
      vec const  vorticity{J(2, 1) - J(1, 2), J(0, 2) - J(2, 0),
                          J(1, 0) - J(0, 1)};

      bool is_imag = std::abs(lambda_J(0).imag()) > 1e-10 ||
                     std::abs(lambda_J(1).imag()) > 1e-10 ||
                     std::abs(lambda_J(2).imag()) > 1e-10;
      imag_prop.data_at(is...)    = is_imag ? 1 : 0;
      lambda2_prop.data_at(is...) = eigenvalues_sym(SSOO)(1);
      Q_prop.data_at(is...)       = (sqr_norm(Omega) - sqr_norm(S)) / 2;
      vorticity_magnitude_prop.data_at(is...) = length(vorticity);
      divergence_prop.data_at(is...)          = J(0, 0) + J(1, 1) + J(2, 2);

      // lagrangian properties
      // setup new pathline
      parameterized_line<double, 3, interpolation::linear> pathline;
      auto&                                                vort_pathline_prop =
          pathline.template add_vertex_property<double>("vorticity");
      auto& Q_pathline_prop =
          pathline.template add_vertex_property<double>("Q");
      auto& lambda2_pathline_prop =
          pathline.template add_vertex_property<double>("lambda2");

      // integrate pathline
      ode::vclibs::rungekutta43<V::real_t, 3> solver;
      solver.solve(v, vec{x(0), x(1), x(2)}, t, 10,
                   [&pathline](auto t, const auto& y) {
                     if (pathline.empty()) {
                       pathline.push_back(y, t);
                     } else if (distance(pathline.back_vertex(), y) > 1e-6) {
                       pathline.push_back(y, t);
                     }
                   });
      solver.solve(v, vec{x(0), x(1), x(2)}, t, -10,
                   [&pathline](auto t, const auto& y) {
                     if (pathline.empty()) {
                       pathline.push_back(y, t);
                     } else if (distance(pathline.front_vertex(), y) > 1e-6) {
                       pathline.push_front(y, t);
                     }
                   });

      // for each vertex of the pathline calculate properties
      for (size_t i = 0; i < pathline.num_vertices(); ++i) {
        typename std::decay_t<decltype(pathline)>::vertex_idx vert{i};
        auto const& x = pathline.vertex_at(i);
        auto const& t = pathline.parameterization_at(i);
        if (v.in_domain(x, t)) {
          auto const Jv     = Jf(x, t);
          auto const Sv     = (Jv + transposed(Jv)) / 2;
          auto const Omegav = (Jv - transposed(Jv)) / 2;
          auto const SSv    = Sv * Sv;
          auto const OOv    = Omegav * Omegav;
          auto const SSOOv  = SSv + OOv;

          vec const vort{Jv(2, 1) - Jv(1, 2), Jv(0, 2) - Jv(2, 0),
                         Jv(1, 0) - Jv(0, 1)};
          vort_pathline_prop[vert]    = length(vort);
          Q_pathline_prop[vert]       = (sqr_norm(Omegav) - sqr_norm(Sv)) / 2;
          lambda2_pathline_prop[vert] = eigenvalues_sym(SSOOv)(1);
        } else {
          vort_pathline_prop[vert]    = 0.0 / 0.0;
          Q_pathline_prop[vert]       = 0.0 / 0.0;
          lambda2_pathline_prop[vert] = 0.0 / 0.0;
        }
      }

      // set langragian properties to grid data
      if (!pathline.empty()) {
        lagrangian_vorticity_magnitude_prop.data_at(is...) =
            pathline.integrate_property(vort_pathline_prop);
        lagrangian_lambda2_prop.data_at(is...) =
            pathline.integrate_property(lambda2_pathline_prop);
        lagrangian_Q_prop.data_at(is...) =
            pathline.integrate_property(Q_pathline_prop);
      }
    } else {
      imag_prop.data_at(is...)                           = 0.0 / 0.0;
      lambda2_prop.data_at(is...)                        = 0.0 / 0.0;
      Q_prop.data_at(is...)                              = 0.0 / 0.0;
      vorticity_magnitude_prop.data_at(is...)            = 0.0 / 0.0;
      divergence_prop.data_at(is...)                     = 0.0 / 0.0;
      lagrangian_vorticity_magnitude_prop.data_at(is...) = 0.0 / 0.0;
      lagrangian_lambda2_prop.data_at(is...)             = 0.0 / 0.0;
      lagrangian_Q_prop.data_at(is...)                   = 0.0 / 0.0;
    }
  };
  parallel_for_loop(apply_properties, g.size<0>(), g.size<1>(), g.size<2>());

  namespace fs        = std::filesystem;
  fs::path    p       = argv[1];
  std::string outpath = fs::path{p.filename()}.replace_extension(
      "eddy_detection_" + std::to_string(t) + ".vtk");

  g.dimension<2>().front() *= -0.0025;
  g.dimension<2>().back() *= -0.0025;

  g.write_vtk(outpath);
  // std::string outpath_pathlines = fs::path{p.filename()}.replace_extension(
  //    "eddy_detection_pathlines_" + std::to_string(t) + ".vtk");
  // write_vtk(pathlines, outpath_pathlines);
}
void print_usage(char** argv) {
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " <path/to/ensemble> <time>\n";
}
