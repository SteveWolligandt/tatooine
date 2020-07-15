#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>

#include <filesystem>
#include <mutex>
//==============================================================================
using namespace tatooine;
using V = fields::scivis_contest_2020_ensemble_member;
//==============================================================================
std::mutex mutex;
//==============================================================================
void print_usage(char** argv);
//------------------------------------------------------------------------------
int  main(int argc, char** argv) {
  // check arguments
  if (argc < 2) {
    print_usage(argv);
    throw std::runtime_error{"specify path to ensemble member!"};
  }
  if (argc < 3) {
    print_usage(argv);
    throw std::runtime_error{"specify time!"};
  }
  V v{argv[1]};
  auto Jf               = diff(v, 1e-4);
  double t = std::stod(argv[2]);

  // setup grid and properties
  grid   g{v.xc_axis, v.yc_axis,
         linspace{v.z_axis.front(), v.z_axis.back(), 100}};
  g.dimension<0>().pop_front();
  g.dimension<1>().pop_front();
  g.dimension<2>().pop_front();
  g.dimension<0>().pop_back();
  g.dimension<1>().pop_back();
  g.dimension<0>().pop_back();
  g.dimension<1>().pop_back();
  g.dimension<2>().pop_back();

  auto& imag_prop =
      g.add_contiguous_vertex_property<double, x_fastest>(
          "imag");
  auto& lambda2_prop =
      g.add_contiguous_vertex_property<double, x_fastest>(
          "lambda2");
  auto& Q_prop =
      g.add_contiguous_vertex_property<double, x_fastest>(
          "Q");
  auto& vorticity_magnitude_prop =
      g.add_contiguous_vertex_property<double, x_fastest>(
          "vorticity_magnitude");
  auto& lagrangian_vorticity_magnitude_prop =
      g.add_contiguous_vertex_property<double, x_fastest>(
          "lagrangian_vorticity_magnitude");
  auto& lagrangian_lambda2_prop =
      g.add_contiguous_vertex_property<double, x_fastest>(
          "lagrangian_lambda2");
  auto& lagrangian_Q_prop =
      g.add_contiguous_vertex_property<double, x_fastest>(
          "lagrangian_Q");

  // setup pathlines data
  std::vector<parameterized_line<double, 3, interpolation::linear>> pathlines;
  pathlines.reserve(g.num_vertices());
  auto add_pathline = [&]() -> decltype(auto) {
    std::lock_guard lock{mutex};
    return pathlines.emplace_back();
  };

  auto apply_properties = [&](auto const... is) {
    vec const x = g.vertex_at(is...);
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

      // lagrangian properties
      // setup new pathline
      auto& pathline = add_pathline();
      auto& vort_pathline_prop =
          pathline.template add_vertex_property<double>("vorticity");
      auto& Q_pathline_prop =
          pathline.template add_vertex_property<double>("Q");
      auto& lambda2_pathline_prop =
          pathline.template add_vertex_property<double>("lambda2");

      // integrate pathline
      ode::vclibs::rungekutta43<V::real_t, 3> solver;
      solver.solve(
          v, vec{x(0), x(1), x(2)}, t, 100,
          [&pathline](auto t, const auto& y) { pathline.push_back(y, t); });
      solver.solve(
          v, vec{x(0), x(1), x(2)}, t, -100,
          [&pathline](auto t, const auto& y) { pathline.push_front(y, t); });

      // for each vertex of the pathline calculate properties
      for (size_t i = 0; i < pathline.num_vertices(); ++i) {
        typename std::decay_t<decltype(pathline)>::vertex_idx v{i};
        auto const& x      = pathline.vertex_at(v);
        auto const& t      = pathline.parameterization_at(i);
        auto const  Jv     = Jf(x, t);
        auto const  Sv     = (Jv + transposed(Jv)) / 2;
        auto const  Omegav = (Jv - transposed(Jv)) / 2;
        auto const  SSv    = Sv * Sv;
        auto const  OOv    = Omegav * Omegav;
        auto const  SSOOv  = SSv + OOv;

        vec const vort{Jv(2, 1) - Jv(1, 2), Jv(0, 2) - Jv(2, 0),
                       Jv(1, 0) - Jv(0, 1)};
        vort_pathline_prop[v]    = length(vort);
        Q_pathline_prop[v]       = (sqr_norm(Omegav) - sqr_norm(Sv)) / 2;
        lambda2_pathline_prop[v] = eigenvalues_sym(SSOOv)(1);
      }

      // set langragian properties to grid data
      lagrangian_vorticity_magnitude_prop.data_at(is...) =
          pathline.integrate_property(vort_pathline_prop);
      lagrangian_lambda2_prop.data_at(is...) =
          pathline.integrate_property(lambda2_pathline_prop);
      lagrangian_Q_prop.data_at(is...) =
          pathline.integrate_property(Q_pathline_prop);
    } else {
      imag_prop.data_at(is...)                           = 0.0 / 0.0;
      lambda2_prop.data_at(is...)                        = 0.0 / 0.0;
      Q_prop.data_at(is...)                              = 0.0 / 0.0;
      vorticity_magnitude_prop.data_at(is...)            = 0.0 / 0.0;
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
  g.write_vtk(outpath);
  std::string outpath_pathlines = fs::path{p.filename()}.replace_extension(
      "eddy_detection_pathlines_" + std::to_string(t) + ".vtk");
  write_vtk(pathlines, outpath_pathlines);
}
void print_usage(char** argv) {
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " <path/to/ensemble> <time>\n";
}
