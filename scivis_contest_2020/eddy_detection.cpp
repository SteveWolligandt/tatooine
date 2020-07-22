#include "ensemble_member.h"
#include "integrate_pathline.h"
#include "positions_in_domain.h"
#include "ensemble_file_paths.h"

#include <filesystem>
//==============================================================================
using namespace tatooine;
using namespace tatooine::scivis_contest_2020;
using V = ensemble_member<interpolation::hermite>;
//==============================================================================
void print_usage(char** argv);
//------------------------------------------------------------------------------
template <typename V>
auto create_grid(V const& v) {
  auto     dim0 = v.xc_axis;
  auto     dim1 = v.yc_axis;
  auto dim2 = v.z_axis;
  //linspace dim2{v.z_axis.front(), v.z_axis.back(), 100};
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};
  return g;
}
template <typename V, typename Grid, typename Prop, typename Xs, typename XIs>
auto eddy_detection(V const& v, real_number auto t, Grid& g,
                    Prop& lagrangian_Q_prop, Xs const& xs, XIs const& xis) {
  // setup grid and properties
  auto Jf = diff(v, 1e-8);

  g.parallel_loop_over_vertex_indices([&](auto const... is) {
    // imag_prop.data_at(is...)                           = 0.0 / 0.0;
    // lambda2_prop.data_at(is...)                        = 0.0 / 0.0;
    // Q_prop.data_at(is...)                              = 0.0 / 0.0;
    // vorticity_magnitude_prop.data_at(is...)            = 0.0 / 0.0;
    // divergence_prop.data_at(is...)                     = 0.0 / 0.0;
    // lagrangian_vorticity_magnitude_prop.data_at(is...) = 0.0 / 0.0;
    // lagrangian_lambda2_prop.data_at(is...)             = 0.0 / 0.0;
    lagrangian_Q_prop.data_at(is...) = 0.0 / 0.0;
  });
  std::atomic_size_t cnt = 0;
  bool done = false;
  std::thread monitor{[&g, &done, &cnt, &xs]{
    while (!done) {
      std::cerr << static_cast<double>(cnt) / size(xs) * 100 << "  %        \r";
      std::this_thread::sleep_for(std::chrono::milliseconds{200});
    }
  }};
  #pragma omp parallel for schedule(dynamic, 2)
  for (size_t i = 0; i < size(xs); ++i) {
    //// eulerian properties
    // auto const J        = Jf(x, t);
    // auto const lambda_J = eigenvalues(J);
    // auto const S        = (J + transposed(J)) / 2;
    // auto const Omega    = (J - transposed(J)) / 2;
    // auto const SS       = S * S;
    // auto const OO       = Omega * Omega;
    // auto const SSOO     = SS + OO;
    // vec const  vorticity{J(2, 1) - J(1, 2), J(0, 2) - J(2, 0),
    //                    J(1, 0) - J(0, 1)};
    //
    // bool is_imag = std::abs(lambda_J(0).imag()) > 1e-10 ||
    //               std::abs(lambda_J(1).imag()) > 1e-10 ||
    //               std::abs(lambda_J(2).imag()) > 1e-10;
    // imag_prop.data_at(is...)    = is_imag ? 1 : 0;
    // lambda2_prop.data_at(is...) = eigenvalues_sym(SSOO)(1);
    // Q_prop.data_at(is...)       = (sqr_norm(Omega) - sqr_norm(S)) / 2;
    // vorticity_magnitude_prop.data_at(is...) = length(vorticity);
    // divergence_prop.data_at(is...)          = J(0, 0) + J(1, 1) + J(2,
    // 2);

    // lagrangian properties
    auto const& x        = xs[i];
    auto        pathline = integrate_pathline(v, x, t);
    // auto& vort_pathline_prop =
    //    pathline.template add_vertex_property<double>("vorticity");
    // auto& lambda2_pathline_prop =
    //    pathline.template add_vertex_property<double>("lambda2");
    auto& Q_pathline_prop = pathline.template add_vertex_property<double>("Q");
    // for each vertex of the pathline calculate properties
    for (size_t i = 0; i < pathline.num_vertices(); ++i) {
      typename std::decay_t<decltype(pathline)>::vertex_idx vert{i};
      auto const& x   = pathline.vertex_at(i);
      auto const& t   = pathline.parameterization_at(i);
      auto const  eps = 1e-4;
      if (v.in_domain(x, t) && v.in_domain(vec{x(0) + eps, x(1), x(2)}, t) &&
          v.in_domain(vec{x(0) - eps, x(1), x(2)}, t) &&
          v.in_domain(vec{x(0), x(1) + eps, x(2)}, t) &&
          v.in_domain(vec{x(0), x(1) - eps, x(2)}, t) &&
          v.in_domain(vec{x(0), x(1), x(2) + eps}, t) &&
          v.in_domain(vec{x(0), x(1), x(2) - eps}, t)) {
        auto const Jv     = Jf(x, t);
        auto const Sv     = (Jv + transposed(Jv)) / 2;
        auto const Omegav = (Jv - transposed(Jv)) / 2;
        // auto const SSv    = Sv * Sv;
        // auto const OOv    = Omegav * Omegav;
        // auto const SSOOv  = SSv + OOv;

        // vec const vort{Jv(2, 1) - Jv(1, 2), Jv(0, 2) - Jv(2, 0),
        //               Jv(1, 0) - Jv(0, 1)};
        // vort_pathline_prop[vert]    = length(vort);
        // lambda2_pathline_prop[vert] = eigenvalues_sym(SSOOv)(1);
        Q_pathline_prop[vert] = (sqr_norm(Omegav) - sqr_norm(Sv)) / 2;
      } else {
        // vort_pathline_prop[vert]    = 0.0 / 0.0;
        // lambda2_pathline_prop[vert] = 0.0 / 0.0;
        Q_pathline_prop[vert] = 0.0 / 0.0;
      }
    }

    // set langragian properties to grid data
    if (!pathline.empty()) {
      // lagrangian_vorticity_magnitude_prop.data_at(is...) =
      //    pathline.integrate_property(vort_pathline_prop);
      // lagrangian_lambda2_prop.data_at(is...) =
      //    pathline.integrate_property(lambda2_pathline_prop);
      lagrangian_Q_prop.data_at(xis[i](0), xis[i](1), xis[i](2)) =
          pathline.integrate_property(Q_pathline_prop);
    }
    ++cnt;
  }
  monitor.join();
}
auto eddy_detection(std::string const& ensemble_id) {
  auto const ensemble_path = [&] {
    if (std::string{ensemble_id} == "MEAN" ||
        std::string{ensemble_id} == "Mean" ||
        std::string{ensemble_id} == "mean") {
      return tatooine::scivis_contest_2020::mean_file_path;
    }
    return tatooine::scivis_contest_2020::ensemble_file_paths[std::stoi(
        ensemble_id)];
  }();
  V v{ensemble_path};
  auto g  = create_grid(v);
  // auto& imag_prop = g.add_contiguous_vertex_property<double,
  // x_fastest>("imag"); auto& lambda2_prop =
  //    g.add_contiguous_vertex_property<double, x_fastest>("lambda2");
  // auto& Q_prop = g.add_contiguous_vertex_property<double, x_fastest>("Q");
  // auto& vorticity_magnitude_prop =
  //    g.add_contiguous_vertex_property<double, x_fastest>(
  //        "vorticity_magnitude");
  // auto& divergence_prop =
  //    g.add_contiguous_vertex_property<double, x_fastest>("divergence");
  // auto& lagrangian_vorticity_magnitude_prop =
  //    g.add_contiguous_vertex_property<double, x_fastest>(
  //        "lagrangian_vorticity_magnitude");
  // auto& lagrangian_lambda2_prop =
  //    g.add_contiguous_vertex_property<double,
  //    x_fastest>("lagrangian_lambda2");
  auto& lagrangian_Q_prop =
      g.template add_contiguous_vertex_property<double, x_fastest>("lagrangian_Q");

  auto [xs, xis] = positions_in_domain(v, g);

  size_t ti = 0;
  for (auto t : v.t_axis) {
    std::cerr << "processing time " << t << ", at index " << ti << " ...\n";
    namespace fs = std::filesystem;
    fs::path p   = "eddy_detection " + std::string{ensemble_id} + "/";
    if (!fs::exists(p)) { fs::create_directory(p); }
    p += "eddy_detection_";
    p += ti++;
    p += ".vtk";

    if (!fs::exists(p)) {
      eddy_detection(v, t, g, lagrangian_Q_prop, xs, xis);
      g.dimension<2>().front() *= -0.0025;
      g.dimension<2>().back() *= -0.0025;

      g.write_vtk(p);
      g.dimension<2>().front() /= -0.0025;
      g.dimension<2>().back() /= -0.0025;
    }
  }
}

int main(int argc, char** argv) {
  // check arguments
  if (argc < 2) {
    print_usage(argv);
    throw std::runtime_error{"specify path to ensemble member!"};
  }
  eddy_detection(argv[1]);
}
void print_usage(char** argv) {
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " <path/to/ensemble>\n";
}
