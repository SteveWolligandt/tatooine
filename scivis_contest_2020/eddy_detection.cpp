#include <filesystem>

#include "ensemble_file_paths.h"
#include "ensemble_member.h"
#include "integrate_pathline.h"
#include "monitor.h"
#include "positions_in_domain.h"
//==============================================================================
using namespace tatooine;
using namespace tatooine::scivis_contest_2020;
namespace fs = std::filesystem;
using V      = ensemble_member<interpolation::hermite>;
//==============================================================================
void print_usage(char** argv);
//------------------------------------------------------------------------------
template <typename V>
auto create_grid(V const& v) {
  auto dim0 = v.xc_axis;
  auto dim1 = v.yc_axis;
  auto dim2 = v.z_axis;
  // size(dim0) /= 5;
  // size(dim1) /= 5;
  // linspace dim2{v.z_axis.front(), v.z_axis.back(), 100};
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};
  return g;
}
//------------------------------------------------------------------------------
auto ensemble_id_to_path(std::string const& ensemble_id) {
  if (std::string{ensemble_id} == "MEAN" ||
      std::string{ensemble_id} == "Mean" ||
      std::string{ensemble_id} == "mean") {
    return tatooine::scivis_contest_2020::mean_file_path;
  }
  return tatooine::scivis_contest_2020::ensemble_file_paths[std::stoi(
      ensemble_id)];
}
//------------------------------------------------------------------------------
auto eddy_detection(std::string const& ensemble_id) {
  auto const ensemble_path = ensemble_id_to_path(ensemble_id);
  V          v{ensemble_path};
  auto const Jf = diff(v, 1e-8);

  auto  g  = create_grid(v);
  auto& eQ = g.template add_contiguous_vertex_property<double, x_fastest>(
      "eulerian_Q");
  auto& fQ = g.template add_contiguous_vertex_property<double, x_fastest>(
      "finite_Q_time_0_5_days");

  g.parallel_loop_over_vertex_indices([&](auto const... is) {
    eQ.container().at(is...) = 0.0 / 0.0;
    fQ.container().at(is...) = 0.0 / 0.0;
  });

  auto const  P             = positions_in_domain(v, g);
  auto const& xs_in_domain  = P.first;
  auto const& xis_in_domain = P.second;
  for (auto& z : g.dimension<2>()) {
    z *= -0.0025;
  }

  size_t   ti = 0;
  linspace times{front(v.t_axis), back(v.t_axis),
                 (size(v.t_axis) - 1) * 12 + 1};
  std::cerr << times << '\n';
  for (auto t : times) {
    std::cerr << "processing time " << t << ", at index " << ti << " ...\n";
    fs::path p = "eddy_detection_" + std::string{ensemble_id} + "/";
    if (!fs::exists(p)) {
      fs::create_directory(p);
    }
    p += "eddy_detection_" + std::to_string(ti++) + ".vtk";

    if (!fs::exists(p)) {
      std::decay_t<decltype(xs_in_domain)>  xs_Q_ge_zero;
      std::decay_t<decltype(xis_in_domain)> xis_Q_ge_zero;
      xs_Q_ge_zero.reserve(xs_in_domain.size() / 4);
      xis_Q_ge_zero.reserve(xis_in_domain.size() / 4);
      std::mutex Q_ge_zero_mutex;
      {
        std::atomic_size_t cnt = 0;
        monitor(
            [&] {
#pragma omp parallel for
              for (size_t i = 0; i < size(xs_in_domain); ++i) {
                auto const& x          = xs_in_domain[i];
                auto const& xi         = xis_in_domain[i];
                auto const  eulerian_J = Jf(x, t);
                auto const  eulerian_S =
                    (eulerian_J + transposed(eulerian_J)) / 2;
                auto const eulerian_Omega =
                    (eulerian_J - transposed(eulerian_J)) / 2;
                auto const eulerian_Q =
                    (sqr_norm(eulerian_Omega) - sqr_norm(eulerian_S)) / 2;

                eQ.container().at(xi(0), xi(1), xi(2)) = eulerian_Q;
                if (eulerian_Q > 0) {
                  std::lock_guard lock{Q_ge_zero_mutex};
                  xs_Q_ge_zero.push_back(x);
                  xis_Q_ge_zero.push_back(xi);
                } else {
                  fQ.container().at(xi(0), xi(1), xi(2)) = 0;
                }
                ++cnt;
              }
            },
            [&] {
              return static_cast<double>(cnt) / size(xs_in_domain);
            },
            "Calculation of eulerian Q");
      }
      {
        std::atomic_size_t cnt = 0;
        monitor(
            [&] {
#pragma omp parallel for
              for (size_t i = 0; i < size(xs_Q_ge_zero); ++i) {
                auto const& x  = xs_Q_ge_zero[i];
                auto const& xi = xis_Q_ge_zero[i];
                parameterized_line<double, 3, interpolation::linear> pathline;

                using solver_t =
                    ode::vclibs::rungekutta43<typename V::real_t, 3>;
                solver_t solver;
                auto     evaluator = [&v, &Jf](auto const& y, auto const t) ->
                    typename solver_t::maybe_vec {
                      if (!v.in_domain(y, t)) {
                        return ode::vclibs::out_of_domain;
                      }

                      //auto const J     = Jf(y, t);
                      //auto const S     = (J + transposed(J)) / 2;
                      //auto const Omega = (J - transposed(J)) / 2;
                      //auto const Q     = (sqr_norm(Omega) - sqr_norm(S)) / 2;
                      //if (Q < 0) {
                      //  return ode::vclibs::out_of_domain;
                      //}

                      return v(y, t);
                    };

                double const max_ftau = v.t_axis.back() - t;
                double const min_btau = v.t_axis.front() - t;
                double const eps      = 1e-6;
                auto const   ftau     = std::min<double>(24 * 5, max_ftau);
                auto const   btau     = std::max<double>(-24 * 5, min_btau);
                auto const   t_range  = ftau - btau;

                auto& pathline_Q_prop =
                    pathline.add_vertex_property<double>("Q");
                if (ftau > 0) {
                  solver.solve(
                      evaluator, x, t, ftau,
                      [&pathline, &pathline_Q_prop, &Jf, eps](
                          const vec<double, 3>& y, double t) {
                        auto const J     = Jf(y, t);
                        auto const S     = (J + transposed(J)) / 2;
                        auto const Omega = (J - transposed(J)) / 2;
                        auto const Q     = (sqr_norm(Omega) - sqr_norm(S)) / 2;
                        if (Q < 0) {
                          return false;
                        }

                        if (pathline.empty()) {
                          pathline.push_back(y, t, false);
                          pathline_Q_prop.back() = Q;
                          return true;
                        }
                        if (distance(pathline.back_vertex(), y) > eps) {
                          pathline.push_back(y, t, false);
                          pathline_Q_prop.back() = Q;
                          return true;
                        }
                        return false;
                      });
                }
                if (btau < 0) {
                  solver.solve(
                      evaluator, x, t, btau,
                      [&pathline, &pathline_Q_prop, &Jf, eps](
                          const vec<double, 3>& y, double t) {
                        auto const J     = Jf(y, t);
                        auto const S     = (J + transposed(J)) / 2;
                        auto const Omega = (J - transposed(J)) / 2;
                        auto const Q     = (sqr_norm(Omega) - sqr_norm(S)) / 2;
                        if (Q < 0) {
                          return false;
                        }
                        if (pathline.empty()) {
                          pathline.push_front(y, t, false);
                          pathline_Q_prop.front() = Q;
                          return true;
                        }
                        if (distance(pathline.front_vertex(), y) > eps) {
                          pathline.push_front(y, t, false);
                          pathline_Q_prop.front() = Q;
                          return true;
                        }
                        return false;
                      });
                }

                auto Q_time = [&](real_number auto const threshold) {
                  double Q_time = 0;
                  for (size_t i = 0; i < pathline.num_vertices() - 1; ++i) {
                    typename decltype(pathline)::vertex_idx vi{i};
                    typename decltype(pathline)::vertex_idx vj{i + 1};
                    auto const& t0 = pathline.parameterization_at(i);
                    auto const& t1 = pathline.parameterization_at(i + 1);
                    auto const& Q0 = pathline_Q_prop[vi];
                    auto const& Q1 = pathline_Q_prop[vj];
                    if (Q0 >= threshold && Q1 >= threshold) {
                      Q_time += t1 - t0;
                    } else if (Q0 >= threshold && Q1 < threshold) {
                      auto const t_root =
                          ((t1 - t0) * threshold - Q0 * t1 + Q1 * t0) /
                          (Q1 - Q0);
                      Q_time += t_root - t0;
                    } else if (Q0 < threshold && Q1 >= threshold) {
                      auto const t_root =
                          ((t1 - t0) * threshold - Q0 * t1 + Q1 * t0) /
                          (Q1 - Q0);
                      Q_time += t1 - t_root;
                    }
                  }
                  return Q_time / t_range;
                };

                fQ.container().at(xi(0), xi(1), xi(2)) = 0;
                Q_time(0);
                ++cnt;
              }
            },
            [&] {
              return static_cast<double>(cnt) / size(xs_Q_ge_zero);
            },
            "lagrangian Q");
      }

      g.write_vtk(p);
    }
  }
}
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // check arguments
  if (argc < 2) {
    print_usage(argv);
    throw std::runtime_error{"specify path to ensemble member!"};
  }
  eddy_detection(argv[1]);
}
//------------------------------------------------------------------------------
void print_usage(char** argv) {
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " <path/to/ensemble>\n";
}
