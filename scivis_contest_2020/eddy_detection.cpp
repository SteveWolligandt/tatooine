#include <tatooine/filesystem.h>
#include <tatooine/ode/vclibs/rungekutta43.h>

#include "ensemble_file_paths.h"
#include "ensemble_member.h"
#include "monitor.h"
#include "positions_in_domain.h"
//==============================================================================
using namespace tatooine;
using namespace tatooine::scivis_contest_2020;
namespace fs = filesystem;
using V      = ensemble_member<interpolation::hermite>;
//==============================================================================
template <typename V>
auto create_grid(V const& v) {
  auto dim0 = v.xc_axis;
  auto dim1 = v.yc_axis;
  auto dim2 = v.z_axis;
  //size(dim0) /= 10;
  //size(dim1) /= 10;
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
auto eddy_detection(std::vector<std::string> const& ensemble_ids) {
  V        v_front{ensemble_id_to_path(ensemble_ids.front())};
  //linspace times{front(v_front.t_axis), back(v_front.t_axis), 3};
   linspace    times{front(v_front.t_axis), back(v_front.t_axis),
                 (size(v_front.t_axis) - 1) * 6 + 1};
  std::vector ensemble_grids(ensemble_ids.size(), create_grid(v_front));
  auto const  P = positions_in_domain(v_front, ensemble_grids.front());
  auto const& xs_in_domain  = P.first;
  auto const& xis_in_domain = P.second;

  std::vector<grid_vertex_property<
      typename decltype(ensemble_grids)::value_type,
      dynamic_multidim_array<double, x_fastest>, interpolation::hermite<double>,
      interpolation::hermite<double>, interpolation::hermite<double>>*>
      mean_eQs, mean_fQs;

  monitor(
      [&] {
        for (size_t i = 0; i < ensemble_grids.size(); ++i) {
          for (auto& z : ensemble_grids[i].dimension<2>()) {
            z *= -0.0025;
          }
          mean_eQs.push_back(
              &ensemble_grids[i]
                   .template add_contiguous_vertex_property<double, x_fastest>(
                       "mean_eulerian_Q"));
          mean_fQs.push_back(
              &ensemble_grids[i]
                   .template add_contiguous_vertex_property<double, x_fastest>(
                       "mean_finite_Q"));
          ensemble_grids[i].iterate_over_vertex_indices(
              [&](auto const... is) {
                mean_eQs[i]->container().at(is...) = 0.0 / 0.0;
                mean_fQs[i]->container().at(is...) = 0.0 / 0.0;
              });
          for (auto const& xi : xis_in_domain) {
            mean_eQs[i]->container().at(xi(0), xi(1), xi(2)) = 0;
            mean_fQs[i]->container().at(xi(0), xi(1), xi(2)) = 0;
          }
        }
      },
      "setting up ensemble grid");

  size_t ensemble_index = 0;
  for (auto& ensemble_id : ensemble_ids) {
    monitor(
        [&] {
          auto const ensemble_path = ensemble_id_to_path(ensemble_id);
          V          v{ensemble_path};
          auto const Jf = diff(v, 1e-8);

          auto  ensemble_member_grid = create_grid(v_front);
          auto& eQ =
              ensemble_member_grid
                  .template add_contiguous_vertex_property<double, x_fastest>(
                      "eulerian_Q");
          auto& fQ =
              ensemble_member_grid
                  .template add_contiguous_vertex_property<double, x_fastest>(
                      "finite_Q_time_0_5_days");

          ensemble_member_grid.iterate_over_vertex_indices(
              [&](auto const... is) {
                eQ.container().at(is...) = 0.0 / 0.0;
                fQ.container().at(is...) = 0.0 / 0.0;
              }, execution_policy::parallel);

          for (auto& z : ensemble_member_grid.dimension<2>()) {
            z *= -0.0025;
          }

          size_t ti = 0;
          for (auto t : times) {
            monitor(
                [&] {
                  fs::path const working_dir =
                      "eddy_detection_" + std::string{ensemble_id} + "/";
                  if (!fs::exists(working_dir)) {
                    fs::create_directory(working_dir);
                  }
                  fs::path out_path = working_dir;
                  out_path += fs::path{"eddy_detection_" +
                                       std::to_string(ti++) + ".vtk"};

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
                            auto const eulerian_Q = (sqr_norm(eulerian_Omega) -
                                                     sqr_norm(eulerian_S)) /
                                                    2;

                            eQ.container().at(xi(0), xi(1), xi(2)) = eulerian_Q;
                            mean_eQs[ensemble_index]->container().at(
                                xi(0), xi(1), xi(2)) +=
                                eulerian_Q / ensemble_ids.size();
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
                            parameterized_line<double, 3, interpolation::linear>
                                pathline;

                            using solver_t =
                                ode::vclibs::rungekutta43<typename V::real_type,
                                                          3>;
                            solver_t solver;
                            auto     evaluator = [&v, &Jf](auto const& y,
                                                       auto const  t) ->
                                typename solver_t::maybe_vec {
                                  if (!v.in_domain(y, t)) {
                                    return ode::vclibs::out_of_domain;
                                  }

                                  return v(y, t);
                                };

                            double const max_ftau = v.t_axis.back() - t;
                            double const min_btau = v.t_axis.front() - t;
                            double const eps      = 1e-6;
                            auto const   ftau =
                                std::min<double>(24 * 5, max_ftau);
                            auto const btau =
                                std::max<double>(-24 * 5, min_btau);
                            auto const t_range = ftau - btau;

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
                                    auto const Q =
                                        (sqr_norm(Omega) - sqr_norm(S)) / 2;
                                    if (pathline.empty()) {
                                      pathline.push_back(y, t, false);
                                      pathline_Q_prop.back() = Q;
                                    }
                                    if (distance(pathline.back_vertex(), y) >
                                        eps) {
                                      pathline.push_back(y, t, false);
                                      pathline_Q_prop.back() = Q;
                                    }
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
                                    auto const Q =
                                        (sqr_norm(Omega) - sqr_norm(S)) / 2;
                                    if (pathline.empty()) {
                                      pathline.push_front(y, t, false);
                                      pathline_Q_prop.front() = Q;
                                    }
                                    if (distance(pathline.front_vertex(), y) >
                                        eps) {
                                      pathline.push_front(y, t, false);
                                      pathline_Q_prop.front() = Q;
                                    }
                                  });
                            }

                            auto Q_time = [&](arithmetic auto const
                                                  threshold) {
                              double Q_time = 0;
                              for (size_t i = 0;
                                   i < pathline.vertices().size() - 1; ++i) {
                                typename decltype(pathline)::vertex_idx vi{i};
                                typename decltype(pathline)::vertex_idx vj{i +
                                                                           1};
                                auto const&                             t0 =
                                    pathline.parameterization_at(i);
                                auto const& t1 =
                                    pathline.parameterization_at(i + 1);
                                auto const& Q0 = pathline_Q_prop[vi];
                                auto const& Q1 = pathline_Q_prop[vj];
                                if (Q0 >= threshold && Q1 >= threshold) {
                                  Q_time += t1 - t0;
                                } else if (Q0 >= threshold && Q1 < threshold) {
                                  auto const t_root = ((t1 - t0) * threshold -
                                                       Q0 * t1 + Q1 * t0) /
                                                      (Q1 - Q0);
                                  Q_time += t_root - t0;
                                } else if (Q0 < threshold && Q1 >= threshold) {
                                  auto const t_root = ((t1 - t0) * threshold -
                                                       Q0 * t1 + Q1 * t0) /
                                                      (Q1 - Q0);
                                  Q_time += t1 - t_root;
                                }
                              }
                              return Q_time / t_range;
                            };

                            auto const Q_time_0                    = Q_time(0);
                            fQ.container().at(xi(0), xi(1), xi(2)) = Q_time_0;
                            mean_fQs[ensemble_index]->container().at(
                                xi(0), xi(1), xi(2)) =
                                Q_time_0 / ensemble_ids.size();
                            ++cnt;
                          }
                        },
                        [&] {
                          return static_cast<double>(cnt) / size(xs_Q_ge_zero);
                        },
                        "lagrangian Q");
                  }

                  monitor(
                      [&] {
                        ensemble_member_grid.write_vtk(out_path);
                      },
                      "writing " + out_path.string());
                },
                "[ensemble " + ensemble_id + "] t = " + std::to_string(t) +
                    ", ti = " + std::to_string(ti));
          }
          ++ensemble_index;
        },
        "ensemble [" + ensemble_id + "]");
  }

  for (size_t i = 0; i < ensemble_grids.size(); ++i) {
    if (!fs::exists("ensemble_data/")) {
      fs::create_directory("ensemble_data/");
    }
    ensemble_grids[i].write_vtk("ensemble_data/ensemble_data_" +
                                std::to_string(i));
  }
}
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // check arguments
  if (argc < 2) {
    throw std::runtime_error{"specify ensemble member ids!"};
  }
  std::vector<std::string> ids;
  ids.reserve(argc - 1);
  for (size_t i = 1; i < static_cast<size_t>(argc); ++i) {
    ids.push_back(argv[i]);
  }
  eddy_detection(ids);
}
