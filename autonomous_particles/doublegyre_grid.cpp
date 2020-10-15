#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/grid.h>
#include <tatooine/progress_bars.h>
#include <tatooine/vtk_legacy.h>

#include <boost/program_options.hpp>
#include <execution>
//==============================================================================
using namespace tatooine;
auto ellipsis_vertices(mat<double, 2, 2> const& S, vec<double, 2> const& x0,
                       size_t const resolution, size_t const i0) {
  std::vector<vec<double, 3>> ellipse;
  std::vector<size_t>         indices;
  size_t                      i = 0;
  for (auto t : linspace{0.0, M_PI*2, resolution}) {
    auto const x = S * vec{std::cos(t), std::sin(t)} + x0;
    ellipse.emplace_back(x(0), x(1), 0);
    indices.push_back(i0 + i++);
  }
  return std::pair{std::move(ellipse), std::move(indices)};
}
//------------------------------------------------------------------------------
auto parse_args(int argc, char** argv) {
  namespace po = boost::program_options;

  size_t width, height, num_splits;
  double t0, tau, tau_step;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("width", po::value<size_t>(), "set width")
    ("height", po::value<size_t>(), "set height")
    ("num_splits", po::value<size_t>(), "set number of splits")
    ("t0", po::value<double>(), "set initial time")
    ("tau", po::value<double>(), "set integration length tau")
    ("tau_step", po::value<double>(), "set stepsize for integrator")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::stringstream ss;
    ss << desc;
    throw std::runtime_error{ss.str()};
  }
  if (vm.count("width")) {
    width = vm["width"].as<size_t>();
  } else {
    throw std::runtime_error{"width was not set."};
    }
  if (vm.count("height")) {
    height = vm["height"].as<size_t>();
  } else {
    throw std::runtime_error{"Height was not set."};
  }
  if (vm.count("num_splits")) {
    num_splits = vm["num_splits"].as<size_t>();
  } else {
    throw std::runtime_error{"Number of splits 'num_splits' was not set."};
  }
  if (vm.count("t0")) {
    t0 = vm["t0"].as<double>();
  } else {
    throw std::runtime_error{ "Initial time t0 was not set."};
  }
  if (vm.count("tau")) {
    tau= vm["tau"].as<double>();
  } else {
    throw std::runtime_error{"Integration length tau was not set."};
  }
  if (vm.count("tau_step")) {
    tau_step = vm["tau_step"].as<double>();
  } else {
    throw std::runtime_error{"Step width tau_step was not set."};
  }
  struct {
    size_t width, height, num_splits;
    double t0, tau, tau_step;
  } args{width, height, num_splits, t0, tau, tau_step};
  return args;
}
//------------------------------------------------------------------------------
auto main(int argc, char** argv) -> int {
  auto [width, height, num_splits, t0, tau, tau_step] = parse_args(argc, argv);
  auto calc_particles =
      [tau_step = tau_step, t0 = t0, tau = tau,
       num_splits = num_splits](auto const& p0) -> std::vector<std::decay_t<decltype(p0)>> {
    switch (num_splits) {
      case 2:
        return p0.advect_with_2_splits(tau_step, t0 + tau);
      case 3:
        return p0.advect_with_3_splits(tau_step, t0 + tau);
      case 5:
        return p0.advect_with_5_splits(tau_step, t0 + tau);
      case 7:
        return p0.advect_with_7_splits(tau_step, t0 + tau);
    }
    return {};
  };

  grid g{linspace{0.0, 2.0, width + 1}, linspace{0.0, 1.0, height + 1}};
  g.dimension<0>().pop_front();
  g.dimension<1>().pop_front();
  auto const spacing_x = g.dimension<0>().spacing();
  auto const spacing_y = g.dimension<1>().spacing();
  g.dimension<0>().front() -= spacing_x/2;
  g.dimension<0>().back() -= spacing_x/2;
  g.dimension<1>().front() -= spacing_y/2;
  g.dimension<1>().back() -= spacing_y/2;
  double const r0         = g.dimension<0>().spacing() / 2;


  analytical::fields::numerical::doublegyre v;
  v.set_infinite_domain(true);

  std::vector<vec<double, 3>> initial_ellipses, advected_ellipses, back_calculation_ellipses;
  std::vector<std::vector<size_t>> indices_initial, indices_advected, indices_back_calculation;
  std::mutex                  initial_mutex, advected_mutex;
  size_t     i0_advected = 0, i0_initial = 0, i0_back = 0;

  size_t const ellipsis_res = 100;
  std::atomic_size_t           cnt          = 0;

  progress_bar([&, num_splits = std::ref(num_splits), r0 = std::ref(r0),
                t0 = std::ref(t0), tau_step = std::ref(tau_step),
                tau = std::ref(tau)](auto indicator) {
    std::for_each(std::execution::par_unseq, begin(g.vertices()),
                  end(g.vertices()), [&](auto const& x0) {
                    autonomous_particle p0{v, x0, t0.get(), r0.get()};
                    {
                      auto [vs, is] = ellipsis_vertices(
                          p0.S(), p0.x1(), ellipsis_res, i0_initial);
                      {
                        std::lock_guard lock{initial_mutex};
                        std::move(begin(vs), end(vs),
                                  std::back_inserter(initial_ellipses));
                        indices_initial.push_back(std::move(is));
                        i0_initial += ellipsis_res;
                      }
                    }
                    auto const particles = calc_particles(p0);
                    for (auto const& p : particles) {
                      {
                        auto [vs, is] = ellipsis_vertices(
                            p.S(), p.x1(), ellipsis_res, i0_advected);
                        {
                          std::lock_guard lock{advected_mutex};
                          std::move(begin(vs), end(vs),
                                    std::back_inserter(advected_ellipses));
                          indices_advected.push_back(std::move(is));
                          i0_advected += ellipsis_res;
                        }
                      }

                      {
                        auto sqrS = inv(p.nabla_phi1()) * p.S() * p.S() *
                                    inv(transposed(p.nabla_phi1()));
                        auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
                        eig_vals                  = {std::sqrt(eig_vals(0)),
                                    std::sqrt(eig_vals(1))};
                        auto S =
                            eig_vecs * diag(eig_vals) * transposed(eig_vecs);
                        auto [vs, is] = ellipsis_vertices(
                            S, p.x0(), ellipsis_res, i0_back);
                        {
                          std::lock_guard lock{advected_mutex};
                          std::move(begin(vs), end(vs),
                                    std::back_inserter(back_calculation_ellipses));
                          indices_back_calculation.push_back(std::move(is));
                          i0_back += ellipsis_res;
                        }
                      }
                    }
                    ++cnt;
                    indicator.progress = cnt / double(g.num_vertices());
                  });
  });

  indeterminate_progress_bar([&](auto option) {
    {
      vtk::legacy_file_writer initial_file{"dg_grid_initial.vtk",
                                           vtk::POLYDATA};
      initial_file.write_header();
      option = "Writing initial points";
      initial_file.write_points(initial_ellipses);
      option = "Writing initial lines";
      initial_file.write_lines(indices_initial);
    }

    {
      vtk::legacy_file_writer advection_file{"dg_grid_advected.vtk",
                                             vtk::POLYDATA};
      advection_file.write_header();
      option = "Writing advected points";
      advection_file.write_points(advected_ellipses);
      option = "Writing advected lines";
      advection_file.write_lines(indices_advected);
      advection_file.close();
    }
    {
      vtk::legacy_file_writer back_calc_file{"dg_grid_back_calculation.vtk",
                                             vtk::POLYDATA};
      back_calc_file.write_header();
      option = "Writing back calculated points";
      back_calc_file.write_points(back_calculation_ellipses);
      option = "Writing back calculated lines";
      back_calc_file.write_lines(indices_back_calculation);
      back_calc_file.close();
    }
  });
}
