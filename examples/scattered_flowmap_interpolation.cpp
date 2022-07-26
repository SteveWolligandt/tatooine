#include <tatooine/analytical/doublegyre.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/pointset.h>
#include <tatooine/differentiated_flowmap.h>
#include <tatooine/rectilinear_grid.h>

#include <boost/program_options.hpp>
#include <cstdint>
//==============================================================================
using namespace tatooine;
//==============================================================================
struct options_t {
  real_number t_0, t_end;
  std::size_t output_res_x, output_res_y, num_datapoints;
};
using autonomous_particle_flowmap_type =
    AutonomousParticleFlowmapDiscretization<
        2, AutonomousParticle<2>::split_behaviors::five_splits>;
//==============================================================================
auto parse_args(int argc, char const** argv) -> std::optional<options_t>;
//==============================================================================
auto main(int argc, char const** argv) -> int {
  auto const options_opt = parse_args(argc, argv);
  if (!options_opt) {
    return 1;
  }
  auto const                                   options           = *options_opt;
  auto                                         ps                = pointset2{};
  pointset2::typed_vertex_property_type<vec2>* scattered_flowmap = nullptr;
  pointset2::typed_vertex_property_type<mat2>* scattered_flowmap_gradient =
      nullptr;
  scattered_flowmap          = &ps.vec2_vertex_property("flowmap");
  scattered_flowmap_gradient = &ps.mat2_vertex_property("flowmap_gradient");

  auto f = analytical::numerical::doublegyre{};

  auto phi      = flowmap(f);
  phi.use_caching(false);
  auto phi_grad = diff(phi);

  auto const eps               = 1e-3;
  auto       uuid_generator    = std::atomic_uint64_t{};
  auto const initial_particles = autonomous_particle2::particles_from_grid(
      options.t_0,
      rectilinear_grid{linspace{0.0 + eps, 2.0 - eps, 51},
                       linspace{0.0 + eps, 1.0 - eps, 26}},
      uuid_generator);
  {
    auto flowmap_autonomous_particles = autonomous_particle_flowmap_type{
        phi, options.t_end, 0.01, initial_particles, uuid_generator};
    for (auto const& sampler : flowmap_autonomous_particles.samplers()) {
      auto v                            = ps.insert_vertex(sampler.x0(forward));
      scattered_flowmap->at(v)          = sampler.phi(forward);
      scattered_flowmap_gradient->at(v) = sampler.nabla_phi(forward);
    }
    std::cout << "advecting done\n";
    std::cout << flowmap_autonomous_particles.samplers().size() << '\n';
  }

  // auto       rand    = random::uniform{0.0, 1.0, std::mt19937_64{1234}};
  // for (std::size_t i = 0; i < options.num_datapoints; ++i) {
  //   auto v                            = ps.insert_vertex(rand() * 2, rand());
  //   scattered_flowmap->at(v)          = phi(ps[v], options.t_0,
  //   options.t_end); scattered_flowmap_gradient->at(v) = phi_grad(ps[v],
  //   options.t_0, options.t_end);
  // }

  auto gr = uniform_rectilinear_grid2{linspace{0.0, 2.0, options.output_res_x},
                                      linspace{0.0, 1.0, options.output_res_y}};
  //auto nnc_sampler_without_gradients =
  //    ps.natural_neighbor_coordinates_sampler(*scattered_flowmap);
  //std::cout << "nnc done\n";
  //auto nnc_sampler_with_gradients =
  //    ps.natural_neighbor_coordinates_sampler_with_gradients(
  //        *scattered_flowmap, *scattered_flowmap_gradient);
  //std::cout << "nnc grad done\n";
  //auto quartic_harmonic_polynomial = [](auto const squared_distance) {
  //  return squared_distance * squared_distance * gcem::log(squared_distance) /
  //         2;
  //};
  //auto rbf_sampler = ps.radial_basis_functions_sampler(
  //    *scattered_flowmap, quartic_harmonic_polynomial);
  //std::cout << "rbf done\n";
  //auto rbf_grad_sampler = ps.radial_basis_functions_sampler(
  //    *scattered_flowmap, *scattered_flowmap_gradient);
  //std::cout << "rbf grad done\n";
  //auto diff_rbf_grad_sampler = diff(rbf_grad_sampler);
  //gr.sample_to_vertex_property(rbf_sampler, "rbf", execution_policy::parallel);
  //std::cout << "rbf sampling done\n";
  //gr.sample_to_vertex_property(rbf_grad_sampler, "rbf_grad",
  //                             execution_policy::parallel);
  //std::cout << "rbf grad sampling done\n";
  ////gr.sample_to_vertex_property(diff_rbf_grad_sampler, "diff_rbf_grad",
  ////                             execution_policy::parallel);
  ////std::cout << "diff rbf grad sampling done\n";
  //gr.sample_to_vertex_property(nnc_sampler_without_gradients, "nnc",
  //                             execution_policy::parallel);
  //std::cout << "nnc sampling done\n";
  //gr.sample_to_vertex_property(nnc_sampler_with_gradients, "nnc_grad",
  //                             execution_policy::parallel);
  //std::cout << "nnc grad sampling done\n";

  gr.sample_to_vertex_property(
      [&](auto const& x) {
        auto phi2 = phi;
        phi2.use_caching(false);
        return phi2(x, options.t_0, options.t_end);
      },
      "flowmap_rk43", execution_policy::parallel);
  std::cout << "flowmap rk43 sampling done\n";
  gr.sample_to_vertex_property(
      [&](auto const& x) {
        auto phi2 = phi;
        phi2.use_caching(false);
        return diff(phi2)(x, options.t_0, options.t_end);
      },
      "flowmap_rk43_grad", execution_policy::parallel);
  std::cout << "flowmap grad rk43 sampling done\n";
  gr.write("scattered_flowmap_sampler.vtr");
  ps.write("scattered_flowmap_data.vtp");
}
//==============================================================================
auto parse_args(int const argc, char const** argv) -> std::optional<options_t> {
  namespace po = boost::program_options;
  std::size_t output_res_x{}, output_res_y{}, num_datapoints{};
  real_number t_0{}, t_end{};

  auto desc = po::options_description{"Allowed options"};
  auto vm   = po::variables_map{};

  // Declare supported options.
  desc.add_options()("help", "produce help message")(
      "num_datapoints", po::value<std::size_t>(), "number of data points")(
      "t_0", po::value<real_number>(), "t_0")("t_end", po::value<real_number>(),
                                              "t_end")(
      "output_res_x", po::value<std::size_t>(), "set output resolution width")(
      "output_res_y", po::value<std::size_t>(), "set output resolution height");

  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") > 0) {
    std::cout << desc;
    return std::nullopt;
  }
  if (vm.count("t_0") > 0) {
    t_0 = vm["t_0"].as<real_number>();
  } else {
    std::cerr << "--t_0 not specified!\n";
    return std::nullopt;
  }
  if (vm.count("t_end") > 0) {
    t_end = vm["t_end"].as<real_number>();
  } else {
    std::cerr << "--t_end not specified!\n";
    return std::nullopt;
  }
  if (vm.count("output_res_x") > 0) {
    output_res_x = vm["output_res_x"].as<std::size_t>();
  } else {
    std::cerr << "--output_res_x not specified!\n";
    return std::nullopt;
  }
  if (vm.count("output_res_y") > 0) {
    output_res_y = vm["output_res_y"].as<std::size_t>();
  } else {
    std::cerr << "--output_res_y not specified!\n";
    return std::nullopt;
  }
  if (vm.count("num_datapoints") > 0) {
    num_datapoints = vm["num_datapoints"].as<std::size_t>();
  } else {
    std::cerr << "--num_datapoints not specified!\n";
    return std::nullopt;
  }
  return options_t{t_0, t_end, output_res_x, output_res_y, num_datapoints};
}
