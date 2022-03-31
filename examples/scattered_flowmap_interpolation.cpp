#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/pointset.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/numerical_flowmap.h>

#include <boost/program_options.hpp>
#include <cstdint>
//==============================================================================
using namespace tatooine;
//==============================================================================
struct options_t {
  real_number t_0, t_end;
  std::size_t output_res_x, output_res_y, num_datapoints;
};
//==============================================================================
auto parse_args(int argc, char const** argv) -> std::optional<options_t>;
//==============================================================================
auto main(int argc, char const** argv) -> int {
  auto const options_opt = parse_args(argc, argv);
  if (!options_opt) {
    return 1;
  }
  auto const options = *options_opt;
  auto       rand    = random::uniform{0.0, 1.0, std::mt19937_64{1234}};
  auto       ps      = pointset2{};
  pointset2::typed_vertex_property_type<vec2>* scattered_flowmap = nullptr;
  pointset2::typed_vertex_property_type<mat2>* scattered_flowmap_gradient =
      nullptr;
  scattered_flowmap          = &ps.vec2_vertex_property("flowmap");
  scattered_flowmap_gradient = &ps.mat2_vertex_property("flowmap_gradient");

  auto f = analytical::fields::numerical::doublegyre{};

  auto phi      = flowmap(f);
  auto phi_grad = diff(phi);
  for (std::size_t i = 0; i < options.num_datapoints; ++i) {
    auto v                            = ps.insert_vertex(rand() * 2, rand());
    scattered_flowmap->at(v)          = phi(ps[v], options.t_0, options.t_end);
    scattered_flowmap_gradient->at(v) = phi_grad(ps[v], options.t_0, options.t_end);
    break;
  }

  auto gr = uniform_rectilinear_grid2{linspace{0.0, 2.0, options.output_res_x},
                                      linspace{0.0, 1.0, options.output_res_y}};

  auto nnc_sampler_without_gradients =
      ps.natural_neighbor_coordinates_sampler(*scattered_flowmap);
  gr.sample_to_vertex_property(nnc_sampler_without_gradients,
                               "without_gradients",
                               execution_policy::sequential);
  auto nnc_sampler_with_gradients =
      ps.natural_neighbor_coordinates_sampler_with_gradients(
          *scattered_flowmap, *scattered_flowmap_gradient);
  auto rbf_sampler = ps.radial_basis_functions_sampler_with_polynomial(
      *scattered_flowmap, [](auto const squared_distance) {
        return squared_distance * squared_distance *
               gcem::log(squared_distance) / 2;
      });
  gr.sample_to_vertex_property(rbf_sampler, "rbf", execution_policy::parallel);
  gr.sample_to_vertex_property(nnc_sampler_with_gradients, "with_gradients",
                               execution_policy::sequential);
  gr.sample_to_vertex_property(f, "doublegyre", execution_policy::sequential);
  gr.write("natural_neighbor_coordinates_sampler.vtr");
  ps.write("natural_neighbor_coordinates_data.vtp");
}
//==============================================================================
auto parse_args(int const argc, char const** argv) -> std::optional<options_t> {
  namespace po = boost::program_options;
  std::size_t output_res_x, output_res_y, num_datapoints;
  real_number t_0, t_end;

  auto desc = po::options_description{"Allowed options"};
  auto vm   = po::variables_map{};

  // Declare supported options.
  desc.add_options()("help", "produce help message")(
      "num_datapoints", po::value<std::size_t>(), "number of data points")(
      "t_0", po::value<real_number>(), "t_0")("t_end", po::value<real_number>(),
                                              "t_end")(
      "output_res_x", po::value<std::size_t>(), "set outputresolution width")(
      "output_res_y", po::value<std::size_t>(), "set outputresolution height");

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
