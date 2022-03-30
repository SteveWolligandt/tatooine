#include <tatooine/pointset.h>
#include <tatooine/analytical/fields/frankes_test.h>
#include <tatooine/rectilinear_grid.h>

#include <boost/program_options.hpp>
#include <cstdint>
//==============================================================================
using namespace tatooine;
//==============================================================================
enum class type_t : std::uint8_t { franke, unknown };
auto operator>>(std::istream& in, type_t& t) -> std::istream&;
//==============================================================================
struct options_t {
  type_t      type;
  size_t      output_res_x, output_res_y, num_datapoints;
};
//==============================================================================
auto parse_args(int const argc, char const** argv) -> std::optional<options_t>;
//==============================================================================
auto main(int argc, char const** argv) -> int {
  auto const options_opt = parse_args(argc, argv);
  if (!options_opt) {
    return 1;
  }
  auto const options = *options_opt;
  auto       rand    = random::uniform{0.0, 1.0, std::mt19937_64{1234}};
  auto       ps      = pointset2{};
  pointset2::typed_vertex_property_type<real_number>* scalar_prop = nullptr;
  pointset2::typed_vertex_property_type<vec2>* gradient_prop = nullptr;
  switch (options.type) {
    case type_t::franke:
      scalar_prop = &ps.scalar_vertex_property("natural_neighbor_coordinates");
      gradient_prop = &ps.vec2_vertex_property("natural_neighbor_coordinates_gradients");
      break;
    case type_t::unknown:
    default:
      std::cerr << "unknown type.\n";
      return 1;
  }

  auto f = analytical::fields::numerical::frankes_test{};

  for (size_t i = 0; i < options.num_datapoints; ++i) {
    auto v = ps.insert_vertex(rand() * 4 - 2, rand() * 2 - 1);

    switch (options.type) {
      case type_t::franke:
        scalar_prop->at(v) = f(ps[v]);
        gradient_prop->at(v) = diff(f)(ps[v]);
        break;
      case type_t::unknown:
      default:
        std::cerr << "unknown type.\n";
        return 1;
    }
  }

  auto gr = uniform_rectilinear_grid2{linspace{-2.0, 2.0, options.output_res_x},
                                      linspace{-1.0, 1.0, options.output_res_y}};

  auto sample_scalar = [&] {
    auto sampler_without_gradients =
        ps.natural_neighbor_coordinates_sampler(*scalar_prop);
    gr.sample_to_vertex_property(sampler_without_gradients, "without_gradients",
                                 execution_policy::sequential);
    auto sampler_with_gradients =
        ps.natural_neighbor_coordinates_sampler_with_gradients(*scalar_prop,
                                                               *gradient_prop);
    gr.sample_to_vertex_property(sampler_with_gradients, "with_gradients",
                                 execution_policy::sequential);
    gr.sample_to_vertex_property(f, "franke", execution_policy::sequential);
  };
  switch (options.type) {
    case type_t::franke:
      sample_scalar();
      break;
    case type_t::unknown:
    default:
      std::cerr << "unknown type.\n";
      return 1;
  }
  gr.write("natural_neighbor_coordinates_sampler.vtr");
  ps.write("natural_neighbor_coordinates_data.vtp");
}
//==============================================================================
auto parse_args(int const argc, char const** argv) -> std::optional<options_t> {
  namespace po = boost::program_options;
  auto type = type_t::franke;
  size_t output_res_x, output_res_y, num_datapoints;

  auto desc = po::options_description{"Allowed options"};
  auto vm   = po::variables_map{};

  // Declare supported options.
  desc.add_options()("help", "produce help message")(
      "type", po::value<type_t>(), "franke or franke?")(
      "num_datapoints", po::value<size_t>(), "number of data points")(
      "output_res_x", po::value<size_t>(), "set outputresolution width")(
      "output_res_y", po::value<size_t>(), "set outputresolution height");

  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") > 0) {
    std::cout << desc;
    return std::nullopt;
  }
  if (vm.count("type") > 0) {
    type = vm["type"].as<type_t>();
  }
  if (vm.count("output_res_x") > 0) {
    output_res_x = vm["output_res_x"].as<size_t>();
  } else {
    std::cerr << "--output_res_x not specified!\n";
    return std::nullopt;
  }
  if (vm.count("output_res_y") > 0) {
    output_res_y = vm["output_res_y"].as<size_t>();
  } else {
    std::cerr << "--output_res_y not specified!\n";
    return std::nullopt;
  }
  if (vm.count("num_datapoints") > 0) {
    num_datapoints = vm["num_datapoints"].as<size_t>();
  } else {
    std::cerr << "--num_datapoints not specified!\n";
    return std::nullopt;
  }
  return options_t{type, output_res_x, output_res_y, num_datapoints};
}
//------------------------------------------------------------------------------
auto operator>>(std::istream& in, type_t& t) -> std::istream& {
  std::string token;
  in >> token;
  if (token == "franke") {
    t = type_t::franke;
  } else {
    t = type_t::unknown;
    in.setstate(std::ios_base::failbit);
  }
  return in;
}
//==============================================================================
