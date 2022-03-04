#include <tatooine/pointset.h>

#include <boost/program_options.hpp>
#include <cstdint>
//==============================================================================
using namespace tatooine;
//==============================================================================
enum class type_t : std::uint8_t { scalar, vector, unknown };
auto operator>>(std::istream& in, type_t& t) -> std::istream&;
//==============================================================================
struct options_t {
  type_t type;
  real_number epsilon;
  size_t output_res_x, output_res_y, num_datapoints;
};
//==============================================================================
auto parse_args(int const argc, char const** argv) -> std::optional<options_t>;
//==============================================================================
auto main(int argc, char const** argv) -> int {
  auto const options_opt = parse_args(argc, argv);
  if (!options_opt) {
    return 1;
  }
  auto const      options = *options_opt;
  random::uniform rand{-1.0, 1.0, std::mt19937_64{1234}};
  auto            ps                                              = pointset2{};
  pointset2::typed_vertex_property_type<vec3>*        vector_prop = nullptr;
  pointset2::typed_vertex_property_type<real_number>* scalar_prop = nullptr;
  switch (options.type) {
    case type_t::scalar:
      scalar_prop = &ps.scalar_vertex_property("scalar");
      break;
    case type_t::vector:
      vector_prop = &ps.vec3_vertex_property("vector");
      break;
    case type_t::unknown:
    default:
      std::cerr << "unknown type.\n";
      return 1;
  }

  for (size_t i = 0; i < options.num_datapoints; ++i) {
    auto v = ps.insert_vertex(rand(), rand());

    switch (options.type) {
      case type_t::scalar:
        scalar_prop->at(v) = rand() * 10;
        break;
      case type_t::vector:
        vector_prop->at(v)(0) = rand() * 10;
        vector_prop->at(v)(1) = rand() * 10;
        vector_prop->at(v)(2) = rand() * 10;
        break;
      case type_t::unknown:
      default:
        std::cerr << "unknown type.\n";
        return 1;
    }
  }

  auto gr = uniform_rectilinear_grid2{linspace{-1.0, 1.0, options.output_res_x},
                                      linspace{-1.0, 1.0, options.output_res_y}};

  auto sample_scalar = [&] {
    auto sampler =
        ps.radial_basis_functions_sampler(*scalar_prop, options.epsilon);
    gr.sample_to_vertex_property(sampler, "scalar", execution_policy::parallel);
  };
  auto sample_vector = [&] {
    auto sampler =
        ps.radial_basis_functions_sampler(*vector_prop, options.epsilon);
    gr.sample_to_vertex_property(sampler, "vector", execution_policy::parallel);
  };
  switch (options.type) {
    case type_t::scalar:
      sample_scalar();
      break;
    case type_t::vector:
      sample_vector();
      break;
    case type_t::unknown:
    default:
      std::cerr << "unknown type.\n";
      return 1;
  }
  gr.write_vtk("radial_basis_functions_sampler.vtk");
  ps.write_vtk("radial_basis_functions_data.vtk");
}
//==============================================================================
auto parse_args(int const argc, char const** argv) -> std::optional<options_t> {
  namespace po = boost::program_options;
  type_t type;
  real_number epsilon;
  size_t output_res_x, output_res_y, num_datapoints;

  auto desc = po::options_description{"Allowed options"};
  auto vm   = po::variables_map{};

  // Declare supported options.
  desc.add_options()("help", "produce help message")(
      "type", po::value<type_t>(), "scalar or vector")(
      "epsilon", po::value<real_number>(), "epsilion")(
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
  } else {
    std::cerr << "--type not specified!\n";
    return std::nullopt;
  }
  if (vm.count("epsilon") > 0) {
    epsilon = vm["epsilon"].as<real_number>();
  } else {
    std::cerr << "--epsilon not specified!\n";
    return std::nullopt;
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
  return options_t{type, epsilon, output_res_x, output_res_y, num_datapoints};
}
//------------------------------------------------------------------------------
auto operator>>(std::istream& in, type_t& t) -> std::istream& {
  std::string token;
  in >> token;
  if (token == "scalar") {
    t = type_t::scalar;
  } else if (token == "vector") {
    t = type_t::vector;
  } else {
    t = type_t::unknown;
    in.setstate(std::ios_base::failbit);
  }
  return in;
}
//==============================================================================
