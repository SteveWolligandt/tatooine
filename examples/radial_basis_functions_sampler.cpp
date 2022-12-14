#include <tatooine/pointset.h>
#include <tatooine/analytical/frankes_test.h>
#include <tatooine/rectilinear_grid.h>

#include <boost/program_options.hpp>
#include <cstdint>
//==============================================================================
using namespace tatooine;
//==============================================================================
enum class type_t : std::uint8_t { franke, unknown };
auto operator>>(std::istream& in, type_t& t) -> std::istream&;
enum class kernel_t : std::uint8_t {
  linear,
  cubic,
  thin_plate_spline,
  gaussian,
  unknown
};
auto operator>>(std::istream& in, kernel_t& t) -> std::istream&;
//==============================================================================
struct options_t {
  type_t      type;
  kernel_t    kernel;
  real_number epsilon;
  std::size_t output_res_x, output_res_y, num_datapoints;
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
  pointset2::typed_vertex_property_type<vec2>*        vector_prop = nullptr;
  pointset2::typed_vertex_property_type<real_number>* scalar_prop = nullptr;
  switch (options.type) {
    case type_t::franke:
      scalar_prop = &ps.scalar_vertex_property("franke");
      vector_prop = &ps.vec2_vertex_property("franke_gradient");
      break;
    //case type_t::vector:
    //  vector_prop = &ps.vec3_vertex_property("vector");
    //  break;
    case type_t::unknown:
    default:
      std::cerr << "unknown type.\n";
      return 1;
  }

  for (std::size_t i = 0; i < options.num_datapoints; ++i) {
    ps.insert_vertex(rand(), rand());
  }

  for (auto const v : ps.vertices()) {
    switch (options.type) {
      case type_t::franke: {
        auto f             = analytical::numerical::frankes_test{};
        auto df            = diff(f);
        scalar_prop->at(v) = f(ps[v]);
        vector_prop->at(v) = df(ps[v]);
      } break;
      //case type_t::vector:
      //  vector_prop->at(v)(0) = rand();
      //  vector_prop->at(v)(1) = rand();
      //  vector_prop->at(v)(2) = rand();
      //  break;
      case type_t::unknown:
      default:
        std::cerr << "unknown type.\n";
        return 1;
    }
  }

  auto gr =
      uniform_rectilinear_grid2{linspace{0.0, 1.0, options.output_res_x},
                                linspace{0.0, 1.0, options.output_res_y}};

  auto sample_scalar = [&] {
    {
    auto sampler =
        ps.radial_basis_functions_sampler(*scalar_prop, *vector_prop);
    gr.sample_to_vertex_property(sampler, "rbf_with_gradient",
                                 execution_policy::parallel);
    gr.sample_to_vertex_property(diff(sampler), "gradient_of_rbf_with_gradient",
                                 execution_policy::parallel);
    }
    if (options.kernel == kernel_t::linear) {
      gr.sample_to_vertex_property(
          ps.radial_basis_functions_sampler_with_linear_kernel(*scalar_prop),
          "rbf_linear", execution_policy::parallel);
    } else if (options.kernel == kernel_t::cubic) {
      gr.sample_to_vertex_property(
          ps.radial_basis_functions_sampler_with_cubic_kernel(*scalar_prop),
          "rbf_cubic", execution_policy::parallel);
    } else if (options.kernel == kernel_t::thin_plate_spline) {
      gr.sample_to_vertex_property(
          ps.radial_basis_functions_sampler(
              *scalar_prop,
              [](auto const dd) { return dd * dd * gcem::log(dd) / 2; }),
          "rbf_thin_plate_spline", execution_policy::parallel);
    } else if (options.kernel == kernel_t::gaussian) {
      auto sampler = ps.radial_basis_functions_sampler_with_gaussian_kernel(
          *scalar_prop, options.epsilon);
      gr.sample_to_vertex_property(
          sampler, "rbf_gaussian_" + std::to_string(options.epsilon),
          execution_policy::parallel);
    }
  };
  switch (options.type) {
    case type_t::franke:
      sample_scalar();

      gr.sample_to_vertex_property(
          analytical::numerical::frankes_test{}, "franke",
          execution_policy::parallel);
      gr.sample_to_vertex_property(
          diff(analytical::numerical::frankes_test{}), "franke_gradient",
          execution_policy::parallel);
      break;
    // case type_t::vector:
    //   sample_vector();
    //   break;
    case type_t::unknown:
    default:
      std::cerr << "unknown type.\n";
      return 1;
  }
  gr.write("radial_basis_functions_sampler.vtr");
  ps.write("radial_basis_functions_data.vtp");
}
//==============================================================================
auto parse_args(int const argc, char const** argv) -> std::optional<options_t> {
  namespace po             = boost::program_options;
  auto        type         = type_t ::unknown;
  auto        kernel       = kernel_t::unknown;
  auto        epsilon      = real_number{};
  std::size_t output_res_x = 0, output_res_y = 0, num_datapoints = 0;

  auto desc = po::options_description{"Allowed options"};
  auto vm   = po::variables_map{};

  // Declare supported options.
  desc.add_options()("help", "produce help message")(
      "type", po::value<type_t>(), "franke")("kernel", po::value<kernel_t>(),
                                             "kernel")(
      "epsilon", po::value<real_number>(), "epsilion")(
      "num_datapoints", po::value<std::size_t>(), "number of data points")(
      "output_res_x", po::value<std::size_t>(), "set outputresolution width")(
      "output_res_y", po::value<std::size_t>(), "set outputresolution height");

  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") > 0) {
    std::cout << desc;
    return std::nullopt;
  }
  if (vm.count("kernel") > 0) {
    kernel = vm["kernel"].as<kernel_t>();
  } else {
    std::cerr << "--kernel not specified!\n";
    return std::nullopt;
  }
  if (vm.count("type") > 0) {
    type = vm["type"].as<type_t>();
  } else {
    std::cerr << "--type not specified!\n";
    return std::nullopt;
  }
  if (kernel == kernel_t::gaussian) {
    if (vm.count("epsilon") > 0) {
      epsilon = vm["epsilon"].as<real_number>();
    } else {
      std::cerr << "kernel is gaussian but --epsilon was not specified!\n";
      return std::nullopt;
    }
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
  return options_t{type,         kernel,       epsilon,
                   output_res_x, output_res_y, num_datapoints};
}
//------------------------------------------------------------------------------
auto operator>>(std::istream& in, type_t& t) -> std::istream& {
  std::string token;
  in >> token;
  if (token == "franke") {
    t = type_t::franke;
    //} else if (token == "vector") {
    //  t = type_t::vector;
  } else {
    t = type_t::unknown;
    in.setstate(std::ios_base::failbit);
  }
  return in;
}
//------------------------------------------------------------------------------
auto operator>>(std::istream& in, kernel_t& t) -> std::istream& {
  std::string token;
  in >> token;
  if (token == "linear") {
    t = kernel_t::linear;
  } else if (token == "cubic") {
    t = kernel_t::cubic;
  } else if (token == "thin_plate_spline") {
    t = kernel_t::thin_plate_spline;
  } else if (token == "gaussian") {
    t = kernel_t::gaussian;
  } else {
    t = kernel_t::unknown;
    in.setstate(std::ios_base::failbit);
  }
  return in;
}
//==============================================================================
