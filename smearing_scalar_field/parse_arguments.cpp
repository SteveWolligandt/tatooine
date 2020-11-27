#include "parse_arguments.h"
#include <boost/program_options.hpp>
//==============================================================================
namespace tatooine::smearing {
//==============================================================================
auto parse_arguments(int argc, char const** argv) -> std::optional<arguments> { 
  namespace po = boost::program_options;

  std::filesystem::path    input_file, output_file;
  double                   t0 = 0;
  std::vector<double>      starting_point;
  std::vector<double>      end_point;
  double                   inner_radius   = 0.03;
  double                   outer_radius   = 0.1;
  double                   temporal_range = 0.1;
  size_t                   num_steps      = 1;
  bool                     write_vtk      = false;
  std::vector<std::string> fields;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "fields", po::value<std::vector<std::string>>()->multitoken(),
      "fields to smear. must be a, b or both")(
      "input", po::value<std::string>(), "file to read")(
      "output", po::value<std::string>(), "file to write")(
      "inner_radius", po::value<double>(),
      "specifies the inner radius of the smearing")(
      "outer_radius", po::value<double>(),
      "specifies the outer radius of the smearing")(
      "t0", po::value<double>(),
      "specifies the point where the smearing has its greatest impact")(
      "temporal_range", po::value<double>(),
      "specifies the temporal width of the smearing")(
      "start", po::value<std::vector<double>>()->multitoken(),
      "starting point of the smearing")(
      "end", po::value<std::vector<double>>()->multitoken(),
      "end point of the smearing")("num_steps", po::value<size_t>(),
                                   "number of smearing between start and end")(
      "write_vtk", po::value<bool>(), "additionally writes data as vtk");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") > 0) {
    std::cerr << desc;
    return {};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("input") > 0) {
    input_file = vm["input"].as<std::string>();
  } else {
    throw std::runtime_error{"You need to specify an input file!"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("output") > 0) {
    output_file = vm["output"].as<std::string>();
  } else {
    throw std::runtime_error{"You need to specify an output file!"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("inner_radius") > 0) {
    inner_radius = vm["inner_radius"].as<double>();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("outer_radius") > 0) {
    outer_radius = vm["outer_radius"].as<double>();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("t0") > 0) {
    t0 = vm["t0"].as<double>();
  } else {
    throw std::runtime_error{"You need to specify t0!"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("temporal_range") > 0) {
    temporal_range = vm["temporal_range"].as<double>();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("fields") > 0) {
    fields = vm["fields"].as<std::vector<std::string>>();
    for (auto const& f : fields) {
      if (f != "a" && f != "b") {
        throw std::runtime_error{"Fields only can be \"a\" or \"b\"!"};
      }
    }
  } else {
    throw std::runtime_error{"You need to specify which fields to smear!"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("write_vtk") > 0) {
    write_vtk = vm["write_vtk"].as<bool>();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("start") > 0) {
    starting_point = vm["start"].as<std::vector<double>>();
    if (size(starting_point) != 2) {
      throw std::runtime_error{
          "starting point must have exactly 2 components!"};
    }
  } else {
      throw std::runtime_error{"You have to specify a starting point!"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("end") > 0) {
    end_point = vm["end"].as<std::vector<double>>();
    if (size(end_point) != 2) {
      throw std::runtime_error{"end point must have exactly 2 components!"};
    }
  } else {
      throw std::runtime_error{"You have to specify an end point!"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if (vm.count("num_steps") > 0) {
    num_steps = vm["num_steps"].as<size_t>();
  }

  auto const dir = (vec2{end_point[0], end_point[1]} -
                    vec2{starting_point[0], starting_point[1]}) /
                   num_steps;
  return arguments{input_file,
                   output_file,
                   geometry::sphere2{outer_radius, vec2{starting_point[0],
                                                        starting_point[1]}},
                   inner_radius,
                   vec2{end_point[0], end_point[1]},
                   temporal_range,
                   t0,
                   dir,
                   num_steps,
                   write_vtk,
                   fields};
}
//==============================================================================
}  // namespace tatooine::smearing
//==============================================================================
