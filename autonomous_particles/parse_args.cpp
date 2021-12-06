#include "parse_args.h"

#include <boost/program_options.hpp>
#include <iostream>
//------------------------------------------------------------------------------
auto parse_args(int argc, char** argv) -> std::optional<args_t> {
  namespace po = boost::program_options;

  size_t width = 10, height = 10, depth = 10, num_splits = 3,
         max_num_particles = 500000, output_res_x = 200, output_res_y = 100,
         output_res_z = 100;
  double t0 = 0, tau = 2, tau_step = 0.05, r0 = 0.01,
         agranovsky_delta_t      = 0.1;
  auto x0                        = tatooine::vec3{};
  bool write_ellipses            = true;
  bool show_dimensions           = false;
  auto autonomous_particles_file = std::optional<tatooine::filesystem::path>{};
  auto velocity_file             = std::optional<tatooine::filesystem::path>{};
  // Declare the supported options.
  auto desc = po::options_description{"Allowed options"};
  desc.add_options()("help", "produce help message")(
      "write_ellipses", po::value<bool>(), "write ellipses")(
      "width", po::value<size_t>(), "set width")("height", po::value<size_t>(),
                                                 "set height")(
      "output_res_x", po::value<size_t>(), "set output resolution width")(
      "output_res_y", po::value<size_t>(), "set output resolution height")(
      "output_res_z", po::value<size_t>(), "set output resolution depth")(
      "depth", po::value<size_t>(), "set depth")(
      "num_splits", po::value<size_t>(), "set number of splits")(
      "max_num_particles", po::value<size_t>(),
      "set maximum number of particles")("t0", po::value<double>(),
                                         "set initial time")(
      "tau", po::value<double>(), "set integration length tau")(
      "tau_step", po::value<double>(), "set stepsize for integrator")(
      "r0", po::value<double>(),
      "set minimal condition number of back calculation for advected "
      "particles")("agranovsky_delta_t", po::value<double>(), "time gaps")(
      "x0", po::value<std::vector<double>>()->multitoken(), "x0")(
      "autonomous_particles_file", po::value<std::string>(),
      "already integrated particles")("velocity_file", po::value<std::string>(),
                                      "file with velocity data")(
      "showdimensions,sd", po::bool_switch(&show_dimensions),
      "show dimensions of dataset");

  auto variables_map = po::variables_map{};
  po::store(po::parse_command_line(argc, argv, desc), variables_map);
  po::notify(variables_map);

  if (variables_map.count("help") > 0) {
    std::cout << desc;
    return {};
  }
  if (variables_map.count("autonomous_particles_file") > 0) {
    autonomous_particles_file = tatooine::filesystem::path{
        variables_map["autonomous_particles_file"].as<std::string>()};
    std::cout << "reading particles from  " << *autonomous_particles_file
              << '\n';
  }
  if (variables_map.count("velocity_file") > 0) {
    velocity_file = tatooine::filesystem::path{
        variables_map["velocity_file"].as<std::string>()};
    std::cout << "reading velocity from file " << *velocity_file << '\n';
  }
  if (variables_map.count("width") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify width!"};
    }
    width = variables_map["width"].as<size_t>();
    std::cout << "specified width = " << width << '\n';
  } else {
    std::cout << "default width = " << width << '\n';
  }
  if (variables_map.count("height") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify "
          "height!"};
    }
    height = variables_map["height"].as<size_t>();
    std::cout << "specified height = " << height << '\n';
  } else {
    std::cout << "default height = " << height << '\n';
  }
  if (variables_map.count("depth") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify depth!"};
    }
    depth = variables_map["depth"].as<size_t>();
    std::cout << "specified depth = " << depth << '\n';
  } else {
    std::cout << "default depth = " << depth << '\n';
  }
  if (variables_map.count("output_res_x") > 0) {
    output_res_x = variables_map["output_res_x"].as<size_t>();
    std::cout << "specified output_res_x = " << output_res_x << '\n';
  } else {
    std::cout << "default output_res_x = " << output_res_x << '\n';
  }
  if (variables_map.count("output_res_y") > 0) {
    output_res_y = variables_map["output_res_y"].as<size_t>();
    std::cout << "specified output_res_y = " << output_res_y << '\n';
  } else {
    std::cout << "default output_res_y = " << output_res_y << '\n';
  }
  if (variables_map.count("output_res_z") > 0) {
    output_res_z = variables_map["output_res_z"].as<size_t>();
    std::cout << "specified output_res_z = " << output_res_z << '\n';
  } else {
    std::cout << "default output_res_z = " << output_res_z << '\n';
  }
  if (variables_map.count("num_splits") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify "
          "num_splits!"};
    }
    num_splits = variables_map["num_splits"].as<size_t>();
    std::cout << "specified number of splits num_splits = " << num_splits
              << '\n';
  } else {
    std::cout << "default number of splits num_splits = " << num_splits << '\n';
  }
  if (variables_map.count("max_num_particles") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify "
          "max_num_particles!"};
    }
    max_num_particles = variables_map["max_num_particles"].as<size_t>();
    std::cout << "specified maximum number of particles = " << max_num_particles
              << '\n';
  } else {
    std::cout << "default maximum number of particles = " << max_num_particles
              << '\n';
  }
  if (variables_map.count("t0") > 0) {
    t0 = variables_map["t0"].as<double>();
    std::cout << "specified t0 = " << t0 << '\n';
  } else {
    std::cout << "default t0 = " << t0 << '\n';
  }
  if (variables_map.count("tau") > 0) {
    tau = variables_map["tau"].as<double>();
    std::cout << "specified integration length tau = " << tau << '\n';
  } else {
    std::cout << "default integration length tau = " << tau << '\n';
  }
  if (variables_map.count("agranovsky_delta_t") > 0) {
    agranovsky_delta_t = variables_map["agranovsky_delta_t"].as<double>();
    std::cout << "specified agranovsky_delta_t = " << agranovsky_delta_t
              << '\n';
  } else {
    std::cout << "default integration length tau = " << tau << '\n';
  }
  if (variables_map.count("x0") > 0) {
    auto const x0_data = variables_map["x0"].as<std::vector<double>>();
    auto       i       = std::size_t{};
    for (auto c : x0_data) {
      x0(i++) = c;
    }
    std::cout << "specified x0 = " << x0 << '\n';
  } else {
    std::cout << "default integration length tau = " << tau << '\n';
  }
  if (variables_map.count("tau_step") > 0) {
    tau_step = variables_map["tau_step"].as<double>();
    std::cout << "specified step width tau_step = " << tau_step << '\n';
  } else {
    std::cout << "default step width tau_step = " << tau_step << '\n';
  }
  if (variables_map.count("r0") > 0) {
    r0 = variables_map["r0"].as<double>();
    std::cout << "specified r0 = " << r0 << '\n';
  } else {
    std::cout << "default r0 = " << r0 << '\n';
  }
  if (variables_map.count("write_ellipses") > 0) {
    write_ellipses = variables_map["write_ellipses"].as<bool>();
    std::cout << "specified write_ellipses = " << write_ellipses << '\n';
  } else {
    std::cout << "default write_ellipses = " << write_ellipses << '\n';
  }
  return args_t{width,
                height,
                depth,
                num_splits,
                max_num_particles,
                output_res_x,
                output_res_y,
                output_res_z,
                t0,
                tau,
                tau_step,
                r0,
                agranovsky_delta_t,
                x0,
                write_ellipses,
                show_dimensions,
                autonomous_particles_file,
                velocity_file};
}
