#include "parse_args.h"

#include <boost/program_options.hpp>
#include <iostream>
//------------------------------------------------------------------------------
auto parse_args(int argc, char** argv) -> std::optional<args_t> {
  namespace po = boost::program_options;

  size_t width = 10, height = 10, depth = 10, num_splits = 3,
         max_num_particles = 500000, output_res_x = 200, output_res_y = 100,
         output_res_z = 100;
  double t0 = 0, tau = 2, tau_step = 0.05, min_cond = 0.01,
         agranovsky_delta_t      = 0.1;
  bool write_ellipses            = true;
  auto autonomous_particles_file = std::optional<tatooine::filesystem::path>{};
  // Declare the supported options.
  po::options_description desc("Allowed options");
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
      "min_cond", po::value<double>(),
      "set minimal condition number of back calculation for advected "
      "particles")("agranovsky_delta_t", po::value<double>(), "time gaps")(
      "autonomous_particles_file", po::value<std::string>(),
      "already integrated particles");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") > 0) {
    std::cerr << desc;
    return {};
  }
  if (vm.count("autonomous_particles_file") > 0) {
    autonomous_particles_file = tatooine::filesystem::path{
        vm["autonomous_particles_file"].as<std::string>()};
    std::cerr << "reading from file " << *autonomous_particles_file << '\n';
  }
  if (vm.count("width") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify width!"};
    }
    width = vm["width"].as<size_t>();
    std::cerr << "specified width = " << width << '\n';
  } else {
    std::cerr << "default width = " << width << '\n';
  }
  if (vm.count("height") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify "
          "height!"};
    }
    height = vm["height"].as<size_t>();
    std::cerr << "specified height = " << height << '\n';
  } else {
    std::cerr << "default height = " << height << '\n';
  }
  if (vm.count("depth") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify depth!"};
    }
    depth = vm["depth"].as<size_t>();
    std::cerr << "specified depth = " << depth << '\n';
  } else {
    std::cerr << "default depth = " << depth << '\n';
  }
  if (vm.count("output_res_x") > 0) {
    output_res_x = vm["output_res_x"].as<size_t>();
    std::cerr << "specified output_res_x = " << output_res_x << '\n';
  } else {
    std::cerr << "default output_res_x = " << output_res_x << '\n';
  }
  if (vm.count("output_res_y") > 0) {
    output_res_y = vm["output_res_y"].as<size_t>();
    std::cerr << "specified output_res_y = " << output_res_y << '\n';
  } else {
    std::cerr << "default output_res_y = " << output_res_y << '\n';
  }
  if (vm.count("output_res_z") > 0) {
    output_res_z = vm["output_res_z"].as<size_t>();
    std::cerr << "specified output_res_z = " << output_res_z << '\n';
  } else {
    std::cerr << "default output_res_z = " << output_res_z << '\n';
  }
  if (vm.count("num_splits") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify "
          "num_splits!"};
    }
    num_splits = vm["num_splits"].as<size_t>();
    std::cerr << "specified number of splits num_splits = " << num_splits
              << '\n';
  } else {
    std::cerr << "default number of splits num_splits = " << num_splits << '\n';
  }
  if (vm.count("max_num_particles") > 0) {
    if (autonomous_particles_file) {
      throw std::runtime_error{
          "\"autonomous_particles_file\" was specified. do not specify "
          "max_num_particles!"};
    }
    max_num_particles = vm["max_num_particles"].as<size_t>();
    std::cerr << "specified maximum number of particles = " << max_num_particles
              << '\n';
  } else {
    std::cerr << "default maximum number of particles = " << max_num_particles
              << '\n';
  }
  if (vm.count("t0") > 0) {
    t0 = vm["t0"].as<double>();
    std::cerr << "specified t0 = " << t0 << '\n';
  } else {
    std::cerr << "default t0 = " << t0 << '\n';
  }
  if (vm.count("tau") > 0) {
    tau = vm["tau"].as<double>();
    std::cerr << "specified integration length tau = " << tau << '\n';
  } else {
    std::cerr << "default integration length tau = " << tau << '\n';
  }
  if (vm.count("agranovsky_delta_t") > 0) {
    agranovsky_delta_t = vm["agranovsky_delta_t"].as<double>();
    std::cerr << "specified agranovsky_delta_t = " << agranovsky_delta_t
              << '\n';
  } else {
    std::cerr << "default integration length tau = " << tau << '\n';
  }
  if (vm.count("tau_step") > 0) {
    tau_step = vm["tau_step"].as<double>();
    std::cerr << "specified step width tau_step = " << tau_step << '\n';
  } else {
    std::cerr << "default step width tau_step = " << tau_step << '\n';
  }
  if (vm.count("min_cond") > 0) {
    min_cond = vm["min_cond"].as<double>();
    std::cerr << "specified min_cond = " << min_cond << '\n';
  } else {
    std::cerr << "default min_cond = " << min_cond << '\n';
  }
  if (vm.count("write_ellipses") > 0) {
    write_ellipses = vm["write_ellipses"].as<bool>();
    std::cerr << "specified write_ellipses = " << write_ellipses << '\n';
  } else {
    std::cerr << "default write_ellipses = " << write_ellipses << '\n';
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
                min_cond,
                agranovsky_delta_t,
                write_ellipses,
                autonomous_particles_file};
}
