#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>

#include <boost/program_options.hpp>
//==============================================================================
using namespace tatooine;
//=============================================================================
auto advect(auto const& v, auto& particles, auto& uuid_generator,
            auto const t_0, auto const t_end, std::size_t const num_frames) {
  auto phi = flowmap(v);

  auto i      = std::size_t{};
  auto t_ends = linspace{t_0, t_end, num_frames + 1};
  t_ends.pop_front();
  for (auto const t : t_ends) {
    write_vtp(particles, 33, "animation_" + std::to_string(i) + ".vtp",
              backward);
    particles = std::get<0>(autonomous_particle2::advect_with_three_splits(
        phi, 0.01, t, particles, uuid_generator));
    std::cout << "current time: " << t
              << "\nnumber of particles: " << size(particles) << '\n';
    ++i;
  }
  write_vtp(particles, 33, "animation_" + std::to_string(i) + ".vtp", backward);
  std::cout << "current time: " << t_end
            << "\nnumber of particles: " << size(particles) << '\n';
}
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto uuid_generator = std::atomic_uint64_t{};
  auto t_0            = real_number(0);
  auto t_end          = real_number(4);
  auto num_frames     = std::size_t(10);
  namespace po        = boost::program_options;

  auto desc = po::options_description{"Allowed options"};
  desc.add_options()("help", "produce help message")(
      "num_frames", po::value<std::size_t>(), "number of frames")(
      "t_0", po::value<real_number>(), "start time of integration")(
      "t_end", po::value<real_number>(), "end time of integration");

  auto variables_map = po::variables_map{};
  po::store(po::parse_command_line(argc, argv, desc), variables_map);
  po::notify(variables_map);

  if (variables_map.count("help") > 0) {
    std::cout << desc;
    return 0;
  }

  if (variables_map.count("num_frames") > 0) {
    num_frames = variables_map["num_frames"].as<std::size_t>();
  } else {
    throw std::runtime_error{"Flag --num_frames not specified."};
  }
  if (variables_map.count("t_0") > 0) {
    t_0 = variables_map["t_0"].as<real_number>();
  }
  if (variables_map.count("t_end") > 0) {
    t_end = variables_map["t_end"].as<real_number>();
  } else {
    throw std::runtime_error{"Flag --t_end not specified."};
  }
  //============================================================================
  auto dg = analytical::fields::numerical::doublegyre{};
  dg.set_infinite_domain(true);

  auto const eps                  = 1e-3;
  auto       initial_particles_dg = autonomous_particle2::particles_from_grid(
            t_0,
            rectilinear_grid{linspace{0.0 + eps, 2.0 - eps, 61},
                       linspace{0.0 + eps, 1.0 - eps, 31}},
            uuid_generator);
  advect(dg, initial_particles_dg, uuid_generator, t_0, t_end, num_frames);
}
