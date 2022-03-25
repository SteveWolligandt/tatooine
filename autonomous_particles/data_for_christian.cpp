#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>

#include <boost/program_options.hpp>
//==============================================================================
using namespace tatooine;
auto constexpr advection_direction = forward;
//==============================================================================
using autonomous_particle_flowmap_type =
    AutonomousParticleFlowmapDiscretization<
        2, AutonomousParticle<2>::split_behaviors::five_splits>;
//==============================================================================
auto doit(auto const& v, auto const& initial_particles, auto& uuid_generator,
          auto const& q, auto const t_0, auto const t_end) {
  auto const tau = t_end - t_0;
  auto       phi = flowmap(v);

  auto flowmap_autonomous_particles = autonomous_particle_flowmap_type{
      phi, t_end, 0.01, initial_particles, uuid_generator};

  auto const num_particles_after_advection =
      flowmap_autonomous_particles.num_particles();
  std::cout << "num_particles_after_advection: "
            << num_particles_after_advection << '\n';

  auto kd_tree_data = flann::Matrix<real_number>{
      const_cast<real_number*>(flowmap_autonomous_particles.samplers()
                                   .front()
                                   .x0(advection_direction)
                                   .data_ptr()),
      flowmap_autonomous_particles.samplers().size(), 2,
      sizeof(autonomous_particle_flowmap_type::sampler_type)};
  auto kd_tree = flann::Index<flann::L2<real_number>>{
      kd_tree_data, flann::KDTreeSingleIndexParams{}};
  kd_tree.buildIndex();
  auto kd_tree_query =
      flann::Matrix<real_number>{const_cast<real_number*>(q.data_ptr()), 1, 2};
  auto nearest_indices_   = std::vector<std::vector<int>>{};
  auto squared_distances_ = std::vector<std::vector<real_number>>{};
  kd_tree.knnSearch(kd_tree_query, nearest_indices_, squared_distances_, 40,
                    flann::SearchParams{});
  auto const& nearest_indices   = nearest_indices_.front();
  auto const& squared_distances = squared_distances_.front();

  auto rbf_kernel = thin_plate_spline_from_squared;

  // construct lower part of symmetric matrix A
  auto const dim_A = nearest_indices.size()  // number of vertices
                     + 1                     // constant monomial part
                     + 2                     // linear monomial part
      ;

  auto A = tensor<real_number>::zeros(dim_A, dim_A);
  auto radial_and_monomial_coefficients = tensor<real_number>::zeros(dim_A, 2);
  for (std::size_t c = 0; c < nearest_indices.size(); ++c) {
    for (std::size_t r = c + 1; r < nearest_indices.size(); ++r) {
      A(r, c) = rbf_kernel(squared_euclidean_distance(
          flowmap_autonomous_particles.samplers()[nearest_indices[c]].x0(
              advection_direction),
          flowmap_autonomous_particles.samplers()[nearest_indices[r]].x0(
              advection_direction)));
    }
  }
  // construct polynomial requirements
  for (std::size_t c = 0; c < nearest_indices.size(); ++c) {
    auto const& p =
        flowmap_autonomous_particles.samplers()[nearest_indices[c]].x0(
            advection_direction);
    // constant part
    A(nearest_indices.size(), c) = 1;

    // linear part
    for (std::size_t i = 0; i < 2; ++i) {
      A(nearest_indices.size() + 1 + i, c) = p(i);
    }
  }

  for (std::size_t i = 0; i < nearest_indices.size(); ++i) {
    auto const v = nearest_indices[i];
    auto const phi =
        flowmap_autonomous_particles.samplers()[v].phi(advection_direction);
    for (std::size_t j = 0; j < 2; ++j) {
      radial_and_monomial_coefficients(i, j) = phi(j);
    }
  }
  radial_and_monomial_coefficients = *solve_symmetric_lapack(
      std::move(A), std::move(radial_and_monomial_coefficients),
      tatooine::lapack::Uplo::Lower);

  // reconstruct
  auto reconstructed_flowmap_q = vec2{};
  // radial bases
  for (std::size_t i = 0; i < nearest_indices.size(); ++i) {
    if (squared_distances[i] == 0) {
      reconstructed_flowmap_q =
          flowmap_autonomous_particles.samplers()[nearest_indices[i]].phi(
              advection_direction);
      break;
    }
    for (std::size_t j = 0; j < 2; ++j) {
      reconstructed_flowmap_q(j) +=
          radial_and_monomial_coefficients(i, j) *
          rbf_kernel(squared_euclidean_distance(
              q, flowmap_autonomous_particles.samplers()[nearest_indices[i]].x0(
                     advection_direction)));
    }
  }
  // monomial bases
  for (std::size_t j = 0; j < 2; ++j) {
    reconstructed_flowmap_q(j) +=
        radial_and_monomial_coefficients(nearest_indices.size(), j);
    for (std::size_t k = 0; k < 2; ++k) {
      reconstructed_flowmap_q(j) +=
          radial_and_monomial_coefficients(nearest_indices.size() + 1 + k, j) *
          q(k);
    }
  }

  auto const flowmap_runge_kutta_q = phi(q, t_0, tau);
  auto const error =
      euclidean_distance(reconstructed_flowmap_q, flowmap_runge_kutta_q);
  std::cout << "t_0           = " << t_0 << '\n';
  std::cout << "tau           = " << tau << '\n';
  std::cout << "q             = " << q << '\n';
  std::cout << "ground truth  = " << flowmap_runge_kutta_q << '\n';
  std::cout << "reconstructed = " << reconstructed_flowmap_q << '\n';
  std::cout << "error         = " << error << '\n';
  auto f = std::ofstream{"data_for_christian.bin"};
  if (f.is_open()) {
    f.write(reinterpret_cast<char const*>(q.data_ptr()),
            sizeof(real_number) * 2);
    f.write(reinterpret_cast<char const*>(flowmap_runge_kutta_q.data_ptr()),
            sizeof(real_number) * 2);
    f.write(reinterpret_cast<char const*>(reconstructed_flowmap_q.data_ptr()),
            sizeof(real_number) * 2);

    for (auto const i : nearest_indices) {
      auto const& sampler = flowmap_autonomous_particles.samplers()[i];

      f.write(reinterpret_cast<char const*>(
                  sampler.x0(advection_direction).data_ptr()),
              sizeof(real_number) * 2);
      f.write(reinterpret_cast<char const*>(
                  sampler.phi(advection_direction).data_ptr()),
              sizeof(real_number) * 2);
      f.write(reinterpret_cast<char const*>(
                  sampler.nabla_phi(advection_direction).data_ptr()),
              sizeof(real_number) * 4);
    }
    f.close();
  }

  write_vtp(flowmap_autonomous_particles.samplers(), 33,
            "data_for_christian_ellipses_forward.vtp", forward);
  write_vtp(flowmap_autonomous_particles.samplers(), 33,
            "data_for_christian_ellipses_backward.vtp", backward);
}
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto uuid_generator = std::atomic_uint64_t{};
  auto t_0            = real_number(0);
  auto t_end          = real_number{};
  auto x0             = vec2{};

  namespace po        = boost::program_options;

  auto desc = po::options_description{"Allowed options"};
  desc.add_options()("help", "produce help message")(
      "t_0", po::value<real_number>(), "start time of integration")(
      "x0", po::value<std::vector<real_number>>()->multitoken(), "x0")(
      "t_end", po::value<real_number>(), "end time of integration");

  auto variables_map = po::variables_map{};
  po::store(po::parse_command_line(argc, argv, desc), variables_map);
  po::notify(variables_map);

  if (variables_map.count("help") > 0) {
    std::cout << desc;
    return 0;
  }

  if (variables_map.count("x0") > 0) {
    auto const x0_data = variables_map["x0"].as<std::vector<real_number>>();
    if (size(x0_data) != 2) {
      throw std::runtime_error{"x0 needs to have exactly two values"};
    }
    auto       i       = std::size_t{};
    for (auto c : x0_data) {
      x0(i++) = c;
    }
  } else {
    throw std::runtime_error{"Flag --x0 not specified."};
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
  auto const initial_particles_dg = autonomous_particle2::particles_from_grid(
      t_0,
      rectilinear_grid{linspace{0.0 + eps, 2.0 - eps, 61},
                       linspace{0.0 + eps, 1.0 - eps, 31}},
      uuid_generator);
  doit(dg, initial_particles_dg, uuid_generator,
       x0, t_0, t_end);
}
