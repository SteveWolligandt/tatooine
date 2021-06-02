#include <tatooine/field.h>
#include <tatooine/hdf5.h>
//==============================================================================
namespace tat = tatooine;
//==============================================================================
auto add_Q_pnorm(auto const& domain, auto& channelflow_file, auto const& velx,
                 auto const& vely, auto const& velz) {
  auto calc_Q = [&](auto const ix, auto const iy, auto const iz) {
    tat::mat3  J;
    J.col(0) = diff(velx)(ix, iy, iz);
    J.col(1) = diff(vely)(ix, iy, iz);
    J.col(2) = diff(velz)(ix, iy, iz);
    tat::mat3 const S     = (J + transposed(J)) / 2;
    tat::mat3 const Omega = (J - transposed(J)) / 2;
    return (sqr_norm(Omega, 2) - sqr_norm(S, 2)) / 2;
  };
  std::atomic_size_t cnt  = 0;
  size_t const       max  = domain.num_vertices();
  bool               stop = false;
  tat::dynamic_multidim_array<double> Q_data{domain.size(0) - 1, domain.size(1),
                                             domain.size(2)};
  std::thread            watcher{[&cnt, &stop, max] {
    while (cnt < max && !stop) {
      std::cerr << std::setprecision(2) << double(cnt) / max * 100
                << "%                \r";
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
  }};
  domain.parallel_loop_over_vertex_indices(
      [&](auto const ix, auto const iy, auto const iz) {
        if (ix < domain.size(0) - 2) {
          Q_data(ix, iy, iz) = calc_Q(ix, iy, iz);
        }
        ++cnt;
      });
  stop = true;
  watcher.join();
  auto Q = channelflow_file.template add_dataset<double>(
      "Q_pnorm", domain.size(0) - 1, domain.size(1), domain.size(2));
  Q.write(Q_data);
}
//==============================================================================
auto main() -> int {
  tat::hdf5::file axis0_file{"/home/vcuser/channel_flow/axis0.h5"};
  tat::hdf5::file axis1_file{"/home/vcuser/channel_flow/axis1.h5"};
  tat::hdf5::file axis2_file{"/home/vcuser/channel_flow/axis2.h5"};
  auto const      axis0 =
      axis0_file.dataset<double>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      axis1_file.dataset<double>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      axis2_file.dataset<double>("CartGrid/axis2").read_as_vector();

  tat::grid full_domain{axis0, axis1, axis2};
  full_domain.set_chunk_size_for_lazy_properties(512);
  std::cerr << "full_domain:\n" << full_domain << '\n';

  auto axis0_Q = axis0;
  axis0_Q.pop_back();
  tat::grid full_domain_Q{axis0_Q, axis1, axis2};
  full_domain_Q.set_chunk_size_for_lazy_properties(512);
  std::cerr << "full_domain_Q:\n" << full_domain_Q << '\n';

  // open hdf5 files
  tat::hdf5::file channelflow_122_full_file{
      "/home/vcuser/channel_flow/dino_res_122000_full.h5"};

  auto velx_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/xvel");
  auto vely_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/yvel");
  auto velz_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/zvel");

  auto& velx_122_full = full_domain.insert_lazy_vertex_property<tat::x_slowest>(
      velx_122_full_dataset, "Vx_122");
  auto& vely_122_full = full_domain.insert_lazy_vertex_property<tat::x_slowest>(
      vely_122_full_dataset, "Vy_122");
  auto& velz_122_full = full_domain.insert_lazy_vertex_property<tat::x_slowest>(
      velz_122_full_dataset, "Vz_122");
  velx_122_full.limit_num_chunks_loaded();
  velx_122_full.set_max_num_chunks_loaded(4);
  vely_122_full.limit_num_chunks_loaded();
  vely_122_full.set_max_num_chunks_loaded(4);
  velz_122_full.limit_num_chunks_loaded();
  velz_122_full.set_max_num_chunks_loaded(4);


  std::cout << "adding Q...\n";
  add_Q_pnorm(full_domain, channelflow_122_full_file, velx_122_full,
              vely_122_full, velz_122_full);
}
