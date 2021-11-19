#include <tatooine/field.h>
#include <tatooine/hdf5.h>
//==============================================================================
namespace tat = tatooine;
//==============================================================================
static constexpr size_t chunk_size = 256;
//==============================================================================
auto add_velocity_magnitude(auto const& domain, auto& channelflow_file, auto const& velx,
                 auto const& vely, auto const& velz) {
  auto calc_vel_mag = [&](auto const ix, auto const iy, auto const iz) {
    return length(
        tat::vec{velx(ix, iy, iz), vely(ix, iy, iz), velz(ix, iy, iz)});
  };
  std::atomic_size_t cnt  = 0;
  size_t const       max  = domain.num_vertices();
  bool               stop = false;
  std::cerr << "adding dataset to hdf5 file... ";
  auto vel_mag = channelflow_file.template create_dataset<double>(
      "vel_mag", domain.size(0) - 1, domain.size(1), domain.size(2));
  std::cerr << "done.\n";
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::cerr << "allocating data... ";
  tat::dynamic_multidim_array<double> vel_mag_data{domain.size(0) - 1, domain.size(1),
                                             domain.size(2)};
  std::cerr << "done.\n";
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::thread watcher{[&cnt, &stop, max] {
    while (cnt < max && !stop) {
      std::cerr << "Calculating velocity magnitude... " << std::setprecision(5)
                << double(cnt) / max * 100 << "%                \r";
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
  }};

  //iterate chunk-wise
  for (size_t chunk_iz = 0; chunk_iz < domain.size(2) / chunk_size;
       ++chunk_iz) {
    for (size_t chunk_iy = 0; chunk_iy < domain.size(1) / chunk_size;
         ++chunk_iy) {
      for (size_t chunk_ix = 0; chunk_ix < domain.size(0) / chunk_size;
           ++chunk_ix) {
#pragma omp parallel for collapse(3)
        for (size_t inner_iz = 0;
             inner_iz < std::min<size_t>(
                            chunk_size, chunk_iz * chunk_size - domain.size(2));
             ++inner_iz) {
          for (size_t inner_iy = 0;
               inner_iy < std::min<size_t>(chunk_size, chunk_iy * chunk_size -
                                                           domain.size(1));
               ++inner_iy) {
            for (size_t inner_ix = 0;
                 inner_ix < std::min<size_t>(chunk_size, chunk_iz * chunk_size -
                                                             domain.size(0));
                 ++inner_ix) {
              auto const ix = chunk_ix * chunk_size + inner_ix;
              auto const iy = chunk_iy * chunk_size + inner_iy;
              auto const iz = chunk_iz * chunk_size + inner_iz;
              if (ix < domain.size(0) - 2) {
                vel_mag_data(ix, iy, iz) = calc_vel_mag(ix, iy, iz);
              }
              ++cnt;
            }
          }
        }
        std::cerr << "chunk finished\n";
      }
    }
  }
  stop = true;
  watcher.join();
  std::cerr << "Calculating velocity magnitude... done.";
  std::cerr << "writing actual velocity magnitude data... ";
  vel_mag.write(vel_mag_data);
  std::cerr << "done.\n";
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
  full_domain.set_chunk_size_for_lazy_properties(chunk_size);

  auto axis0_vel_mag = axis0;
  axis0_vel_mag.pop_back();
  tat::grid full_domain_vel_mag{axis0_vel_mag, axis1, axis2};
  full_domain_vel_mag.set_chunk_size_for_lazy_properties(chunk_size);

  // open hdf5 files
  tat::hdf5::file channelflow_122_full_file{
      "/home/vcuser/channel_flow/dino_res_122000_full.h5"};

  auto velx_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/xvel");
  auto vely_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/yvel");
  auto velz_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/zvel");

  auto& velx_122_full = full_domain.insert_lazy_vertex_property<tat::x_fastest>(
      velx_122_full_dataset, "Vx_122");
  auto& vely_122_full = full_domain.insert_lazy_vertex_property<tat::x_fastest>(
      vely_122_full_dataset, "Vy_122");
  auto& velz_122_full = full_domain.insert_lazy_vertex_property<tat::x_fastest>(
      velz_122_full_dataset, "Vz_122");
  velx_122_full.limit_num_chunks_loaded();
  vely_122_full.limit_num_chunks_loaded();
  velz_122_full.limit_num_chunks_loaded();
  velx_122_full.set_max_num_chunks_loaded(30);
  vely_122_full.set_max_num_chunks_loaded(30);
  velz_122_full.set_max_num_chunks_loaded(30);

  add_velocity_magnitude(full_domain, channelflow_122_full_file, velx_122_full,
              vely_122_full, velz_122_full);
}
