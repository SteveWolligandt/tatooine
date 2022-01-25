#include <tatooine/field.h>
#include <tatooine/for_loop.h>
#include <tatooine/hdf5.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
static constexpr auto chunk_size = std::size_t(128);
//==============================================================================
auto add_Q_pnorm(auto const& domain, auto& channelflow_file, auto const& velx,
                 auto const& vely, auto const& velz) {
  domain.update_diff_stencil_coefficients();
  auto       cnt  = std::atomic_size_t{0};
  auto const max  = domain.vertices().size();
  auto       stop = false;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::cerr << "allocating data... ";
  auto Q_pnorm_data = dynamic_multidim_array<double>{
      domain.size(0), domain.size(1), domain.size(2)};
  auto Q_cheng_data = dynamic_multidim_array<double>{
      domain.size(0), domain.size(1), domain.size(2)};
  std::cerr << "done.\n";
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto watcher = std::thread{[&cnt, &stop, max] {
    while (cnt < max && !stop) {
      std::cerr << "Calculating Q... " << std::setprecision(5)
                << double(cnt) / max * 100 << "%                \r";
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
  }};

  auto process_chunk = [&](auto const ix, auto const iy, auto const iz) {
    auto J   = mat3{};
    J.row(0) = diff(velx)(ix, iy, iz);
    J.row(1) = diff(vely)(ix, iy, iz);
    J.row(2) = diff(velz)(ix, iy, iz);

    mat3 const S     = (J + transposed(J)) / 2;
    mat3 const Omega = (J - transposed(J)) / 2;

    auto qcr =
        0.5 * (J(0, 0) + J(1, 1) + J(2, 2)) * (J(0, 0) + J(1, 1) + J(2, 2)) -
        0.5 * (J(0, 0) * J(0, 0) + J(1, 1) * J(1, 1) + J(2, 2) * J(2, 2));
    qcr -= J(0, 1) * J(1, 0) + J(0, 2) * J(2, 0) + J(1, 2) * J(2, 1);

    Q_pnorm_data(ix, iy, iz) =
        (squared_norm(Omega, 2) - squared_norm(S, 2)) / 2;
    Q_cheng_data(ix, iy, iz) = qcr;
    ++cnt;
  };

  chunked_for_loop(process_chunk, execution_policy::parallel, chunk_size,
                   domain.size(0), domain.size(1), domain.size(2));
  stop = true;
  watcher.join();
  std::cerr << "Calculating Q... done.";
  std::cerr << "writing actual Q data... ";
  std::cerr << "adding dataset to hdf5 file... ";
  auto Q_pnorm = channelflow_file.template create_dataset<double>(
      "Q_pnorm", domain.size(0), domain.size(1), domain.size(2));
  auto Q_cheng = channelflow_file.template create_dataset<double>(
      "Q_cheng", domain.size(0), domain.size(1), domain.size(2));
  Q_pnorm.write(Q_pnorm_data);
  Q_cheng.write(Q_cheng_data);
  std::cerr << "done.\n";
}
//==============================================================================
auto main(int argc, char** argv) -> int {
  // open hdf5 files
  auto       channelflow_file = hdf5::file{argv[1]};
  auto const axis0 =
      channelflow_file.dataset<double>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      channelflow_file.dataset<double>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      channelflow_file.dataset<double>("CartGrid/axis2").read_as_vector();

  auto full_domain = rectilinear_grid{axis0, axis1, axis2};
  full_domain.set_chunk_size_for_lazy_properties(chunk_size);

  auto velx_dataset = channelflow_file.dataset<double>("velocity/xvel");
  auto vely_dataset = channelflow_file.dataset<double>("velocity/yvel");
  auto velz_dataset = channelflow_file.dataset<double>("velocity/zvel");

  auto& velx = full_domain.insert_vertex_property(velx_dataset, "Vx");
  auto& vely = full_domain.insert_vertex_property(vely_dataset, "Vy");
  auto& velz = full_domain.insert_vertex_property(velz_dataset, "Vz");
  //velx.limit_num_chunks_loaded();
  //vely.limit_num_chunks_loaded();
  //velz.limit_num_chunks_loaded();
  //velx.set_max_num_chunks_loaded(30);
  //vely.set_max_num_chunks_loaded(30);
  //velz.set_max_num_chunks_loaded(30);

  add_Q_pnorm(full_domain, channelflow_file, velx, vely, velz);
}
