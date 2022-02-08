#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/differentiated_field.h>
#include <tatooine/hdf5.h>
#include <tatooine/lazy_reader.h>
#include <tatooine/mpi/program.h>
#include <tatooine/parallel_vectors.h>
#include <tatooine/spacetime_vectorfield.h>
//==============================================================================
namespace tat    = tatooine;
using doublegyre = tat::analytical::fields::numerical::doublegyre<tat::real_type>;
using tat::linspace;
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto&          mpi = tat::mpi::program::get(argc, argv);
  constexpr auto eps = 1e-5;
  //----------------------------------------------------------------------------
  tat::hdf5::file channelflow_150_file{
      "/p/project/dnschannelfspp/project/channelf/data_hdf5/"
      "dino_res_150000.h5"};
  auto full_domain_x =
      channelflow_150_file.dataset<double>("CartGrid/axis0").read_as_vector();
  auto full_domain_y =
      channelflow_150_file.dataset<double>("CartGrid/axis1").read_as_vector();
  auto full_domain_z =
      channelflow_150_file.dataset<double>("CartGrid/axis2").read_as_vector();
  full_domain_z.pop_back();
  auto const xvel_150 = channelflow_150_file.dataset<double>("xvelocity/vel")
                            .read_lazy({2, 2, 2});
  auto const yvel_150 = channelflow_150_file.dataset<double>("yvelocity/vel")
                            .read_lazy({2, 2, 2});
  auto const zvel_150 = channelflow_150_file.dataset<double>("velocity/zvel")
                            .read_lazy({2, 2, 2});
  tat::nonuniform_rectilinear_grid<double, 3> full_domain_grid{
      full_domain_x, full_domain_y, full_domain_z};
  full_domain_grid.update_diff_stencil_coefficients();
  mpi.init_communicator(full_domain_grid);

  // local_grid covers all grid vertices exactly once distributed over all
  // processes
  auto const local_grid = mpi.local_grid(full_domain_grid);
  // halo_grid covers all grid cells exactly once distributed over all processes
  // with some redundant grid vertices.
  tat::nonuniform_rectilinear_grid<double, 3> halo_grid = local_grid;
  if (halo_grid.dimension<0>().back() !=
      full_domain_grid.dimension<0>().back()) {
    halo_grid.dimension<0>().push_back(
        full_domain_grid.dimension<0>()[mpi.process_end(0)]);
  }
  if (halo_grid.dimension<1>().back() !=
      full_domain_grid.dimension<1>().back()) {
    halo_grid.dimension<1>().push_back(
        full_domain_grid.dimension<1>()[mpi.process_end(1)]);
  }
  if (halo_grid.dimension<2>().back() !=
      full_domain_grid.dimension<2>().back()) {
    halo_grid.dimension<2>().push_back(
        full_domain_grid.dimension<2>()[mpi.process_end(2)]);
  }
  if (mpi.rank() == 0) {
    std::cout << "global grid\n" << full_domain_grid << '\n';
    std::cout << "local grid\n" << local_grid << '\n';
    std::cout << "halo grid\n" << halo_grid;
  }
  //----------------------------------------------------------------------------
  // doublegyre dg;
  // auto       v = spacetime(dg);
  // auto       J = diff(v, 1e-10);
  // auto       a = J * v;

  // auto const vortex_core_lines = parallel_vectors(v, a, halo_grid);
  // write_vtk(vortex_core_lines, "pv_stdg_mpi_rank_" +
  // std::to_string(mpi.rank()) + ".vtk");
  //----------------------------------------------------------------------------
  auto calc_J_at = [&](auto ix, auto iy, auto iz) {
    ix += mpi.process_begin(0);
    iy += mpi.process_begin(1);
    iz += mpi.process_begin(2);
    auto J = tat::mat3::zeros();
    // calc x
    if (ix == 0) {
      auto const stencil =
          full_domain_grid.diff_stencil_coefficients_0_p1_p2(0, ix);
      J.col(0) = tat::vec3{xvel_150(ix, iy, iz) * stencil[0] +
                               xvel_150(ix + 1, iy, iz) * stencil[1] +
                               xvel_150(ix + 2, iy, iz) * stencil[2],
                           yvel_150(ix, iy, iz) * stencil[0] +
                               yvel_150(ix + 1, iy, iz) * stencil[1] +
                               yvel_150(ix + 2, iy, iz) * stencil[2],
                           zvel_150(ix, iy, iz) * stencil[0] +
                               zvel_150(ix + 1, iy, iz) * stencil[1] +
                               zvel_150(ix + 2, iy, iz) * stencil[2]};
    } else if (ix == full_domain_grid.size(0) - 1) {
      auto const stencil =
          full_domain_grid.diff_stencil_coefficients_n2_n1_0(0, ix);
      J.col(0) = tat::vec3{xvel_150(ix - 2, iy, iz) * stencil[0] +
                               xvel_150(ix - 1, iy, iz) * stencil[1] +
                               xvel_150(ix, iy, iz) * stencil[2],
                           yvel_150(ix - 2, iy, iz) * stencil[0] +
                               yvel_150(ix - 1, iy, iz) * stencil[1] +
                               yvel_150(ix, iy, iz) * stencil[2],
                           zvel_150(ix - 2, iy, iz) * stencil[0] +
                               zvel_150(ix - 1, iy, iz) * stencil[1] +
                               zvel_150(ix, iy, iz) * stencil[2]};
    } else {
      auto const stencil =
          full_domain_grid.diff_stencil_coefficients_n1_0_p1(0, ix);
      J.col(0) = tat::vec3{xvel_150(ix - 1, iy, iz) * stencil[0] +
                               xvel_150(ix, iy, iz) * stencil[1] +
                               xvel_150(ix + 1, iy, iz) * stencil[2],
                           yvel_150(ix - 1, iy, iz) * stencil[0] +
                               yvel_150(ix, iy, iz) * stencil[1] +
                               yvel_150(ix + 1, iy, iz) * stencil[2],
                           zvel_150(ix - 1, iy, iz) * stencil[0] +
                               zvel_150(ix, iy, iz) * stencil[1] +
                               zvel_150(ix + 1, iy, iz) * stencil[2]};
    }
    // calc y
    if (iy == 0) {
      auto const stencil =
          full_domain_grid.diff_stencil_coefficients_0_p1_p2(1, iy);
      J.col(1) = tat::vec3{xvel_150(ix, iy, iz) * stencil[0] +
                               xvel_150(ix, iy + 1, iz) * stencil[1] +
                               xvel_150(ix, iy + 2, iz) * stencil[2],
                           yvel_150(ix, iy, iz) * stencil[0] +
                               yvel_150(ix, iy + 1, iz) * stencil[1] +
                               yvel_150(ix, iy + 2, iz) * stencil[2],
                           zvel_150(ix, iy, iz) * stencil[0] +
                               zvel_150(ix, iy + 1, iz) * stencil[1] +
                               zvel_150(ix, iy + 2, iz) * stencil[2]};
    } else if (iy == full_domain_grid.size(1) - 1) {
      auto const stencil =
          full_domain_grid.diff_stencil_coefficients_n2_n1_0(1, iy);
      J.col(1) = tat::vec3{yvel_150(ix, iy - 2, iz) * stencil[0] +
                               yvel_150(ix, iy - 1, iz) * stencil[1] +
                               yvel_150(ix, iy, iz) * stencil[2],
                           yvel_150(ix, iy - 2, iz) * stencil[0] +
                               yvel_150(ix, iy - 1, iz) * stencil[1] +
                               yvel_150(ix, iy, iz) * stencil[2],
                           zvel_150(ix, iy - 2, iz) * stencil[0] +
                               zvel_150(ix, iy - 1, iz) * stencil[1] +
                               zvel_150(ix, iy, iz) * stencil[2]};
    } else {
      auto const stencil =
          full_domain_grid.diff_stencil_coefficients_n1_0_p1(1, iy);
      J.col(1) = tat::vec3{zvel_150(ix, iy - 1, iz) * stencil[0] +
                               zvel_150(ix, iy, iz) * stencil[1] +
                               zvel_150(ix, iy + 1, iz) * stencil[2],
                           yvel_150(ix, iy - 1, iz) * stencil[0] +
                               yvel_150(ix, iy, iz) * stencil[1] +
                               yvel_150(ix, iy + 1, iz) * stencil[2],
                           zvel_150(ix, iy - 1, iz) * stencil[0] +
                               zvel_150(ix, iy, iz) * stencil[1] +
                               zvel_150(ix, iy + 1, iz) * stencil[2]};
    }
    // calc z
    if (iz == 0) {
      auto const stencil =
          full_domain_grid.diff_stencil_coefficients_0_p1_p2(2, iz);
      J.col(2) = tat::vec3{xvel_150(ix, iy, iz) * stencil[0] +
                               xvel_150(ix, iy, iz + 1) * stencil[1] +
                               xvel_150(ix, iy, iz + 2) * stencil[2],
                           yvel_150(ix, iy, iz) * stencil[0] +
                               yvel_150(ix, iy, iz + 1) * stencil[1] +
                               yvel_150(ix, iy, iz + 2) * stencil[2],
                           zvel_150(ix, iy, iz) * stencil[0] +
                               zvel_150(ix, iy, iz + 1) * stencil[1] +
                               zvel_150(ix, iy, iz + 2) * stencil[2]};
    } else if (iz == full_domain_grid.size(2) - 1) {
      auto const stencil =
          full_domain_grid.diff_stencil_coefficients_n2_n1_0(2, iz);
      J.col(2) = tat::vec3{yvel_150(ix, iy, iz - 2) * stencil[0] +
                               yvel_150(ix, iy, iz - 1) * stencil[1] +
                               yvel_150(ix, iy, iz) * stencil[2],
                           yvel_150(ix, iy, iz - 2) * stencil[0] +
                               yvel_150(ix, iy, iz - 1) * stencil[1] +
                               yvel_150(ix, iy, iz) * stencil[2],
                           zvel_150(ix, iy, iz - 2) * stencil[0] +
                               zvel_150(ix, iy, iz - 1) * stencil[1] +
                               zvel_150(ix, iy, iz) * stencil[2]};
    } else {
      auto const stencil =
          full_domain_grid.diff_stencil_coefficients_n1_0_p1(2, iz);
      J.col(2) = tat::vec3{zvel_150(ix, iy, iz - 1) * stencil[0] +
                               zvel_150(ix, iy, iz) * stencil[1] +
                               zvel_150(ix, iy, iz + 1) * stencil[2],
                           yvel_150(ix, iy, iz - 1) * stencil[0] +
                               yvel_150(ix, iy, iz) * stencil[1] +
                               yvel_150(ix, iy, iz + 1) * stencil[2],
                           zvel_150(ix, iy, iz - 1) * stencil[0] +
                               zvel_150(ix, iy, iz) * stencil[1] +
                               zvel_150(ix, iy, iz + 1) * stencil[2]};
    }
    return J;
  };
  //----------------------------------------------------------------------------
  auto const vortex_core_lines = tat::detail::calc_parallel_vectors<double>(
      [&](auto ix, auto iy, auto iz, auto const& /*p*/) {
        ix += mpi.process_begin(0);
        iy += mpi.process_begin(1);
        iz += mpi.process_begin(2);
        return tat::vec3{xvel_150(ix, iy, iz), yvel_150(ix, iy, iz),
                         zvel_150(ix, iy, iz)};
      },
      [&](auto ix, auto iy, auto iz, auto const& /*p*/) {
        auto const J = calc_J_at(ix, iy, iz);
        ix += mpi.process_begin(0);
        iy += mpi.process_begin(1);
        iz += mpi.process_begin(2);
        tat::vec3 vel{xvel_150(ix, iy, iz), yvel_150(ix, iy, iz),
                      zvel_150(ix, iy, iz)};
        return J * vel;
      },
      halo_grid
      //,[&](auto const& x) {
      // auto const eig = eigenvalues(J_150_field(x, 0));
      // return std::abs(eig(0).imag()) > 0 ||
      // std::abs(eig(1).imag()) > 0 ||
      // std::abs(eig(2).imag()) > 0;
      //}
  );
  write_vtk(vortex_core_lines,
            "vortex_core_lines_channelflow_150000_mpi_rank_" +
                std::to_string(mpi.rank()) + ".vtk");
}
