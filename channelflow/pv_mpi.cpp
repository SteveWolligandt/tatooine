#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/mpi/program.h>
#include <tatooine/differentiated_field.h>
// #include <tatooine/parallel_vectors.h>
#include <tatooine/spacetime_vectorfield.h>

namespace tat = tatooine;
using doublegyre = tat::analytical::fields::numerical::doublegyre<tat::real_t>;
using tat::linspace;

auto main(int argc, char** argv) -> int {
  auto& mpi = tat::mpi::program::get(argc, argv);
  constexpr auto eps = 1e-5;
  tat::uniform_grid<tat::real_t, 3> global_grid{
      linspace{0.0 - eps,  2.0 + eps,  201},
      linspace{0.0 - eps,  1.0 + eps,  101},
      linspace{0.0 - eps, 10.0 + eps, 1001}};
  mpi.init_communicator(global_grid);
  auto const local_grid = mpi.local_grid(global_grid);
  auto const halo_grid  = [&local_grid, &global_grid] {
    auto halo_grid = local_grid;
    if (local_grid.dimension<0>().back() !=
        global_grid.dimension<0>().back()) {
      halo_grid.dimension<0>().push_back();
    }
    if (local_grid.dimension<1>().back() !=
        global_grid.dimension<1>().back()) {
      halo_grid.dimension<1>().push_back();
    }
    if (local_grid.dimension<2>().back() !=
        global_grid.dimension<2>().back()) {
      halo_grid.dimension<2>().push_back();
    }
    return halo_grid;
  }();
  if (mpi.rank() == 0) {
    std::cout << "global grid\n" << global_grid << '\n';
    std::cout << "local grid\n" << local_grid << '\n';
    std::cout << "halo grid\n" << halo_grid;
  }
  //------------------------------------------------------------------------------
  doublegyre dg;
  auto       v = spacetime(dg);
  //auto       J = diff(v, 1e-10);
  //auto       a = J * v;

  //auto const pv = parallel_vectors(v, a, halo_grid);
  //write_vtk(pv, "pv_stdg_mpi_rank_" + std::to_string(mpi.rank()) + ".vtk");
}
