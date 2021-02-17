#include <tatooine/insitu/mpi_program.h>
namespace tat = tatooine;
auto main(int argc, char** argv) -> int {
  auto& mpi = tat::insitu::mpi_program::get(argc, argv);
  mpi.init_communicator(512, 1024, 256);

  
}
