#include <tatooine/hdf5.h>
#include <tatooine/grid.h>
#include <iomanip>
//==============================================================================
namespace tat = tatooine;
//==============================================================================
auto main() -> int {
  // read full domain axes
  // open hdf5 files
  tat::hdf5::file channelflow_122_threedpart_file{
      "/home/vcuser/channel_flow/dino_res_122000.h5"};
  tat::hdf5::file channelflow_122_full_file{
      "/home/vcuser/channel_flow/dino_res_122000_full.h5"};
  auto velx_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/xvel");
  auto velx_122_threedpart_dataset =
      channelflow_122_threedpart_file.dataset<double>("variables/Vx");
  auto const full_res_x = velx_122_full_dataset.size(0);
  auto const full_res_y = velx_122_full_dataset.size(1);
  auto const full_res_z = velx_122_full_dataset.size(2);
  auto const part_res_x = velx_122_threedpart_dataset.size(0);
  auto const part_res_y = velx_122_threedpart_dataset.size(1);
  auto const part_res_z = velx_122_threedpart_dataset.size(2);
  std::cerr << "full resolution: [" << full_res_x << ", " << full_res_y << ", "
            << full_res_z << "]\n";
  std::cerr << "3dpart resolution: [" << part_res_x << ", " << part_res_y << ", "
            << part_res_z << "]\n";
  auto const axis0_full = tat::hdf5::file{"/home/vcuser/channel_flow/axis0.h5"}
                              .dataset<double>("CartGrid/axis0")
                              .read_as_vector();
  auto const axis1_full = tat::hdf5::file{"/home/vcuser/channel_flow/axis1.h5"}
                              .dataset<double>("CartGrid/axis1")
                              .read_as_vector();
  auto const axis2_full = tat::hdf5::file{"/home/vcuser/channel_flow/axis2.h5"}
                              .dataset<double>("CartGrid/axis2")
                              .read_as_vector();
  tat::grid full_domain{axis0_full, axis1_full, axis2_full};
  std::cerr << "full_domain:\n" << full_domain << '\n';
  auto const axis0_part =
      std::vector<double>(begin(axis0_full), begin(axis0_full) + part_res_x);
  auto const axis1_part =
      std::vector<double>(begin(axis1_full), begin(axis1_full) + part_res_y);
  auto const axis2_part =
      std::vector<double>(begin(axis2_full), begin(axis2_full) + part_res_z);
  tat::grid part_domain{axis0_part, axis1_part, axis2_part};
  std::cerr << "3dpart_domain:\n" << part_domain << '\n';

  auto const velx_122_full = velx_122_full_dataset.read_chunk(
      std::vector<size_t>{0, 0, 0},
      std::vector{part_res_x, part_res_y, part_res_z}).data();
   auto const velx_122_threedpart =
   velx_122_threedpart_dataset.read_as_vector();

   std::cout <<
     "┌──────────────────────────────────────────────────────────────────────────┐\n"
     "│  full plain data                                                         │\n"
     "├────────────────────────┬────────────────────────┬────────────────────────┤\n"
     "│          full          │         3dpart         │       difference       │\n"
     "├────────────────────────┼────────────────────────┼────────────────────────┤\n";
   double acc_err=0;
   for (size_t i = 0; i < size(velx_122_full); ++i) {
     auto const err = std::abs(velx_122_full[i] - velx_122_threedpart[i]);
     acc_err += err;
     // std::cout << std::left << "│   " << std::setw(21) << std::scientific
     //          << velx_122_full[i] << "│   " << std::setw(21) <<
     //          std::scientific
     //          << velx_122_threedpart[i] << "│   " << std::setw(21)
     //          << std::scientific << err << "│\n";
   }
   std::cout << "└────────────────────────┴────────────────────────┴───────────"
                "─────────────┘\n";

   std::cout << "accumulated error: " << acc_err << '\n';
   {
     auto const velx_122_full_chunk = velx_122_full_dataset.read_chunk(
         std::vector<size_t>{0, 0, 0}, std::vector<size_t>{2, 2, 2});
     auto const velx_122_threedpart_chunk =
         velx_122_threedpart_dataset.read_chunk(std::vector<size_t>{0, 0, 0},
                                                std::vector<size_t>{2, 2, 2});

     std::cout <<
       "┌──────────────────────────────────────────────────────────────────────────┐\n"
       "│  "<<std::setw(72)<<"chunk {0 ,0 ,0}, {2, 2, 2}"<<"│\n"
       "├────────────────────────┬────────────────────────┬────────────────────────┤\n"
       "│          full          │         3dpart         │       difference       │\n"
       "├────────────────────────┼────────────────────────┼────────────────────────┤\n";
       for (size_t i = 0; i < 8; ++i) {
         std::cout << std::left << "│   " << std::setw(21) << std::scientific
                   << velx_122_full_chunk.data()[i] << "│   " << std::setw(21)
                   << std::scientific << velx_122_threedpart_chunk.data()[i] << "│   "
                   << std::setw(21) << std::scientific
                   << std::abs(velx_122_full_chunk.data()[i] - velx_122_threedpart_chunk.data()[i])
                   << "│\n";
       }
    std::cout <<
       "└────────────────────────┴────────────────────────┴────────────────────────┘\n";
  }{
    auto const velx_122_full_chunk = velx_122_full_dataset.read_chunk(
        std::vector<size_t>{16, 16, 16}, std::vector<size_t>{2, 2, 2});
    auto const velx_122_threedpart_chunk = velx_122_threedpart_dataset.read_chunk(
        std::vector<size_t>{16, 16, 16}, std::vector<size_t>{2, 2, 2});

     std::cout <<
       "┌──────────────────────────────────────────────────────────────────────────┐\n"
       "│  "<<std::setw(72)<<"chunk {16, 16, 16}, {2, 2, 2}"<<"│\n"
       "├────────────────────────┬────────────────────────┬────────────────────────┤\n"
       "│          full          │         3dpart         │       difference       │\n"
       "├────────────────────────┼────────────────────────┼────────────────────────┤\n";
       for (size_t i = 0; i < 8; ++i) {
         std::cout << std::left << "│   " << std::setw(21) << std::scientific
                   << velx_122_full_chunk.data()[i] << "│   " << std::setw(21)
                   << std::scientific << velx_122_threedpart_chunk.data()[i] << "│   "
                   << std::setw(21) << std::scientific
                   << std::abs(velx_122_full_chunk.data()[i] - velx_122_threedpart_chunk.data()[i])
                   << "│\n";
       }
    std::cout <<
       "└────────────────────────┴────────────────────────┴────────────────────────┘\n";
  }{
    auto const velx_122_full_chunk = velx_122_full_dataset.read_chunk(
        std::vector<size_t>{part_res_x - 3, part_res_y - 3, part_res_z - 3},
        std::vector<size_t>{2, 2, 2});
    auto const velx_122_threedpart_chunk = velx_122_threedpart_dataset.read_chunk(
        std::vector<size_t>{part_res_x-3, part_res_y-3, part_res_z-3}, std::vector<size_t>{2, 2, 2});

     std::cout <<
       "┌──────────────────────────────────────────────────────────────────────────┐\n"
       "│  " << std::setw(72) << "chunk last 3dpart, {2, 2, 2}" <<                "│\n"
       "├────────────────────────┬────────────────────────┬────────────────────────┤\n"
       "│          full          │         3dpart         │       difference       │\n"
       "├────────────────────────┼────────────────────────┼────────────────────────┤\n";
       for (size_t i = 0; i < 8; ++i) {
         std::cout << std::left << "│   " << std::setw(21) << std::scientific
                   << velx_122_full_chunk.data()[i] << "│   " << std::setw(21)
                   << std::scientific << velx_122_threedpart_chunk.data()[i] << "│   "
                   << std::setw(21) << std::scientific
                   << std::abs(velx_122_full_chunk.data()[i] - velx_122_threedpart_chunk.data()[i])
                   << "│\n";
       }
    std::cout <<
       "└────────────────────────┴────────────────────────┴────────────────────────┘\n";
  }{
    auto const velx_122_full_chunk = velx_122_full_dataset.read_chunk(
        std::vector<size_t>{full_res_x-3, full_res_y-3, full_res_z-3}, std::vector<size_t>{2, 2, 2});

     std::cout <<
       "┌────────────────────────┐\n"
       "│  " << std::setw(72) << "chunk last 3dpart, {2, 2, 2}" <<                "│\n"
       "├────────────────────────┤\n"
       "│          full          │\n"
       "├────────────────────────┤\n";
       for (size_t i = 0; i < 8; ++i) {
         std::cout << std::left << "│   " << std::setw(21) << std::scientific
                   << velx_122_full_chunk.data()[i] << "│\n";
       }
    std::cout <<
       "└────────────────────────┘\n";
  }
}
