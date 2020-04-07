#include "datasets.h"
#include <fstream>
#include <vector>
//============================================================================
namespace tatooine::steadification {
//============================================================================
void rbc::read_from_binary() {
  grids.reserve(dim[2]);
  for (size_t ti = 0; ti < dim[2]; ++ti) {
    std::stringstream ss;
    ss << domain.dimension(2)[ti];
    const std::string filename =
      dataset_dir + "RBC/binary/rbc_" + ss.str() + ".bin";
    grids.emplace_back(domain.dimension(0), domain.dimension(1));


    std::ifstream file(filename, std::ifstream::binary);
    if (file.is_open()) {
      std::vector<vec<double, 2>> data(dim[0] * dim[1]);
      // std::cout << "reading: " << filename <<'\n';
      constexpr auto num_bytes = sizeof(double) * dim[0] * dim[1] * 2;
      file.read((char*)(data.data()), num_bytes);
      file.close();

      grids.back().data() = data;
    } else {throw std::runtime_error{"could not open " + filename};}
  }
}

//------------------------------------------------------------------------------
auto rbc::evaluate(const rbc::pos_t& pos, rbc::real_t t) const -> tensor_t {
  const auto& times = domain.dimension(2);
  for (size_t i = 0; i < grids.size() - 1; ++i)
    if (times[i] <= t && t <= times[i + 1]) {
      real_t f = (t - times[i]) / (times[i + 1] - times[i]);
      return (1 - f) * grids[i](pos(0), pos(1)) +
             f * grids[i + 1](pos(0), pos(1));
    }
  return tensor_t{0,0};
}

//==============================================================================
// FlappingWing::vec_t FlappingWing::evaluate(const FlappingWing::pos_t& pos,
//                                           FlappingWing::real_t       t) const
//                                           {
//  for (size_t i = 0; i < grids.size() - 1; ++i)
//    if (times[i] <= t && t <= times[i + 1]) {
//      real_t f = (t - times[i]) / (times[i + 1] - times[i]);
//      return (1 - f) * grids[i](pos(0), pos(1)) +
//             f * grids[i + 1](pos(0), pos(1));
//    }
//  return {0,0};
//}
//
////------------------------------------------------------------------------------
// bool FlappingWing::in_domain(const FlappingWing::vec_t& p, real_t t) const {
//  return times.front() <= t && t <= times.back() &&
//         grids.front().in_domain(p(0), p(1));
//}
//
////------------------------------------------------------------------------------
// void FlappingWing::load() {
//  std::cerr << "loading flapping wing dataset... ";
//  load_times();
//  load_data();
//  std::cerr << "done!\n";
//}
//
////------------------------------------------------------------------------------
// void FlappingWing::load_times() {
//  std::ifstream times_file(dataset_dir + "flapping_wing/vtk/times.txt");
//  std::string   line;
//  if (times_file) {
//    while (std::getline(times_file, line)) times.push_back(stod(line));
//    times_file.close();
//
//    std::vector<real_t>lower_res_times;
//    for (size_t i = 0; i < times.size(); i+=100)
//      lower_res_times.push_back(times[i]);
//    lower_res_times.push_back(times.back());
//    times = lower_res_times;
//  } else
//    throw std::runtime_error(dataset_dir +
//                             "flapping_wing/vtk/times.txt not found");
//}
//
////------------------------------------------------------------------------------
// void FlappingWing::load_data() {
//  grids.reserve(filenames.size());
//  create_filenames();
//  for (size_t i = 0; i < filenames.size(); i+=100) {
//    // std::cout << "reading " << filenames[i] << "... " << std::flush;
//    grids.emplace_back().read_vtk(filenames[i]);
//    // std::cout << "done!\n";
//  }
//  // std::cout << "reading " << filenames.back() << "... " << std::flush;
//  grids.emplace_back().read_vtk(filenames.back());
//  // std::cout << "done!\n";
//}
//
////------------------------------------------------------------------------------
// void FlappingWing::create_filenames() {
//  std::string front = "flapping_wing_";
//  std::string ext   = ".vtk";
//
//  for (size_t i = 0; i <= 4160; ++i)
//    filenames.push_back(dataset_dir + "flapping_wing/vtk/" + front +
//                        std::to_string(i) + ext);
//}
//============================================================================
}  // namespace tatooine
//============================================================================
