#include <Tatooine/grid.h>
#include <Tatooine/interpolators.h>
#include <Tatooine/regular_grid.h>

static std::string dataset_dir = DATASET_DIR;
using real_type                   = double;
using grid_t =
    tatooine::regular_grid_sampler<2, real_type, tatooine::Vec<real_type, 2>,
                                   tatooine::interpolation::hermite,
                                   tatooine::interpolation::hermite>;

//------------------------------------------------------------------------------

auto create_filenames() {
  size_t size  = 40;
  size_t start = 130;
  // size_t size = 4161;

  std::vector<std::string> filenames;
  filenames.reserve(size);
  std::string front = "flapping_wing_";
  std::string ext   = ".vtk";

  for (size_t i = start; i < start + size; ++i)
    filenames.push_back(dataset_dir + "flapping_wing/vtk/" + front +
                        std::to_string(i) + ext);
  std::cout << filenames.front() << '\n';
  return filenames;
}

//------------------------------------------------------------------------------

auto load_grids() {
  auto filenames = create_filenames();
  std::cout << "dadsa\n";
  std::vector<grid_t> grids(filenames.size());
  auto                grid_it = begin(grids);
  for (const auto& file : filenames) {
    std::cout << "reading " << file << "... " << std::flush;
    grid_it->read_vtk(file);
    std::cout << "done!\n";
    ++grid_it;
  }
  return grids;
}

//------------------------------------------------------------------------------

auto load_times() {
  std::vector<real_type> times;

  std::ifstream times_file(dataset_dir + "flapping_wing/vtk/times.txt");
  std::string   line;
  if (times_file) {
    while (std::getline(times_file, line)) times.push_back(stod(line));
    times_file.close();
  } else
    throw std::runtime_error(dataset_dir +
                             "flapping_wing/vtk/times.txt not found");
  return times;
}

//------------------------------------------------------------------------------

int main() {
  // auto times = load_times();
  auto grids = load_grids();
}
