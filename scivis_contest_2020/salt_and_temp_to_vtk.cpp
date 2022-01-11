#include <tatooine/field.h>
#include <tatooine/filesystem.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/lazy_netcdf_reader.h>
#include <tatooine/netcdf.h>

#include <chrono>
#include <thread>

#include "ensemble_file_paths.h"

int main(int , char const** argv) {
  size_t const ensemble_index = std::stoi(argv[1]);
  filesystem::path dir_path{"salt_and_temp_" + argv[1]};
  if (!filesystem::exists(dir_path)) {
    filesystem::create_directory(dir_path);
  }
  using namespace tatooine;
  using namespace tatooine::scivis_contest_2020;
  auto f =
      netcdf::file{ensemble_file_paths[ensemble_index], netCDF::NcFile::read};
  auto t_ax_var    = f.variable<double>("T_AX");
  auto z_mit40_var = f.variable<double>("Z_MIT40");
  auto xc_var      = f.variable<double>("XC");
  auto yc_var      = f.variable<double>("YC");
  auto salt_var    = f.variable<double>("SALT");
  auto temp_var    = f.variable<double>("TEMP");

  linspace<double> xc_axis =
      linspace{xc_var.read_single(0), xc_var.read_single(499), xc_var.size(0)};
  linspace<double> yc_axis =
      linspace{yc_var.read_single(0), yc_var.read_single(499), yc_var.size(0)};
  std::vector<double> z_axis = z_mit40_var.read_as_vector();
  linspace<double>    t_axis = linspace{
      t_ax_var.read_single(0), t_ax_var.read_single(59), t_ax_var.size(0)};

  rectilinear_grid grid_in{xc_axis, yc_axis, z_axis, t_axis};
  rectilinear_grid grid_out{xc_axis, yc_axis, z_axis};
  for (auto& z : grid_out.dimension<2>()) { z *= -0.0025; }

  size_t const chunk_size = 10;
  auto& salt_out = grid_out.add_contiguous_vertex_property<double>("salt");
  auto& salt_in  = grid_in.add_vertex_property<
      netcdf::lazy_reader<double>, interpolation::hermite,
      interpolation::hermite, interpolation::hermite, interpolation::linear>(
      "s", salt_var, std::vector<size_t>(4, chunk_size));
  auto& temp_out =
      grid_out.add_contiguous_vertex_property<double>("temperature");
  auto& temp_in = grid_in.add_vertex_property<
      netcdf::lazy_reader<double>, interpolation::hermite,
      interpolation::hermite, interpolation::hermite, interpolation::linear>(
      "t", temp_var, std::vector<size_t>(4, chunk_size));

  for (size_t t = 0; t < size(t_axis); ++t) {
    std::atomic_size_t cnt = 0;
    bool done = false;
    std::cerr << "\rprocessing time index " << t << " (" << t_axis[t] << ")"
              << '\n';
    std::thread        monitor{[&] {
      while (!done) {
        std::cerr << (double)(cnt) / (grid_out.vertices().size()*2) * 100
                  << "%             \r";
        std::this_thread::sleep_for(std::chrono::milliseconds{200});
      }
    }};
    std::thread salt_thread([&] {
    grid_out.iterate_over_vertex_indices(
        [t, &salt_in, &salt_out, &cnt](auto const... is) {
          salt_out.data_at(is...) = salt_in.data_at(is..., t);
          ++cnt;
        });
    });
    std::thread temp_thread([&] {
      grid_out.iterate_over_vertex_indices(
          [t, &temp_in, &temp_out, &cnt](auto const... is) {
            temp_out.data_at(is...) = temp_in.data_at(is..., t);
            ++cnt;
          });
    });
    salt_thread.join();
    temp_thread.join();
    done = true;
    std::cerr << "100%\n";
    monitor.join();
    std::cerr << "writing time step...";
    grid_out.write_vtk(std::string{dir_path.c_str()} + "/salt_temp_" +
                           std::to_string(t) + ".vtk",
                       "salt and temperature of red sea");
    std::cerr << "\n";
  }
}
