#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/lazy_netcdf_reader.h>
#include <tatooine/netcdf.h>
#include "ensemble_file_paths.h"

int main() {
  using namespace tatooine;
  using namespace tatooine::scivis_contest_2020;
  auto f           = netcdf::file{ensemble_file_paths[0], netCDF::NcFile::read};
  auto t_ax_var    = f.variable<double>("T_AX");
  auto z_mit40_var = f.variable<double>("Z_MIT40");
  auto xc_var      = f.variable<double>("XC");
  auto yc_var      = f.variable<double>("YC");
  auto salt_var    = f.variable<double>("SALT");


  linspace<double>    xc_axis = linspace{xc_var.read_single(0), xc_var.read_single(499), 500};
  linspace<double>    yc_axis = linspace{yc_var.read_single(0), yc_var.read_single(499), 500};
  std::vector<double> z_axis  = z_mit40_var.read_as_vector();
  linspace<double>    t_axis  = linspace{t_ax_var.read_single(0), t_ax_var.read_single(59), 60};

  grid salt_grid_in{xc_axis, yc_axis, z_axis, t_axis};
  grid salt_grid_out{xc_axis, yc_axis, z_axis};
  for (auto& z : salt_grid_out.dimension<2>()) { z *= -0.0025; }

  size_t const chunk_size = 10;
  auto&        salt_in =
      salt_grid_in.add_vertex_property<netcdf::lazy_reader<double>, interpolation::hermite,
                            interpolation::hermite, interpolation::hermite,
                            interpolation::linear>(
          "salt", salt_var, std::vector<size_t>(4, chunk_size));
  auto& salt_out = salt_grid_out.add_contiguous_vertex_property<double>("salt");

  size_t cnt = 0; 
  for (size_t t = 0; t < salt_grid_in.size<3>(); ++t) {
    std::cerr << "processing time index " << t << '\n';
    salt_grid_out.loop_over_vertex_indices([&](auto const... is) {
      salt_out.data_at(is...) = salt_in(is..., t);
      ++cnt;
      std::cerr << (double)(cnt) / salt_grid_out.num_vertices()
                << "             \r";
    });
    std::cerr << "\n";
    salt_grid_out.write_vtk("salt_" + std::to_string(t) + ".vtk");
  }
}
