#include <tatooine/netcdf.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::netcdf {
//==============================================================================
using namespace netCDF::exceptions;
TEST_CASE("netcdf_write_read", "[netcdf][read][write]") {
  std::string const file_path     = "simple_xy.nc";
  std::string const xdim_name     = "x";
  std::string const ydim_name     = "y";
  std::string const variable_name = "data";
  // We are reading 2D data, a 6 x 12 grid.
  size_t constexpr NX = 6;
  size_t constexpr NY = 12;

  std::vector<int> data_out(NX * NY);
  // create some data
  for (size_t i = 0; i < NX; ++i) {
    for (size_t j = 0; j < NY; ++j) {
      size_t idx    = i * NY + j;
      data_out[idx] = idx;
    }
  }
  file f_out{file_path, netCDF::NcFile::replace};

  f_out
      .add_variable<int>(variable_name, {f_out.add_dimension(xdim_name, NX),
                                         f_out.add_dimension(ydim_name, NY)})
      .put(data_out);

  file f_in{file_path, netCDF::NcFile::read};
  // Retrieve the variable
  auto var = f_in.get_variable<int>(variable_name);
  REQUIRE(!var.is_null());
  auto data_in = var.to_vector();

  // Check the values.
  for (size_t i = 0; i < NX; ++i) {
    for (size_t j = 0; j < NY; ++j) {
      size_t idx = i * NY + j;
      REQUIRE(data_in[idx] == data_out[idx]);
    }
  }
}
//==============================================================================
TEST_CASE("netcdf_scivis", "[netcdf][scivis][general]") {
  std::string const file_path = "/home/steve/Downloads/MEAN/2011013100.nc";
  file              f{file_path, netCDF::NcFile::read};
  std::cerr << "attributes:\n";
  for (auto const& [key, val] : f.attributes()) { std::cerr << key << '\n'; }
  std::cerr << "dimensions:\n";
  for (auto const& [key, val] : f.dimensions()) { std::cerr << key << '\n'; }
  std::cerr << "groups:\n";
  for (auto const& [key, val] : f.groups()) { std::cerr << key << '\n'; }
  std::cerr << "variables:\n";
  for (auto const& var : f.variables<double>()) { std::cerr << var.name() << '\n'; }
}
//==============================================================================
TEST_CASE("netcdf_scivis_u", "[netcdf][scivis][u]") {
  std::string const file_path = "/home/steve/Downloads/MEAN/2011013100.nc";
  auto u = file{file_path, netCDF::NcFile::read}.variable<double>("u");
  u.read_chunk({0,0,0}, {10,10,10}, );
}
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
