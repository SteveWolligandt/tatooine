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
}  // namespace tatooine::netcdf
//==============================================================================
