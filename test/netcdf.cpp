#include <tatooine/for_loop.h>
#include <tatooine/grid.h>
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
  // We are reading 2D data, a 4 x 2 grid.
  size_t constexpr NX = 4;
  size_t constexpr NY = 2;

  std::vector<int> data_out(NX * NY);
  // create some data
  for (size_t j = 0; j < NY; ++j) {
    for (size_t i = 0; i < NX; ++i) {
      size_t idx    = i + NX * j;
      data_out[idx] = idx;
    }
  }
  file f_out{file_path, netCDF::NcFile::replace};

  f_out
      .add_variable<int>(variable_name, {f_out.add_dimension(ydim_name, NY),
                                         f_out.add_dimension(xdim_name, NX)})
      .write(data_out);

  file f_in{file_path, netCDF::NcFile::read};
  // Retrieve the variable
  auto var = f_in.variable<int>(variable_name);

  SECTION("read full") {
    auto data_in = var.read();

    // Check the values.
    for (size_t j = 0; j < NY; ++j) {
      for (size_t i = 0; i < NX; ++i) {
        size_t idx = i + NX * j;
        CAPTURE(i, j, idx);
        REQUIRE(data_in(j, i) == data_out[idx]);
      }
    }
  }
  SECTION("read single chunk") {
    auto chunk = var.read_chunk(std::vector<size_t>{0, 0},std::vector<size_t> {2, 2});
    REQUIRE(chunk(0, 0) == 0);
    REQUIRE(chunk(1, 0) == 1);
    REQUIRE(chunk(0, 1) == 4);
    REQUIRE(chunk(1, 1) == 5);
  }

  SECTION("read full") {
    auto data_in = var.read_chunked(std::vector<size_t>{2, 2});

    // Check the values.
    for (size_t j = 0; j < NY; ++j) {
      for (size_t i = 0; i < NX; ++i) {
        size_t idx = i + NX * j;
        CAPTURE(i, j, idx);
        REQUIRE(data_in(i, j) == data_out[idx]);
      }
    }
  }
  SECTION("read chunk-wise") {
    auto data_in = var.read_chunked({2, 2});

    // Check the values.
    for (size_t j = 0; j < NY; ++j) {
      for (size_t i = 0; i < NX; ++i) {
        size_t idx = i + NX * j;
        CAPTURE(i, j, idx);
        REQUIRE(data_in(i, j) == data_out[idx]);
      }
    }
  }
}
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
