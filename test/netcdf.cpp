#include <tatooine/netcdf.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
using namespace netCDF::exceptions;
TEST_CASE("netcdf_write_read", "[netcdf][read][write]") {
  std::string const file_path = "simple_xy.nc";
  std::string const xdim_name = "x";
  std::string const ydim_name = "y";
  std::string const data_name = "data";
  // We are reading 2D data, a 6 x 12 grid.
  size_t constexpr NX = 6;
  size_t constexpr NY = 12;

  {  // write block
    
    // create some data
    int dataOut[NX][NY];
    for (size_t i = 0; i < NX; ++i) {
      for (size_t j = 0; j < NY; ++j) { dataOut[i][j] = i * NY + j; }
    }

    try {
      // Create the file. The Replace parameter tells netCDF to overwrite
      // this file, if it already exists.
      netCDF::NcFile data_file{file_path, netCDF::NcFile::replace};

      data_file
          .addVar(
              data_name, netCDF::ncInt,
              {data_file.addDim(xdim_name, NX), data_file.addDim(ydim_name, NY)})
          .putVar(dataOut);

    } catch (NcException& e) {}
  }

  {  // read block
    int dataIn[NX][NY];
    try {
      // Open the file for read access
      netCDF::NcFile data_file{file_path, netCDF::NcFile::read};

      // Retrieve the variable
      auto data = data_file.getVar(data_name);
      REQUIRE(!data.isNull());
      data.getVar(dataIn);

      // Check the values.
      for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NY; ++j) {
          REQUIRE(dataIn[i][j] == i * NY + j);
        }
      }
    } catch (NcException& e) {}
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
