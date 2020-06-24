#include <tatooine/for_loop.h>
#include <tatooine/grid.h>
#include <tatooine/netcdf.h>
#include <tatooine/lazy_netcdf_reader.h>

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
  // We are reading 2D data, a 8 x 6 grid.
  size_t constexpr NX = 8;
  size_t constexpr NY = 6;

  std::vector<int> data_out(NX * NY);
  // create some data
  for (size_t j = 0; j < NY; ++j) {
    for (size_t i = 0; i < NX; ++i) {
        size_t idx = i + NX * j;
      data_out[idx] = idx;
    }
  }
  data_out[4 + NX * 0] = 0;
  data_out[5 + NX * 0] = 0;
  data_out[4 + NX * 1] = 0;
  data_out[5 + NX * 1] = 0;

  file f_out{file_path, netCDF::NcFile::replace};
  auto dim_x = f_out.add_dimension(xdim_name, NX);
  auto dim_y = f_out.add_dimension(ydim_name, NY);
  f_out.add_variable<int>(variable_name, {dim_y, dim_x}).write(data_out);

  file f_in{file_path, netCDF::NcFile::read};
  // Retrieve the variable
  auto var = f_in.variable<int>(variable_name);
  REQUIRE(var.size(1) == NX);
  REQUIRE(var.size(0) == NY);

  std::cerr << var.dimension_name(0) << '\n';
  std::cerr << var.dimension_name(1) << '\n';
  SECTION("read full") {
    auto data_in = var.read();
    REQUIRE(data_in.size(0) == NX);
    REQUIRE(data_in.size(1) == NY);
    std::cerr << "full: ";
    for (auto d: data_in.data()) std::cerr << d << ' ';
    std::cerr << '\n';

    // Check the values.
    for (size_t j = 0; j < NY; ++j) {
      for (size_t i = 0; i < NX; ++i) {
        size_t idx = i + NX * j;
        CAPTURE(i, j, idx);
        REQUIRE(data_in(i, j) == data_out[idx]);
      }
    }
  }
  SECTION("read single chunk") {
    SECTION("0,0|2,2") {
      auto chunk =
          var.read_chunk(std::vector<size_t>{0, 0}, std::vector<size_t>{2, 2});
      std::cerr << "chunk: ";
      for (auto d : chunk.data()) std::cerr << d << ' ';
      std::cerr << '\n';
      REQUIRE(chunk(0, 0) == 0);
      REQUIRE(chunk(1, 0) == 1);
      REQUIRE(chunk(0, 1) == 8);
      REQUIRE(chunk(1, 1) == 9);
    }
    SECTION("2,0|2,2") {
      auto chunk =
          var.read_chunk(std::vector<size_t>{2, 0}, std::vector<size_t>{2, 2});
      std::cerr << "chunk: ";
      for (auto d : chunk.data()) std::cerr << d << ' ';
      std::cerr << '\n';
      REQUIRE(chunk(0, 0) == 2);
      REQUIRE(chunk(1, 0) == 3);
      REQUIRE(chunk(0, 1) == 10);
      REQUIRE(chunk(1, 1) == 11);
    }
    SECTION("0,2|2,2") {
      auto chunk =
          var.read_chunk(std::vector<size_t>{0, 2}, std::vector<size_t>{2, 2});
      std::cerr << "chunk: ";
      for (auto d : chunk.data()) std::cerr << d << ' ';
      std::cerr << '\n';
      REQUIRE(chunk(0, 0) == 16);
      REQUIRE(chunk(1, 0) == 17);
      REQUIRE(chunk(0, 1) == 24);
      REQUIRE(chunk(1, 1) == 25);
    }
    SECTION("2,2|2,2") {
      auto chunk =
          var.read_chunk(std::vector<size_t>{2, 2}, std::vector<size_t>{2, 2});
      std::cerr << "chunk: ";
      for (auto d : chunk.data()) std::cerr << d << ' ';
      std::cerr << '\n';
      REQUIRE(chunk(0, 0) == 18);
      REQUIRE(chunk(1, 0) == 19);
      REQUIRE(chunk(0, 1) == 26);
      REQUIRE(chunk(1, 1) == 27);
    }
    SECTION("0,0|3,2") {
      auto chunk =
          var.read_chunk(std::vector<size_t>{0, 0}, std::vector<size_t>{3, 2});
      std::cerr << "chunk: ";
      for (auto d : chunk.data()) std::cerr << d << ' ';
      std::cerr << '\n';
      REQUIRE(chunk(0, 0) == 0);
      REQUIRE(chunk(1, 0) == 1);
      REQUIRE(chunk(2, 0) == 2);
      REQUIRE(chunk(0, 1) == 8);
      REQUIRE(chunk(1, 1) == 9);
      REQUIRE(chunk(2, 1) == 10);
    }
  }

  SECTION("read chunk-wise") {
    auto data_in = var.read_chunked({2, 2});
    REQUIRE(data_in.internal_chunk_size(0) == 2);
    REQUIRE(data_in.internal_chunk_size(1) == 2);
    REQUIRE(data_in.size(0) == NX);
    REQUIRE(data_in.size(1) == NY);

    std::cerr << "chunk 0: ";
    for (auto d : data_in.chunk_at(0)->data()) std::cerr << d << ' ';
    std::cerr << '\n';
    std::cerr << "chunk 1: ";
    for (auto d : data_in.chunk_at(1)->data()) std::cerr << d << ' ';
    std::cerr << '\n';
    std::cerr << "chunk 2: ";
    REQUIRE(data_in.chunk_at_is_null(2));
    std::cerr << "chunk 3: ";
    for (auto d : data_in.chunk_at(3)->data()) std::cerr << d << ' ';
    std::cerr << '\n';

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
TEST_CASE("netcdf_lazy","[netcdf][lazy]"){
  std::string const file_path     = "simple_xy.nc";
  std::string const variable_name = "data";
  lazy_reader<int> cont{file_path, variable_name, std::vector<size_t>{2, 2}};

  REQUIRE(cont.chunk_at_is_null(0));
  REQUIRE(cont(0, 0) == 0);
  REQUIRE(!cont.chunk_at_is_null(0));

  REQUIRE(cont.chunk_at_is_null(1));
  REQUIRE(cont(2, 0) == 2);
  REQUIRE(!cont.chunk_at_is_null(1));

  REQUIRE(cont.chunk_at_is_null(2));
  REQUIRE(cont(4, 0) == 0);
  REQUIRE(cont.chunk_at_is_null(2));
  REQUIRE(cont(5, 0) == 0);
  REQUIRE(cont.chunk_at_is_null(2));
}
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
