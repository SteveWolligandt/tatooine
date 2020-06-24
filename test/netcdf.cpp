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
  // We are reading 2D data, a 4 x 4 grid.
  size_t constexpr NX = 4;
  size_t constexpr NY = 4;

  std::vector<int> data_out(NX * NY);
  // create some data
  for (size_t j = 0; j < NY; ++j) {
    for (size_t i = 0; i < NX; ++i) {
        size_t idx = i + NX * j;
      data_out[idx] = idx;
    }
  }
  file f_out{file_path, netCDF::NcFile::replace};

  f_out
      .add_variable<int>(variable_name, {f_out.add_dimension(xdim_name, NX),
                                         f_out.add_dimension(ydim_name, NY)})
      .write(data_out);

  file f_in{file_path, netCDF::NcFile::read};
  // Retrieve the variable
  auto var = f_in.variable<int>(variable_name);

  SECTION("read full") {
    auto data_in = var.read();
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
      REQUIRE(chunk(0, 1) == 4);
      REQUIRE(chunk(1, 1) == 5);
    }
    SECTION("2,0|2,2") {
      auto chunk =
          var.read_chunk(std::vector<size_t>{2, 0}, std::vector<size_t>{2, 2});
      std::cerr << "chunk: ";
      for (auto d : chunk.data()) std::cerr << d << ' ';
      std::cerr << '\n';
      REQUIRE(chunk(0, 0) == 2);
      REQUIRE(chunk(1, 0) == 3);
      REQUIRE(chunk(0, 1) == 6);
      REQUIRE(chunk(1, 1) == 7);
    }
    SECTION("0,2|2,2") {
      auto chunk =
          var.read_chunk(std::vector<size_t>{0, 2}, std::vector<size_t>{2, 2});
      std::cerr << "chunk: ";
      for (auto d : chunk.data()) std::cerr << d << ' ';
      std::cerr << '\n';
      REQUIRE(chunk(0, 0) == 8);
      REQUIRE(chunk(1, 0) == 9);
      REQUIRE(chunk(0, 1) == 12);
      REQUIRE(chunk(1, 1) == 13);
    }
    SECTION("2,2|2,2") {
      auto chunk =
          var.read_chunk(std::vector<size_t>{2, 2}, std::vector<size_t>{2, 2});
      std::cerr << "chunk: ";
      for (auto d : chunk.data()) std::cerr << d << ' ';
      std::cerr << '\n';
      REQUIRE(chunk(0, 0) == 10);
      REQUIRE(chunk(1, 0) == 11);
      REQUIRE(chunk(0, 1) == 14);
      REQUIRE(chunk(1, 1) == 15);
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
      REQUIRE(chunk(0, 1) == 4);
      REQUIRE(chunk(1, 1) == 5);
      REQUIRE(chunk(2, 1) == 6);
    }
  }

  SECTION("read chunk-wise") {
    auto data_in = var.read_chunked({2, 2});
    REQUIRE(data_in.internal_chunk_size(0) == 2);
    REQUIRE(data_in.internal_chunk_size(1) == 2);
    REQUIRE(data_in.size(1) == 4);
    REQUIRE(data_in.size(0) == 4);
    REQUIRE(data_in.size(1) == 4);

      std::cerr << "chunk 0: ";
      for (auto d : data_in.chunk_at(0)->data()) std::cerr << d << ' ';
      std::cerr << '\n';
      std::cerr << "chunk 1: ";
      for (auto d : data_in.chunk_at(1)->data()) std::cerr << d << ' ';
      std::cerr << '\n';
      std::cerr << "chunk 2: ";
      for (auto d : data_in.chunk_at(2)->data()) std::cerr << d << ' ';
      std::cerr << '\n';
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
}  // namespace tatooine::netcdf
//==============================================================================
