#include <tatooine/for_loop.h>
#include <tatooine/grid.h>
#include <tatooine/lazy_netcdf_reader.h>
#include <tatooine/netcdf.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::netcdf {
//==============================================================================
size_t constexpr NX                       = 8;
size_t constexpr NY                       = 6;
size_t constexpr NZ                       = 4;
size_t constexpr NT                       = 6;
std::string const variable_name           = "DATA";
std::string const xdim_name               = "X";
std::string const ydim_name               = "Y";
std::string const zdim_name               = "Z";
std::string const tdim_name               = "T";
std::string const row_dim_name            = "ROWS";
std::string const col_dim_name            = "COLS";
std::string const file_path_xy            = "simple_xy.nc";
std::string const file_unlimited_mat_list = "unlimited_mat_list.nc";
std::string const file_path_xyz           = "simple_xyz.nc";
std::string const file_path_xyzt          = "simple_xyzt.nc";
using namespace netCDF::exceptions;
//==============================================================================
auto write_simple_xy() {
  // 2D data, a 8 x 6 grid.
  std::vector<double> data_out(NX * NY);
  // create some data
  for (size_t j = 0; j < NY; ++j) {
    for (size_t i = 0; i < NX; ++i) {
      size_t idx    = i + NX * j;
      data_out[idx] = idx;
    }
  }
  data_out[4 + NX * 0] = 0;
  data_out[5 + NX * 0] = 0;
  data_out[4 + NX * 1] = 0;
  data_out[5 + NX * 1] = 0;

  file f_out{file_path_xy, netCDF::NcFile::replace};
  auto dim_x = f_out.add_dimension(xdim_name, NX);
  auto dim_y = f_out.add_dimension(ydim_name, NY);
  f_out.add_variable<double>(variable_name, {dim_y, dim_x}).write(data_out);
  return data_out;
}
//------------------------------------------------------------------------------
auto write_unlimited_mat_list() {
  std::vector<mat2> data_out;

  file f_out{file_unlimited_mat_list, netCDF::NcFile::replace};
  auto dim_t    = f_out.add_dimension(tdim_name);
  auto dim_cols = f_out.add_dimension(col_dim_name, 2);
  auto dim_rows = f_out.add_dimension(row_dim_name, 2);
  auto var =
      f_out.add_variable<double>(variable_name, {dim_t, dim_rows, dim_cols});
  
  // create some data
  random_uniform<double>    rand;
  std::vector<size_t>       is{0, 0, 0};
  std::vector<size_t> const cnt{1, 2, 2};
  for (; is.front() < 10; ++is.front()) {
    data_out.push_back(mat2{{is.front() * 4, is.front() * 4 + 2},
                            {is.front() * 4 + 1, is.front() * 4 + 3}});
    // data_out.push_back(mat2{{rand(), rand()}, {rand(), rand()}});
    var.write(is, cnt, data_out.back().data_ptr());
  }

  return data_out;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto write_simple_xyz() {
  // 3D data, a 8 x 6 x 4 grid.
  std::vector<double> data_out(NX * NY * NZ);
  // create some data
  for (size_t k = 0; k < NZ; ++k) {
    for (size_t j = 0; j < NY; ++j) {
      for (size_t i = 0; i < NX; ++i) {
        size_t idx    = i + NX * j + NX * NY * k;
        data_out[idx] = idx;
      }
    }
  }
  data_out[4 + NX * 0 + NX * NY * 0] = 0;
  data_out[5 + NX * 0 + NX * NY * 0] = 0;
  data_out[4 + NX * 1 + NX * NY * 0] = 0;
  data_out[5 + NX * 1 + NX * NY * 0] = 0;
  data_out[4 + NX * 0 + NX * NY * 1] = 0;
  data_out[5 + NX * 0 + NX * NY * 1] = 0;
  data_out[4 + NX * 1 + NX * NY * 1] = 0;
  data_out[5 + NX * 1 + NX * NY * 1] = 0;

  file f_out{file_path_xyz, netCDF::NcFile::replace};
  auto dim_x = f_out.add_dimension(xdim_name, NX);
  auto dim_y = f_out.add_dimension(ydim_name, NY);
  auto dim_z = f_out.add_dimension(zdim_name, NZ);
  f_out.add_variable<double>(variable_name, {dim_z, dim_y, dim_x})
      .write(data_out);
  return data_out;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto write_simple_xyzt() {
  // 4D data
  std::vector<double> data_out(NX * NY * NZ * NT);
  // create some data
  for (size_t l = 0; l < NT; ++l) {
    for (size_t k = 0; k < NZ; ++k) {
      for (size_t j = 0; j < NY; ++j) {
        for (size_t i = 0; i < NX; ++i) {
          size_t idx    = i + NX * j + NX * NY * k + NX * NY * NZ * l;
          data_out[idx] = idx;
        }
      }
    }
  }
  data_out[4 + NX * 0 + NX * NY * 0 + NX * NY * NZ * 0] = 0;
  data_out[5 + NX * 0 + NX * NY * 0 + NX * NY * NZ * 0] = 0;
  data_out[4 + NX * 1 + NX * NY * 0 + NX * NY * NZ * 0] = 0;
  data_out[5 + NX * 1 + NX * NY * 0 + NX * NY * NZ * 0] = 0;
  data_out[4 + NX * 0 + NX * NY * 1 + NX * NY * NZ * 0] = 0;
  data_out[5 + NX * 0 + NX * NY * 1 + NX * NY * NZ * 0] = 0;
  data_out[4 + NX * 1 + NX * NY * 1 + NX * NY * NZ * 0] = 0;
  data_out[5 + NX * 1 + NX * NY * 1 + NX * NY * NZ * 0] = 0;
  data_out[4 + NX * 0 + NX * NY * 0 + NX * NY * NZ * 1] = 0;
  data_out[5 + NX * 0 + NX * NY * 0 + NX * NY * NZ * 1] = 0;
  data_out[4 + NX * 1 + NX * NY * 0 + NX * NY * NZ * 1] = 0;
  data_out[5 + NX * 1 + NX * NY * 0 + NX * NY * NZ * 1] = 0;
  data_out[4 + NX * 0 + NX * NY * 1 + NX * NY * NZ * 1] = 0;
  data_out[5 + NX * 0 + NX * NY * 1 + NX * NY * NZ * 1] = 0;
  data_out[4 + NX * 1 + NX * NY * 1 + NX * NY * NZ * 1] = 0;
  data_out[5 + NX * 1 + NX * NY * 1 + NX * NY * NZ * 1] = 0;

  file f_out{file_path_xyzt, netCDF::NcFile::replace};
  auto dim_x = f_out.add_dimension(xdim_name, NX);
  auto dim_y = f_out.add_dimension(ydim_name, NY);
  auto dim_z = f_out.add_dimension(zdim_name, NZ);
  auto dim_t = f_out.add_dimension(tdim_name, NT);
  f_out.add_variable<double>(variable_name, {dim_t, dim_z, dim_y, dim_x})
      .write(data_out);
  return data_out;
}
//==============================================================================
TEST_CASE("netcdf_write_read", "[netcdf][read][write]") {
  auto const data_out = write_simple_xy();
  file       f_in{file_path_xy, netCDF::NcFile::read};
  // Retrieve the variable
  auto var = f_in.variable<double>(variable_name);
  REQUIRE(var.size(1) == NX);
  REQUIRE(var.size(0) == NY);

  std::cerr << var.dimension_name(0) << '\n';
  std::cerr << var.dimension_name(1) << '\n';
  SECTION("read full") {
    auto data_in = var.read();
    REQUIRE(data_in.size(0) == NX);
    REQUIRE(data_in.size(1) == NY);
    std::cerr << "full: ";
    for (auto d : data_in.data())
      std::cerr << d << ' ';
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
      for (auto d : chunk.data())
        std::cerr << d << ' ';
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
      for (auto d : chunk.data())
        std::cerr << d << ' ';
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
      for (auto d : chunk.data())
        std::cerr << d << ' ';
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
      for (auto d : chunk.data())
        std::cerr << d << ' ';
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
      for (auto d : chunk.data())
        std::cerr << d << ' ';
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
    for (auto d : data_in.chunk_at(0)->data())
      std::cerr << d << ' ';
    std::cerr << '\n';
    std::cerr << "chunk 1: ";
    for (auto d : data_in.chunk_at(1)->data())
      std::cerr << d << ' ';
    std::cerr << '\n';
    std::cerr << "chunk 2: ";
    REQUIRE(data_in.chunk_at_is_null(2));
    std::cerr << "chunk 3: ";
    for (auto d : data_in.chunk_at(3)->data())
      std::cerr << d << ' ';
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
TEST_CASE("netcdf_lazy_xy", "[netcdf][lazy][xy]") {
  auto const          data_out = write_simple_xy();
  lazy_reader<double> cont{file_path_xy, variable_name,
                           std::vector<size_t>{2, 2}};

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
TEST_CASE("netcdf_lazy_xyz", "[netcdf][lazy][xyz]") {
  auto const          data_out = write_simple_xyz();
  lazy_reader<double> cont{file_path_xyz, variable_name,
                           std::vector<size_t>{2, 2, 2}};

  REQUIRE(cont.chunk_at_is_null(0));
  REQUIRE(cont(0, 0, 0) == 0);
  REQUIRE(!cont.chunk_at_is_null(0));

  REQUIRE(cont.chunk_at_is_null(1));
  REQUIRE(cont(2, 0, 0) == 2);
  REQUIRE(!cont.chunk_at_is_null(1));

  REQUIRE(cont.chunk_at_is_null(2));
  REQUIRE(cont(4, 0, 0) == 0);
  REQUIRE(cont.chunk_at_is_null(2));
  REQUIRE(cont(5, 0, 0) == 0);
  REQUIRE(cont.chunk_at_is_null(2));
}
//==============================================================================
TEST_CASE("netcdf_lazy_xyzt", "[netcdf][lazy][xyzt]") {
  auto const          data_out = write_simple_xyzt();
  lazy_reader<double> cont{file_path_xyzt, variable_name,
                           std::vector<size_t>{2, 2, 2, 2}};

  REQUIRE(cont.chunk_at_is_null(0));
  REQUIRE(cont(0, 0, 0, 0) == 0);
  REQUIRE(!cont.chunk_at_is_null(0));

  REQUIRE(cont.chunk_at_is_null(1));
  REQUIRE(cont(2, 0, 0, 0) == 2);
  REQUIRE(!cont.chunk_at_is_null(1));

  REQUIRE(cont.chunk_at_is_null(2));
  REQUIRE(cont(4, 0, 0, 0) == 0);
  REQUIRE(cont.chunk_at_is_null(2));
  REQUIRE(cont(5, 0, 0, 0) == 0);
  REQUIRE(cont.chunk_at_is_null(2));

  for (int l = NT - 1; l >= 0; --l) {
    for (int k = NZ - 1; k >= 0; --k) {
      for (int j = NY - 1; j >= 0; --j) {
        for (int i = NX - 1; i >= 0; --i) {
          size_t idx = i + NX * j + NX * NY * k + NX * NY * NZ * l;
          REQUIRE(cont(i, j, k, l) == data_out[idx]);
        }
      }
    }
  }
}
//==============================================================================
TEST_CASE("netcdf_unlimited_mat", "[netcdf][unlimited][mat]") {
  auto const data_out = write_unlimited_mat_list();
  file       f_in{file_unlimited_mat_list, netCDF::NcFile::read};
  auto       var = f_in.variable<double>(variable_name);
  std::cerr << var.dimension_name(2) << '\n';
  std::cerr << var.dimension_name(1) << '\n';
  std::cerr << var.dimension_name(0) << '\n';

  REQUIRE(var.size(0) == 10);
  REQUIRE(var.size(1) == 2);
  REQUIRE(var.size(2) == 2);

  std::vector<size_t>       is{0, 0, 0};
  std::vector<size_t> const cnt{1, 2, 2};
  for (; is.front() < var.size(0); ++is.front()) {
    auto const data_in = var.read_chunk(is, cnt);
    REQUIRE(data_in.size(0) == 1);
    REQUIRE(data_in.size(1) == 2);
    REQUIRE(data_in.size(2) == 2);
    CAPTURE(is.front());
    REQUIRE(data_in(0, 0, 0) == data_out[is.front()](0, 0));
    REQUIRE(data_in(0, 1, 0) == data_out[is.front()](1, 0));
    REQUIRE(data_in(0, 0, 1) == data_out[is.front()](0, 1));
    REQUIRE(data_in(0, 1, 1) == data_out[is.front()](1, 1));
  }
}
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
