#if TATOOINE_HAS_HDF5_SUPPORT
#include <tatooine/grid.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("hdf5_read_chunk", "[hdf5][read][chunk]") {
  using data_t    = int;
  auto filepath   = std::filesystem::path{"hdf5_unittest.h5"};
  auto array_name = std::string{"Array"};

  auto                full_size = std::vector<size_t>{64, 128};
  std::vector<data_t> data_src(full_size[0] * full_size[1]);
  std::iota(begin(data_src), end(data_src), 0);
  {
    auto out = hdf5::file{filepath, H5F_ACC_TRUNC};
    auto arr_out =
        out.add_dataset<data_t>(array_name, full_size[0], full_size[1]);
    arr_out.write(data_src);
  }

  auto in     = hdf5::file{filepath, H5F_ACC_RDONLY};
  auto arr_in = in.dataset<data_t>(std::filesystem::path{array_name});

  auto const full_data = arr_in.read();
  SECTION("Full Data check") {
    REQUIRE(full_data.num_dimensions() == 2);
    REQUIRE(full_data.size(0) == full_size[0]);
    REQUIRE(full_data.size(1) == full_size[1]);
    REQUIRE(full_data(0,0) == 0);
    REQUIRE(full_data(1,0) == 1);
    REQUIRE(full_data(0,1) == 64);
    size_t i = 0;
    for (size_t y = 0; y < full_size[1]; ++y) {
      for (size_t x = 0; x < full_size[0]; ++x) {
        CHECK(full_data(x, y) == data_src[i++]);
      }
    }
  }

  auto read_chunk = [&](std::vector<size_t> const& offset,
                        std::vector<size_t> const& size) {
    auto const chunk = arr_in.read_chunk(offset, size);

    REQUIRE(chunk.num_dimensions() == 2);
    REQUIRE(chunk.size(0) == size[0]);
    REQUIRE(chunk.size(1) == size[1]);
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t x = 0; x < size[0]; ++x) {
        CAPTURE(x, y, (x + offset[0]), (y + offset[1]));
        CHECK(chunk(x, y) == full_data(x + offset[0], y + offset[1]));
      }
    }
  };
  SECTION("Chunk [0..1] x [0..1] Test") {
    auto const offset = std::vector<size_t>{0, 0};
    auto const size   = std::vector<size_t>{3, 2};
    read_chunk(offset, size);
  }

  SECTION("Chunk [4..5] x [4..5] Test") {
    auto const offset = std::vector<size_t>{4, 5};
    auto const size   = std::vector<size_t>{2, 3};
    read_chunk(offset, size);
  }
}
//==============================================================================
// TEST_CASE("hdf5_grid", "[hdf5][grid]") {
//  uniform_grid<double, 3> grid3{2, 2, 2};
//  REQUIRE_THROWS(grid3.add_lazy_property<double>("../test.h5", "Array"));
//  uniform_grid<double, 2> grid2{2, 2};
//  REQUIRE_THROWS(grid2.add_lazy_property<double>("../test.h5", "Array"));
//  uniform_grid<double, 2> grid{64, 128};
//  auto &data = grid.add_lazy_property<double>("../test.h5", "Array");
//  REQUIRE(data(0, 0) == 0);
//  REQUIRE(data(63,127) == 1);
//}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
#endif
