#if TATOOINE_HAS_HDF5_SUPPORT
#include <tatooine/grid.h>
#include <tatooine/isosurface.h>
#include <tatooine/random.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("hdf5_read_chunk", "[hdf5][read][chunk]") {
  using value_type = int;
  auto filepath    = filesystem::path{"hdf5_unittest.h5"};
  auto array_name  = std::string{"Array"};

  auto                    full_size = std::vector<size_t>{64, 512, 32};
  std::vector<value_type> data_src(full_size[0] * full_size[1] * full_size[2]);
  auto rand = random_uniform{value_type(-1000), value_type(1000)};
  std::generate(begin(data_src), end(data_src), [&rand]() { return rand(); });
  {
    if (!filepath.exists()) {
      auto out     = hdf5::file{filepath};
      auto arr_out = out.add_dataset<value_type>(array_name, full_size[0],
                                                 full_size[1], full_size[2]);
      arr_out.write(data_src);
    }
  }

  auto in     = hdf5::file{filepath};
  auto arr_in = in.dataset<value_type>(array_name);

  auto const full_data = arr_in.read();
  SECTION("Full Data check") {
    REQUIRE(full_data.num_dimensions() == 3);
    REQUIRE(full_data.size(0) == full_size[0]);
    REQUIRE(full_data.size(1) == full_size[1]);
    REQUIRE(full_data.size(2) == full_size[2]);
    size_t i = 0;
    for (size_t z = 0; z < full_size[2]; ++z) {
      for (size_t y = 0; y < full_size[1]; ++y) {
        for (size_t x = 0; x < full_size[0]; ++x) {
          CHECK(full_data(x, y, z) == data_src[i++]);
        }
      }
    }
  }
  SECTION("Full Data check multithreaded") {
#pragma omp parallel for
    for (size_t k = 0; k < 10000; ++k) {
      auto const chunk = arr_in.read_chunk(std::vector<size_t>{16, 16, 16},
                                           std::vector<size_t>{16, 16, 16});
    }
  }

  auto read_chunk = [&](std::vector<size_t> const& offset,
                        std::vector<size_t> const& size) {
    auto const chunk = arr_in.read_chunk(offset, size);

    REQUIRE(chunk.num_dimensions() == 3);
    REQUIRE(chunk.size(0) == size[0]);
    REQUIRE(chunk.size(1) == size[1]);
    REQUIRE(chunk.size(2) == size[2]);
    for (size_t z = 0; z < size[2]; ++z) {
      for (size_t y = 0; y < size[1]; ++y) {
        for (size_t x = 0; x < size[0]; ++x) {
          CAPTURE(x, y, z, (x + offset[0]), (y + offset[1]), (z + offset[2]));
           CHECK(chunk(x, y, z) ==
                full_data(x + offset[0], y + offset[1], z + offset[2]));
        }
      }
    }
  };
  SECTION("Chunk [0..1] x [0..1] Test") {
    auto const offset = std::vector<size_t>{0, 0, 0};
    auto const size   = std::vector<size_t>{3, 2, 4};
    read_chunk(offset, size);
  }

  SECTION("Chunk [4..5] x [4..5] Test") {
    auto const offset = std::vector<size_t>{4, 5, 6};
    auto const size   = std::vector<size_t>{2, 3, 4};
    read_chunk(offset, size);
  }
  // lazy_reader<hdf5::dataset<value_type>> lr{arr_in, {2, 2, 2}};
  // REQUIRE(lr(full_size[0] - 1, full_size[1] - 1, full_size[2] - 1) ==
  //        data_src.back());
  // for (size_t z = 0; z < full_size[2]; ++z) {
  //  for (size_t y = 0; y < full_size[1]; ++y) {
  //    for (size_t x = 0; x < full_size[0]; ++x) {
  //      CHECK(full_data(x, y, z) == lr(x, y, z));
  //    }
  //  }
  //}

  auto  domain = grid{linspace<double>{0.0, double(full_size[0]), full_size[0]},
                     linspace<double>{0.0, double(full_size[1]), full_size[1]},
                     linspace<double>{0.0, double(full_size[2]), full_size[2]}};
  auto& prop   = domain.add_lazy_vertex_property(arr_in);
  auto  ev     = [&](auto const ix, auto const iy, auto const iz,
                auto const&  /*p*/) { return prop(ix, iy, iz); };
  auto  isomesh = isosurface(ev, domain, 0);
  isomesh.write_vtk("isomesh.vtk");
}
//==============================================================================
TEST_CASE("hdf5_grid", "[hdf5][grid]") {
  auto array_name  = std::string{"Array"};
  uniform_grid<double, 3> grid3{2, 2, 2};
  REQUIRE_THROWS(grid3.add_lazy_vertex_property<double>("../test.h5", "Array"));
  uniform_grid<double, 2> grid2{2, 2};
  REQUIRE_THROWS(grid2.add_lazy_vertex_property<double>("../test.h5", "Array"));
  uniform_grid<double, 2> grid{64, 128};
  auto& data = grid.add_lazy_vertex_property<double>("../test.h5", "Array");
  REQUIRE(data(0, 0) == 0);
  REQUIRE(data(63, 127) == 1);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
#endif
