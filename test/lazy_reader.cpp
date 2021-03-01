#include <tatooine/lazy_reader.h>
#include <tatooine/grid.h>

#ifdef TATOOINE_NETCDF_AVAILABLE
#include <tatooine/netcdf.h>
#endif
#ifdef TATOOINE_HDF5_AVAILABLE
#include <tatooine/hdf5.h>
#endif
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
#ifdef TATOOINE_NETCDF_AVAILABLE
TEST_CASE("lazy_reader_netcdf", "[lazy_reader][netcdf]") {
  //netcdf::lazy_reader<double> cont{"simple_xy.nc", "data", {2, 2}};
  //auto const&                 resolution = cont.size();
  //CAPTURE(resolution);
  //SECTION("accessing first chunk") {
  //  SECTION("accessing first element") {
  //    REQUIRE(cont.chunk_at_is_null(0, 0));
  //    REQUIRE(cont(0, 0) == 0);
  //    REQUIRE_FALSE(cont.chunk_at_is_null(0, 0));
  //  }
  //  SECTION("accessing second element") {
  //    REQUIRE(cont.chunk_at_is_null(0, 0));
  //    REQUIRE(cont(1, 0) == 1);
  //    REQUIRE_FALSE(cont.chunk_at_is_null(0, 0));
  //  }
  //}
  //SECTION("accessing last chunk") {
  //  REQUIRE(cont.chunk_at_is_null(3, 2));
  //  REQUIRE(cont(resolution[0] - 1, resolution[1] - 1) == 47);
  //  REQUIRE_FALSE(cont.chunk_at_is_null(3, 2));
  //}
  //SECTION("correct order") {
  //  REQUIRE(cont(resolution[0] - 1, 0) == 7);
  //  REQUIRE(cont(0, resolution[1] - 1) == 40);
  //}
}
//==============================================================================
TEST_CASE("lazy_reader_netcdf_grid", "[lazy_reader][netcdf][grid]") {
  //non_uniform_grid<double, 2> g{"simple_xy.nc"};
  //auto const& prop = g.vertex_property<double>("data");
  //
  //REQUIRE(prop(7, 0) == Approx(7));
  //REQUIRE(prop(0, 5) == Approx(40));
}
//==============================================================================
#endif
#ifdef TATOOINE_HDF5_AVAILABLE
TEST_CASE("lazy_reader_hdf5", "[lazy_reader][hdf5]") {
  auto filepath    = filesystem::path{"unittest.lazy_reader_hdf5.h5"};
  auto array_name  = std::string{"Array"};
  auto full_size   = std::vector<size_t>{64, 64, 64};
  using value_type = int;
  std::vector<value_type> data_src(full_size[0] * full_size[1] * full_size[2]);
  auto rand = random_uniform{value_type(-1000), value_type(1000)};
  std::generate(begin(data_src), end(data_src), [&rand]() { return rand(); });
  {
    if (filesystem::exists(filepath)) {
      auto out     = hdf5::file{filepath};
      auto arr_out = out.dataset<value_type>(array_name);
      arr_out.write(data_src);
    } else {
      auto out     = hdf5::file{filepath};
      auto arr_out = out.add_dataset<value_type>(array_name, full_size[0],
                                                 full_size[1], full_size[2]);
      arr_out.write(data_src);
    }
  }

  auto in     = hdf5::file{filepath};
  auto arr_in = in.dataset<value_type>(array_name);
  auto arr_lazy = arr_in.read_lazy({2, 2, 2});
  arr_lazy.set_max_num_chunks_loaded(8);
  //arr_lazy.limit_num_chunks_loaded();

  for (size_t i = 0; i < 3; ++i) {
    //#pragma omp parallel for collapse(3)
    for (size_t z = 0; z < full_size[2]; ++z) {
      for (size_t y = 0; y < full_size[1]; ++y) {
        for (size_t x = 0; x < full_size[0]; ++x) {
          auto const j = x + y * full_size[0] + z * full_size[0] * full_size[1];
          // CAPTURE(i, j, x, y, z);
          REQUIRE(data_src[j] == arr_lazy(x, y, z));
        }
      }
    }
  }
}
#endif
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
