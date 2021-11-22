#include <tatooine/lazy_reader.h>
#include <tatooine/rectilinear_grid.h>

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
TEST_CASE("lazy_reader_hdf5_stress", "[lazy_reader][hdf5][stress]") {
  auto file    = hdf5::file{"lazy_reader_hdf_stress.h5"};
  auto dataset = file.create_dataset<double>("data", 10, 11, 12);

  auto chunk = dynamic_multidim_array<double>{2, 2, 2};
  for (size_t iz = 0; iz < 4096/512; ++iz) { 
  for (size_t iy = 0; iy < 2048/512; ++iy) { 
  for (size_t ix = 0; ix < 512/512; ++ix) { 
    dataset.write
  }}}
  
  lazy_reader<hdf5>
}
//==============================================================================
TEST_CASE("lazy_reader_netcdf_grid", "[lazy_reader][netcdf][rectilinear_grid]") {
  //nonuniform_rectilinear_grid<double, 2> g{"simple_xy.nc"};
  //auto const& prop = g.vertex_property<double>("data");
  //
  //REQUIRE(prop(7, 0) == Approx(7));
  //REQUIRE(prop(0, 5) == Approx(40));
}
//==============================================================================
#endif
#ifdef TATOOINE_HDF5_AVAILABLE
TEST_CASE("lazy_reader_hdf5_limit_num_chunks",
          "[lazy_reader][hdf5][limit_num_chunks]") {
  auto file_path     = std::filesystem::path{"hdf5_limit_num_chunks.h5"};
  auto dataset_name  = std::string{"data"};
  using dataset_type = double;
  if (!std::filesystem::exists(file_path)) {
    auto const data_size = size_t(256);
    auto const chunk_size = size_t(128);
    auto const num_chunks = data_size / chunk_size;
    auto       file       = hdf5::file{file_path};
    auto       dataset =
        file.create_dataset<dataset_type>(dataset_name, data_size, data_size, data_size);
    auto data_arr =
        dynamic_multidim_array<dataset_type>{chunk_size, chunk_size, chunk_size};
    auto i = size_t(0);
    for (size_t iz = 0; iz < num_chunks; ++iz) {
      for (size_t iy = 0; iy < num_chunks; ++iy) {
        for (size_t ix = 0; ix < num_chunks; ++ix, ++i) {
          // modify data_arr to write
#pragma omp parallel for collapse(3)
          for (size_t idz = 0; idz < chunk_size; ++idz) {
            for (size_t idy = 0; idy < chunk_size; ++idy) {
              for (size_t idx = 0; idx < chunk_size; ++idx) {
                data_arr(idx, idy, idz) = i;
              }
            }
          }

          // write chunk
          std::cerr << "writing " << i + 1 << " / "
                    << num_chunks * num_chunks * num_chunks << "... ";
          dataset.write(data_arr, std::vector{ix, iy, iz});
          std::cerr << "done!\n";
        }
      }
    }
  }

  auto file    = hdf5::file{file_path};
  auto dataset = file.dataset<dataset_type>(dataset_name);
  auto reader  = lazy_reader{dataset, std::vector<size_t>{32, 32, 32}};
  reader.limit_num_chunks_loaded();
  reader.set_max_num_chunks_loaded(2);

  REQUIRE(reader.num_chunks_loaded() == 0);

  reader(0,0,0);
  REQUIRE(reader.num_chunks_loaded() == 1);
  REQUIRE(reader.chunk_is_loaded(0));
  reader(31,31,31);
  REQUIRE(reader.num_chunks_loaded() == 1);
  REQUIRE(reader.chunk_is_loaded(0));
  reader(32,0,0);
  REQUIRE(reader.num_chunks_loaded() == 2);
  REQUIRE(reader.chunk_is_loaded(0));
  REQUIRE(reader.chunk_is_loaded(1));
  reader(64,0,0);
  REQUIRE(reader.num_chunks_loaded() == 2);
  REQUIRE_FALSE(reader.chunk_is_loaded(0));
  REQUIRE(reader.chunk_is_loaded(1));
  REQUIRE(reader.chunk_is_loaded(2));
  reader(32,0,0);
  reader(0,0,0);
  REQUIRE(reader.chunk_is_loaded(0));
  REQUIRE(reader.chunk_is_loaded(1));
  REQUIRE_FALSE(reader.chunk_is_loaded(2));
}
//==============================================================================
TEST_CASE("lazy_reader_hdf5_parallel_access",
          "[lazy_reader][hdf5][parallel_access]") {
  auto file_path = std::filesystem::path{"hdf5_limit_num_chunks_parallel.h5"};
  auto dataset_name  = std::string{"data"};
  auto const data_size = size_t(256);
  using dataset_type = double;
  if (!std::filesystem::exists(file_path)) {
    auto const chunk_size = size_t(128);
    auto const num_chunks = data_size / chunk_size;
    auto       file       = hdf5::file{file_path};
    auto       dataset =
        file.create_dataset<dataset_type>(dataset_name, data_size, data_size, data_size);
    auto data_arr =
        dynamic_multidim_array<dataset_type>{chunk_size, chunk_size, chunk_size};
    auto i = size_t(0);
    for (size_t iz = 0; iz < num_chunks; ++iz) {
      for (size_t iy = 0; iy < num_chunks; ++iy) {
        for (size_t ix = 0; ix < num_chunks; ++ix, ++i) {
          // modify data_arr to write
#pragma omp parallel for collapse(3)
          for (size_t idz = 0; idz < chunk_size; ++idz) {
            for (size_t idy = 0; idy < chunk_size; ++idy) {
              for (size_t idx = 0; idx < chunk_size; ++idx) {
                data_arr(idx, idy, idz) = i;
              }
            }
          }

          // write chunk
          std::cerr << "writing " << i + 1 << " / "
                    << num_chunks * num_chunks * num_chunks << "... ";
          dataset.write(data_arr, std::vector{ix, iy, iz});
          std::cerr << "done!\n";
        }
      }
    }
  }

  auto file    = hdf5::file{file_path};
  auto dataset = file.dataset<dataset_type>(dataset_name);
  auto const chunk_size = std::vector<size_t>(3, 2);
  auto       reader     = lazy_reader{dataset, chunk_size};
  reader.set_max_num_chunks_loaded(100);

  auto iteration = [&](auto const /*i*/) {
    auto rand = random::uniform<size_t>{0, data_size - 1};
    reader(rand(), rand(), rand());
  };
  for_loop(iteration, execution_policy::parallel, size_t(1000000000));
}
//==============================================================================
TEST_CASE("lazy_reader_hdf5", "[lazy_reader][hdf5]") {
  auto filepath    = filesystem::path{"unittest.lazy_reader_hdf5.h5"};
  auto array_name  = std::string{"Array"};
  auto full_size   = std::vector<size_t>{64, 64, 64};
  using value_type = int;
  std::vector<value_type> data_src(full_size[0] * full_size[1] * full_size[2]);
  auto rand = random::uniform{value_type(-1000), value_type(1000)};
  std::generate(begin(data_src), end(data_src), [&rand]() { return rand(); });
  {
    if (filesystem::exists(filepath)) {
      auto out     = hdf5::file{filepath};
      auto arr_out = out.dataset<value_type>(array_name);
      arr_out.write(data_src);
    } else {
      auto out     = hdf5::file{filepath};
      auto arr_out = out.create_dataset<value_type>(array_name, full_size[0],
                                                 full_size[1], full_size[2]);
      arr_out.write(data_src);
    }
  }

  auto in     = hdf5::file{filepath};
  auto arr_in = in.dataset<value_type>(array_name);
  auto arr_lazy_x_fastest = arr_in.read_lazy<x_fastest>({2, 8, 4});
  auto arr_lazy_x_slowest = arr_in.read_lazy<x_slowest>({2, 4, 4});
  REQUIRE(arr_lazy_x_slowest.size(0) == arr_lazy_x_fastest.size(2));
  REQUIRE(arr_lazy_x_slowest.size(2) == arr_lazy_x_fastest.size(0));
  arr_lazy_x_fastest.set_max_num_chunks_loaded(8);
  //arr_lazy_x_fastest.limit_num_chunks_loaded();

  for (size_t i = 0; i < 3; ++i) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < full_size[2]; ++z) {
      for (size_t y = 0; y < full_size[1]; ++y) {
        for (size_t x = 0; x < full_size[0]; ++x) {
          auto const j = x + y * full_size[0] + z * full_size[0] * full_size[1];
          // CAPTURE(i, j, x, y, z);
          REQUIRE(data_src[j] == arr_lazy_x_fastest(x, y, z));
          REQUIRE(arr_lazy_x_fastest(x, y, z) == arr_lazy_x_slowest(z, y, x));
        }
      }
    }
  }
}
#endif
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
