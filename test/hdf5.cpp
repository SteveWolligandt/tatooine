#if TATOOINE_HDF5_AVAILABLE
#include <tatooine/rectilinear_grid.h>
#include <boost/range/algorithm/generate.hpp>
#include <boost/range/algorithm_ext/iota.hpp>
#include <tatooine/random.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("hdf5_dataset_creation", "[hdf5][dataset][creation]") {
  auto filepath = filesystem::path{"hdf5_dataset_creation.h5"};
  auto write    = std::vector<int>(12 * 12, 2);
  auto read     = std::vector<int>{};

  auto f = hdf5::file{filepath};
  auto d = f.create_dataset<int>("test", 12, 12);
  d.write(write.data());
  d.read(read);

  for (std::size_t i = 0; i < 12 * 12; ++i) {
    REQUIRE(write[i] == read[i]);
  }

  filesystem::remove(filepath);
}
//==============================================================================
TEST_CASE("hdf5_read_chunk", "[hdf5][read][chunk]") {
  using value_type = int;
  auto filepath    = filesystem::path{"hdf5_unittest_chunk.h5"};
  auto array_name  = std::string{"Array"};

  auto full_size = std::vector<size_t>{32, 64, 128};
  auto data_src = dynamic_multidim_array<value_type>{full_size[0], full_size[1],
                                                     full_size[2]};

  auto loop = [&](auto&& f) {
    for_loop(std::forward<decltype(f)>(f), full_size[0], full_size[1],
             full_size[2]);
  };

  // auto rand = random::uniform{value_type(-1000), value_type(1000)};
  // boost::generate(data_src, [&rand]() { return rand(); });
  boost::iota(data_src.data(), 1);
  if (filesystem::exists(filepath)) {
    filesystem::remove(filepath);
  }
  {
    auto out     = hdf5::file{filepath};
    auto arr_out = out.create_dataset<value_type>(array_name, full_size[0],
                                                  full_size[1], full_size[2]);
    arr_out.write(data_src);
  }

  auto in     = hdf5::file{filepath};
  auto arr_in = in.dataset<value_type>(array_name);

  auto const full_data = arr_in.read();
  SECTION("Full Data check") {
    REQUIRE(full_data.num_dimensions() == 3);
    REQUIRE(full_data.size(0) == full_size[0]);
    REQUIRE(full_data.size(1) == full_size[1]);
    REQUIRE(full_data.size(2) == full_size[2]);
    loop([&](auto const... is) { REQUIRE(full_data(is...) == data_src(is...)); });
  }
  //  SECTION("Full Data check multithreaded") {
  //#pragma omp parallel for
  //    for (size_t k = 0; k < 10000; ++k) {
  //      auto const chunk = arr_in.read(std::vector<size_t>{16, 16, 16},
  //                                           std::vector<size_t>{16, 16, 16});
  //    }
  //  }

  auto read_chunk = [&](std::vector<size_t> const& offset,
                        std::vector<size_t> const& size) {
    auto const chunk = arr_in.read(offset, size);

    REQUIRE(chunk.num_dimensions() == 3);
    REQUIRE(chunk.size(0) == size[0]);
    REQUIRE(chunk.size(1) == size[1]);
    REQUIRE(chunk.size(2) == size[2]);
    for_loop([&](auto const x, auto const y, auto const z) {
      CAPTURE(x, y, z, (x + offset[0]), (y + offset[1]), (z + offset[2]));
      REQUIRE(chunk(x, y, z) ==
            full_data(x + offset[0], y + offset[1], z + offset[2]));
    }, size[0], size[1], size[2]);
  };
  SECTION("Chunk [0..1] x [0..1] Test") {
    auto const offset = std::vector<size_t>{0, 0, 0};
    auto const size   = std::vector<size_t>{2, 3, 4};
    read_chunk(offset, size);
  }

  SECTION("Chunk [4..5] x [4..5] Test") {
    auto const offset = std::vector<size_t>{4, 5, 6};
    auto const size   = std::vector<size_t>{2, 3, 4};
    read_chunk(offset, size);
  }
  //auto lr = lazy_reader<hdf5::dataset<value_type>>{arr_in, {2, 2, 2}};
  //loop([&](auto const... is) { REQUIRE(full_data(is...) == lr(is...)); });
}
//==============================================================================
TEST_CASE("hdf5_index_order", "[hdf5][index_order]") {
  auto filepath = filesystem::path{"hdf5_unittest_index_order.h5"};

  if (filesystem::exists(filepath)) {
    filesystem::remove(filepath);
  }
  auto file             = hdf5::file{filepath};
  size_t width = 4, height = 6;
  auto arr_out_fastest = file.create_dataset<int, x_fastest>("x_fastest", width, height);
  auto arr_out_slowest = file.create_dataset<int, x_slowest>("x_slowest", width, height);
  dynamic_multidim_array<int, x_slowest> data_x_slowest(width, height);
  dynamic_multidim_array<int, x_fastest> data_x_fastest(width, height);

  int x = 0;
  for (size_t j = 0; j < height; ++j) {
    for (size_t i = 0; i < width; ++i) {
      data_x_slowest(i, j) = x;
      data_x_fastest(i, j) = x++;
    }
  }
  REQUIRE(data_x_fastest.data()[1] == 1);
  REQUIRE(data_x_fastest.data()[1] == data_x_slowest.data()[height]);

  arr_out_slowest.write(data_x_slowest);
  arr_out_fastest.write(data_x_fastest);


  {
    auto arr_fastest = file.dataset<int>("x_fastest").read<x_fastest>();
    for (size_t j = 0; j < height; j++) {
      for (size_t i = 0; i < width; i++) {
        std::cout << arr_fastest(i, height - 1 - j) << ' ';
      }
      std::cout << '\n';
    }
    std::cout << '\n';
    auto arr_slowest = file.dataset<int>("x_slowest").read<x_slowest>();
    for (size_t j = 0; j < height; j++) {
      for (size_t i = 0; i < width; i++) {
        std::cout << arr_slowest(i, height - 1 - j) << ' ';
      }
      std::cout << '\n';
    }
  }
  std::cout << '\n';
  {
    auto arr_fastest = file.dataset<int>("x_fastest")
                           .read<x_fastest>(std::vector<size_t>{1, 2},
                                                  std::vector<size_t>{2, 3});
    for (size_t j = 0; j < 3; j++) {
      for (size_t i = 0; i < 2; i++) {
        std::cout << arr_fastest(i, 3 - 1 - j) << ' ';
      }
      std::cout << '\n';
    }
    std::cout << '\n';
    auto arr_slowest = file.dataset<int>("x_slowest")
                           .read<x_fastest>(std::vector<size_t>{1, 2},
                                                  std::vector<size_t>{2, 3});
    for (size_t j = 0; j < 3; j++) {
      for (size_t i = 0; i < 2; i++) {
        std::cout << arr_slowest(i, 3 - 1 - j) << ' ';
      }
      std::cout << '\n';
    }
  }
  //std::cout << '\n';
  //{
  //  auto arr_fastest_lazy =
  //      file.dataset<int>("x_fastest").read_lazy<x_fastest>({2, 2});
  //  for (size_t j = 0; j < height; j++) {
  //    for (size_t i = 0; i < width; i++) {
  //      std::cout << arr_fastest_lazy(i, height - 1 - j) << ' ';
  //    }
  //    std::cout << '\n';
  //  }
  //}
}
//==============================================================================
TEST_CASE("hdf5_reversed_memspace", "[hdf5][reversed_memspace]") {
  using value_type        = int;
  auto const width        = size_t(4);
  auto const height       = size_t(2);
  auto const data         = std::vector<value_type>{1, 2, 3, 4, 5, 6, 7, 8};
  auto const dataset_name = "foo";
  auto const filepath = filesystem::path{"hdf5_unittest_reversed_memspace.h5"};

  if (filesystem::exists(filepath)) {
    filesystem::remove(filepath);
  }
  {
    auto out     = hdf5::file{filepath};
    auto arr_out = out.create_dataset<value_type>(dataset_name, width, height);
    arr_out.write(data);
  }
  auto const ds = hdf5::file{filepath}.dataset<value_type>(dataset_name);
  auto const chunk =
      ds.read(std::vector<size_t>{0, 0}, std::vector<size_t>{2, 2});

  REQUIRE(chunk(0, 0) == data[0]);
  REQUIRE(chunk(1, 0) == data[1]);
  REQUIRE(chunk(0, 1) == data[width]);
  REQUIRE(chunk(1, 1) == data[width + 1]);
}
//==============================================================================
TEST_CASE("hdf5_unlimited_1d", "[hdf5][unlimited][1d]") {
  using value_type        = linspace<real_t>;
  auto const dataset_name = "foo";
  auto const filepath = filesystem::path{"hdf5_unittest_unlimited_1d.h5"};

  if (filesystem::exists(filepath)) {
    filesystem::remove(filepath);
  }
  auto file    = hdf5::file{filepath};
  auto dataset = file.create_dataset<value_type>(dataset_name, hdf5::unlimited);
  dataset.push_back(linspace{0.0, 1.0, 11});
  dataset.push_back(linspace{0.0, 2.0, 21});
  dataset.push_back(linspace{0.0, 3.0, 31});
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
#endif
