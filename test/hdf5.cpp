#if TATOOINE_HDF5_AVAILABLE
#include <tatooine/rectilinear_grid.h>
#include <boost/range/algorithm/generate.hpp>
#include <boost/range/algorithm_ext/iota.hpp>
#include <tatooine/random.h>

#include <catch2/catch_test_macros.hpp>
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
  boost::iota(data_src.internal_container(), 1);
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
}
//==============================================================================
TEST_CASE("hdf5_unlimited_1d", "[hdf5][unlimited][1d]") {
  using value_type        = linspace<real_number>;
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
TEST_CASE("hdf5_attribute", "[hdf5][attribute]") {
  auto const filepath = filesystem::path{"hdf5_unittest_attribute.h5"};

  if (filesystem::exists(filepath)) {
    filesystem::remove(filepath);
  }
  auto file    = hdf5::file{filepath};
  file.attribute("string") = std::string{"abc"};
  file.attribute("char*") = "def";
  file.attribute("int") = 1;
  REQUIRE("abc" == file.attribute("string").read_as_string());
  REQUIRE("def" == file.attribute("char*").read_as_string());
  REQUIRE(1 == file.attribute("int").read_as<int>());
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
#endif
