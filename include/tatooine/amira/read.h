#ifndef TATOOINE_AMIRA_READ_H
#define TATOOINE_AMIRA_READ_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/concepts.h>
#include <tatooine/filesystem.h>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>
//==============================================================================
namespace tatooine::amira {
//==============================================================================
static constexpr auto lattice      = std::string_view{"define Lattice"};
static constexpr auto boundingbox  = std::string_view{"BoundingBox"};
static constexpr auto lattice_size = std::string_view{"Lattice { float["};
static constexpr auto data_follows = std::string_view{"# Data section follows"};
static constexpr auto num_bytes_header = std::size_t(2048);
//==============================================================================
inline auto read_header(std::ifstream& file) {
  auto const cur_cursor_pos = file.tellg();
  // We read the first 2048 bytes into memory to parse the header.
  auto constexpr num_bytes_header = std::size_t(2048);
  auto buf                        = std::string(
                             num_bytes_header / sizeof(typename std::string::value_type), ' ');
  file.read(buf.data(), num_bytes_header);

  // remove eof flag if read all data
  if (file.eof()) {
    file.clear();
  }
  file.seekg(0, std::ios_base::beg);
  auto buffer = std::istringstream{std::move(buf)};

  // Find the Lattice definition, i.e., the dimensions of the uniform grid
  auto       dims = std::array<std::size_t, 3>{};
  auto const lattice_pos =
      static_cast<std::streamoff>(buffer.str().find(lattice));
  buffer.seekg(lattice_pos, std::ios_base::beg);
  buffer.seekg(lattice.size(), std::ios_base::cur);
  buffer >> dims[0] >> dims[1] >> dims[2];

  // Find the boundingbox
  auto       aabb = axis_aligned_bounding_box<float, 3>{};
  auto const boundingbox_pos =
      static_cast<std::streamoff>(buffer.str().find(boundingbox));
  buffer.seekg(boundingbox_pos, std::ios_base::beg);
  buffer.seekg(boundingbox.size(), std::ios_base::cur);
  buffer >> aabb.min(0) >> aabb.max(0) >> aabb.min(1) >> aabb.max(1) >>
      aabb.min(2) >> aabb.max(2);

  // Is it a uniform grid? We need this only for the sanity check below.
  auto const is_uniform =
      buffer.str().find("CoordType \"uniform\"") != std::string::npos;

  // Type of the field: scalar, vector
  auto num_components = std::size_t{};
  if (buffer.str().find("Lattice { float Data }") != std::string::npos) {
    // Scalar field
    num_components = 1;
  } else {
    // A field with more than one component, i.e., a vector field
    auto const pos = buffer.str().find(lattice_size);
    buffer.seekg(static_cast<std::streamoff>(pos), std::ios_base::beg);
    buffer.seekg(lattice_size.size(), std::ios_base::cur);
    buffer >> num_components;
  }

  // Sanity check
  if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 ||
      aabb.min(0) > aabb.max(0) || aabb.min(1) > aabb.max(1) ||
      aabb.min(2) > aabb.max(2) || !is_uniform || num_components <= 0) {
    throw std::runtime_error("something went wrong");
  }

  // Find the beginning of the data section
  auto const data_follows_pos = buffer.str().find(data_follows);
  auto break_pos = static_cast<std::streamoff>(
      buffer.str().find('\n', static_cast<std::streamoff>(data_follows_pos)));
  break_pos =
      static_cast<std::streamoff>(buffer.str().find('\n', break_pos + 1));
  auto data_begin_pos   = std::streamoff{};
  if (data_follows_pos != std::string::npos) {
    data_begin_pos = static_cast<std::streamoff>(break_pos + 1);
  }
  file.seekg(cur_cursor_pos);
  return std::tuple{dims, std::move(aabb), num_components, data_begin_pos};
}
//------------------------------------------------------------------------------
inline auto read_header(filesystem::path const& path) {
  auto file = std::ifstream{path};
  if (!file.is_open()) {
    throw std::runtime_error("could not open file " + path.string());
  }
  return read_header(file);
}
//==============================================================================
/// A simple routine to read an AmiraMesh file
/// that defines a scalar/vector field on a uniform grid.
template <floating_point T = float>
auto read(std::ifstream& file) {
  auto [dims, aabb, num_components, data_begin_pos] = read_header(file);
  auto const num_to_read = dims[0] * dims[1] * dims[2] * num_components;
  auto       data        = std::vector<float>(num_to_read);

  file.seekg(data_begin_pos, std::ios_base::beg);
  file.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(sizeof(float) * num_to_read));

  if constexpr (is_float<T>) {
    return std::tuple{std::move(data), dims, std::move(aabb), num_components};
  } else {
    return std::tuple{std::vector<T>(begin(data), end(data)), dims,
                      std::move(aabb), num_components};
  }
}
//------------------------------------------------------------------------------
template <floating_point T = float>
auto read(filesystem::path const& path) {
  auto file = std::ifstream{path};
  if (!file.is_open()) {
    throw std::runtime_error("could not open file " + path.string());
  }
  return read<T>(file);
}
//==============================================================================
}  // namespace tatooine::amira
//==============================================================================
#endif
