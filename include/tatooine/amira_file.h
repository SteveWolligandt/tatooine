#ifndef TATOOINE_AMIRA_READER_H
#define TATOOINE_AMIRA_READER_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/concepts.h>
#include <tatooine/filesystem.h>
#include <tatooine/vec.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
//==============================================================================
namespace tatooine::amira {
//==============================================================================
///// Find a string in the given buffer and return a pointer
///// to the contents directly behind the search_string.
///// If not found, return the buffer. A subsequent sscanf()
///// will fail then, but at least we return a decent pointer.
// inline auto find_and_jump(char const* buffer, char const* search_string)
//     -> char const* {
//   auto const found_loc = std::strstr(buffer, search_string);
//   if (found_loc) {
//     return found_loc + std::strlen(search_string);
//   }
//   return buffer;
// }
////------------------------------------------------------------------------------
///// Find a string in the given buffer and return a pointer
///// to the contents directly behind the search_string.
///// If not found, return the buffer. A subsequent sscanf()
///// will fail then, but at least we return a decent pointer.
// inline auto find_and_jump(std::string const& buffer,
//                           std::string_view   search_string) -> char const* {
//   return find_and_jump(buffer.data(), search_string.data());
// }
//------------------------------------------------------------------------------
/// A simple routine to read an AmiraMesh file
/// that defines a scalar/vector field on a uniform grid.
template <floating_point T = float>
auto read(filesystem::path const& path) {
  auto constexpr lattice = std::string_view{"define Lattice"};
  auto constexpr boundingbox        = std::string_view{"BoundingBox"};

  auto file = std::ifstream{path};
  if (!file.is_open()) {
    throw std::runtime_error("could not open file " + path.string());
  }

  // We read the first 2k bytes into memory to parse the header.
  // The fixed buffer size looks a bit like a hack, and it is one, but it gets
  // the job done.
  auto constexpr num_bytes_header = std::size_t(2048);
  auto buf                        = std::string(num_bytes_header, ' ');
  file.read(buf.data(), num_bytes_header);
  auto buffer = std::stringstream{std::move(buf)};

  // Find the Lattice definition, i.e., the dimensions of the uniform grid
  auto       dims                   = std::array<std::size_t, 3>{};
  auto const lattice_pos            = buffer.str().find(lattice);
  buffer.seekg(static_cast<std::streamoff>(lattice_pos), std::ios_base::beg);
  buffer.seekg(lattice.size(), std::ios_base::cur);
  buffer >> dims[0] >> dims[1] >> dims[2];

  // Find the boundingbox
  auto       aabb            = axis_aligned_bounding_box<float, 3>{};
  auto const boundingbox_pos = buffer.str().find(boundingbox);
  buffer.seekg(static_cast<std::streamoff>(boundingbox_pos),
               std::ios_base::beg);
  buffer.seekg(boundingbox.size(), std::ios_base::cur);
  buffer >> aabb.min(0) >> aabb.max(0) >> aabb.min(1) >> aabb.max(1) >>
      aabb.min(2) >> aabb.max(2);

  // Is it a uniform grid? We need this only for the sanity check below.
  auto const is_uniform =
       buffer.str().find("CoordType \"uniform\"") != std::string::npos;

  // Type of the field: scalar, vector
   int num_comps{0};
   if (buffer.str().find("Lattice { float Data }") != std::string::npos)
     // Scalar field
     num_comps = 1;
   else {
     // A field with more than one component, i.e., a vector field
     auto constexpr lattice_size = std::string_view{"Lattice { float["};
     auto const pos = buffer.str().find(lattice_size);
     buffer.seekg(static_cast<std::streamoff>(pos), std::ios_base::beg);
     buffer.seekg(lattice_size.size(), std::ios_base::cur);
     buffer >> num_comps;
   }

  //// Sanity check
  // if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 ||
  //     aabb.min(0) > aabb.max(0) || aabb.min(1) > aabb.max(1) ||
  //     aabb.min(2) > aabb.max(2) || !is_uniform || num_comps <= 0) {
  //   fclose(fp);
  //   throw std::runtime_error("something went wrong");
  // }
  //
  // auto data = std::vector<float>{};
  //// Find the beginning of the data section
  // long const idx_start_data = std::strstr(buffer.data(), "# Data section
  // follows") - buffer.data(); if (idx_start_data > 0) {
  //   // Set the file pointer to the beginning of "# Data section follows"
  //   fseek(fp, idx_start_data, SEEK_SET);
  //   // Consume this line, which is "# Data section follows"
  //   [[maybe_unused]] auto const ret0 = fgets(buffer.data(), 2047, fp);
  //   // Consume the next line, which is "@1"
  //   [[maybe_unused]] auto const ret1 = fgets(buffer.data(), 2047, fp);
  //
  //   // Read the data
  //   // - how much to read
  //   auto const num_to_read =
  //       static_cast<std::size_t>(dims[0] * dims[1] * dims[2] * num_comps);
  //   // - prepare memory; use malloc() if you're using pure C
  //   // - do it
  //   data.resize(num_to_read);
  //   std::size_t const act_read =
  //       fread((void*)data.data(), sizeof(float), num_to_read, fp);
  //   // - ok?
  //   if (num_to_read != act_read) {
  //     fclose(fp);
  //     throw std::runtime_error(
  //         "Something went wrong while reading the binary data "
  //         "section. Premature end of file?");
  //   }
  // }
  //
  // fclose(fp);
  // if constexpr (is_float<T>) {
  //   return std::tuple{std::move(data), std::move(dims), std::move(aabb),
  //                     num_comps};
  // } else {
  //   return std::tuple{std::vector<T>(begin(data), end(data)),
  //   std::move(dims),
  //                     std::move(aabb), num_comps};
  // }
}
////------------------------------------------------------------------------------
///// A simple routine to read an AmiraMesh file
///// that defines a scalar/vector field on a uniform grid.
// template <floating_point T = float>
// auto read(filesystem::path const& path) {
//   auto fp = fopen(path.string().c_str(), "rb");
//   if (!fp) {
//     throw std::runtime_error("could not open file " + path.string());
//   }
//
//   // We read the first 2k bytes into memory to parse the header.
//   // The fixed buffer size looks a bit like a hack, and it is one, but it
//   gets
//   // the job done.
//   auto buffer = std::string(2048, ' ');
//
//   [[maybe_unused]] auto const ret = fread(buffer.data(), sizeof(char), 2047,
//   fp); buffer.back() =
//       '\0';  // The following string routines prefer null-terminated strings
//
//   // if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1")) {
//   //   fclose(fp);
//   //   throw std::runtime_error("not a proper amira mesh file");
//   // }
//
//   // Find the Lattice definition, i.e., the dimensions of the uniform grid
//   auto dims = std::array<std::size_t, 3>{};
//   std::sscanf(find_and_jump(buffer, "define Lattice"), "%zu %zu %zu",
//   &dims[0],
//          &dims[1], &dims[2]);
//
//   // Find the boundingbox
//   auto aabb = axis_aligned_bounding_box<float, 3>{};
//   std::sscanf(find_and_jump(buffer, "BoundingBox"), "%g %g %g %g %g %g",
//          &aabb.min(0), &aabb.max(0), &aabb.min(1), &aabb.max(1),
//          &aabb.min(2), &aabb.max(2));
//
//   // Is it a uniform grid? We need this only for the sanity check below.
//   bool const is_uniform =
//       (std::strstr(buffer.data(), "CoordType \"uniform\"") != nullptr);
//
//   // Type of the field: scalar, vector
//   int num_comps{0};
//   if (std::strstr(buffer.data(), "Lattice { float Data }"))
//     // Scalar field
//     num_comps = 1;
//   else{
//     // A field with more than one component, i.e., a vector field
//     std::sscanf(find_and_jump(buffer, "Lattice { float["), "%d", &num_comps);
//   }
//
//   // Sanity check
//   if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 ||
//       aabb.min(0) > aabb.max(0) || aabb.min(1) > aabb.max(1) ||
//       aabb.min(2) > aabb.max(2) || !is_uniform || num_comps <= 0) {
//     fclose(fp);
//     throw std::runtime_error("something went wrong");
//   }
//
//   auto data = std::vector<float>{};
//   // Find the beginning of the data section
//   long const idx_start_data = std::strstr(buffer.data(), "# Data section
//   follows") - buffer.data(); if (idx_start_data > 0) {
//     // Set the file pointer to the beginning of "# Data section follows"
//     fseek(fp, idx_start_data, SEEK_SET);
//     // Consume this line, which is "# Data section follows"
//     [[maybe_unused]] auto const ret0 = fgets(buffer.data(), 2047, fp);
//     // Consume the next line, which is "@1"
//     [[maybe_unused]] auto const ret1 = fgets(buffer.data(), 2047, fp);
//
//     // Read the data
//     // - how much to read
//     auto const num_to_read =
//         static_cast<std::size_t>(dims[0] * dims[1] * dims[2] * num_comps);
//     // - prepare memory; use malloc() if you're using pure C
//     // - do it
//     data.resize(num_to_read);
//     std::size_t const act_read =
//         fread((void*)data.data(), sizeof(float), num_to_read, fp);
//     // - ok?
//     if (num_to_read != act_read) {
//       fclose(fp);
//       throw std::runtime_error(
//           "Something went wrong while reading the binary data "
//           "section. Premature end of file?");
//     }
//   }
//
//   fclose(fp);
//   if constexpr (is_float<T>) {
//     return std::tuple{std::move(data), std::move(dims), std::move(aabb),
//                       num_comps};
//   } else {
//     return std::tuple{std::vector<T>(begin(data), end(data)),
//     std::move(dims),
//                       std::move(aabb), num_comps};
//   }
// }
//==============================================================================
}  // namespace tatooine::amira
//==============================================================================
#endif
