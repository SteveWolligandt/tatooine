#ifndef __TATOOINE_AMIRA_READER_H__
#define __TATOOINE_AMIRA_READER_H__

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "boundingbox.h"
#include "tensor.h"

//==============================================================================
namespace tatooine::amira {
//==============================================================================

//! Find a string in the given buffer and return a pointer
//!    to the contents directly behind the search_string.
//!    If not found, return the buffer. A subsequent sscanf()
//!    will fail then, but at least we return a decent pointer.
inline const char* find_and_jump(const char* buffer,
                                 const char* search_string) {
  const char* found_loc = strstr(buffer, search_string);
  if (found_loc) return found_loc + strlen(search_string);
  return buffer;
}

//! A simple routine to read an AmiraMesh file
//! that defines a scalar/vector field on a uniform grid.
inline auto read(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) throw std::runtime_error("could not open file " + filename);

  // We read the first 2k bytes into memory to parse the header.
  // The fixed buffer size looks a bit like a hack, and it is one, but it gets
  // the job done.
  char buffer[2048];
  fread(buffer, sizeof(char), 2047, fp);
  buffer[2047] =
      '\0';  // The following string routines prefer null-terminated strings

  // if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1")) {
  //   fclose(fp);
  //   throw std::runtime_error("not a proper amira mesh file");
  // }

  // Find the Lattice definition, i.e., the dimensions of the uniform grid
  std::array<size_t, 3> dims;
  sscanf(find_and_jump(buffer, "define Lattice"), "%zu %zu %zu", &dims[0],
         &dims[1], &dims[2]);

  // Find the boundingbox
  boundingbox<float, 3> bb;
  sscanf(find_and_jump(buffer, "BoundingBox"), "%g %g %g %g %g %g", &bb.min(0),
         &bb.max(0), &bb.min(1), &bb.max(1), &bb.min(2), &bb.max(2));

  // Is it a uniform grid? We need this only for the sanity check below.
  const bool b_is_uniform = (strstr(buffer, "CoordType \"uniform\"") != NULL);

  // Type of the field: scalar, vector
  int num_comps{0};
  if (strstr(buffer, "Lattice { float Data }"))
    // Scalar field
    num_comps = 1;
  else
    // A field with more than one component, i.e., a vector field
    sscanf(find_and_jump(buffer, "Lattice { float["), "%d", &num_comps);

  // Sanity check
  if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 || bb.min(0) > bb.max(0) ||
      bb.min(1) > bb.max(1) || bb.min(2) > bb.max(2) || !b_is_uniform ||
      num_comps <= 0) {
    fclose(fp);
    throw std::runtime_error("something went wrong");
  }

  std::vector<float> data;
  // Find the beginning of the data section
  const long idx_start_data = strstr(buffer, "# Data section follows") - buffer;
  if (idx_start_data > 0) {
    // Set the file pointer to the beginning of "# Data section follows"
    fseek(fp, idx_start_data, SEEK_SET);
    // Consume this line, which is "# Data section follows"
    fgets(buffer, 2047, fp);
    // Consume the next line, which is "@1"
    fgets(buffer, 2047, fp);

    // Read the data
    // - how much to read
    const size_t num_to_read = dims[0] * dims[1] * dims[2] * num_comps;
    // - prepare memory; use malloc() if you're using pure C
    // - do it
    data.resize(num_to_read);
    const size_t act_read =
        fread((void*)data.data(), sizeof(float), num_to_read, fp);
    // - ok?
    if (num_to_read != act_read) {
      fclose(fp);
      throw std::runtime_error(
          "Something went wrong while reading the binary data "
          "section. Premature end of file?");
    }
  }

  fclose(fp);
  return std::tuple{std::move(data), std::move(dims), std::move(bb), num_comps};
}

//==============================================================================
}  // namespace tatooine::amira
//==============================================================================

#endif
