#include <tatooine/vtk/xml/data_attribute.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
auto to_data_attribute(char const* str) -> data_attribute {
  if (std::strcmp(str, "Scalars") == 0) {
    return data_attribute::scalars;
  }
  if (std::strcmp(str, "Vectors") == 0) {
    return data_attribute::vectors;
  }
  if (std::strcmp(str, "Normals") == 0) {
    return data_attribute::normals;
  }
  if (std::strcmp(str, "Tensors") == 0) {
    return data_attribute::tensors;
  }
  if (std::strcmp(str, "TCoords") == 0) {
    return data_attribute::tcoords;
  }
  return data_attribute::unknown;
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
