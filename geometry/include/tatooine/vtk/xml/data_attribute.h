#ifndef TATOOINE_VTK_XML_DATA_ATTRIBUTE_H
#define TATOOINE_VTK_XML_DATA_ATTRIBUTE_H
//==============================================================================
#include <cstring>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
enum class data_attribute {
  scalars,
  vectors,
  normals,
  tensors,
  tcoords,
  unknown
};
//==============================================================================
auto to_data_attribute(char const* str) -> data_attribute;
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
