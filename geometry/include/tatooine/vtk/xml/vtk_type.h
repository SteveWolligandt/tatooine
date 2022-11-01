#ifndef TATOOINE_GEOMETRY_VTK_XML_VTK_TYPE_H
#define TATOOINE_GEOMETRY_VTK_XML_VTK_TYPE_H
//==============================================================================
#include <tatooine/concepts.h>

#include <cstring>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
enum class vtk_type {
  image_data,
  rectilinear_grid,
  structured_grid,
  poly_data,
  unstructured_grid,
  unknown
};
//------------------------------------------------------------------------------
auto parse_vtk_type(char const* str) -> vtk_type;
auto to_string(vtk_type const t) -> std::string_view;
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
