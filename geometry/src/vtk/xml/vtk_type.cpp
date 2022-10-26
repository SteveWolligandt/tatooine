#include <tatooine/vtk/xml/vtk_type.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
auto parse_vtk_type(char const* str) -> vtk_type {
  if (std::strcmp(str, "ImageData") == 0) {
    return vtk_type::image_data;
  }
  if (std::strcmp(str, "RectilinearGrid") == 0) {
    return vtk_type::rectilinear_grid;
  }
  if (std::strcmp(str, "StructuredGrid") == 0) {
    return vtk_type::structured_grid;
  }
  if (std::strcmp(str, "PolyData") == 0) {
    return vtk_type::poly_data;
  }
  if (std::strcmp(str, "UnstructuredGrid") == 0) {
    return vtk_type::unstructured_grid;
  }
  return vtk_type::unknown;
}
//------------------------------------------------------------------------------
auto to_string(vtk_type const t) -> std::string_view {
  switch (t) {
    case vtk_type::image_data:
      return "ImageData";
    case vtk_type::rectilinear_grid:
      return "RectilinearGrid";
    case vtk_type::structured_grid:
      return "StructuredGrid";
    case vtk_type::poly_data:
      return "PolyData";
    case vtk_type::unstructured_grid:
      return "UnstructuredGrid";
    default:
      return "UnknownVTKType";
  }
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
