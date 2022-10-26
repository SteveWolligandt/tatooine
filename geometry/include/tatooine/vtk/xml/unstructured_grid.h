#ifndef TATOOINE_GEOMETRY_VTK_XML_UNSTRUCTURED_GRID_H
#define TATOOINE_GEOMETRY_VTK_XML_UNSTRUCTURED_GRID_H
//==============================================================================
#include <tatooine/vtk/xml/piece_set.h>

#include <rapidxml.hpp>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct reader;
struct unstructured_grid : piece_set {
  explicit unstructured_grid(reader& r, rapidxml::xml_node<>* node)
      : piece_set{r, node} {}
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
