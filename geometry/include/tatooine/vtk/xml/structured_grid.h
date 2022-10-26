#ifndef TATOOINE_GEOMETRY_VTK_XML_STRUCTURED_GRID_H
#define TATOOINE_GEOMETRY_VTK_XML_STRUCTURED_GRID_H
//==============================================================================
#include <tatooine/vtk/xml/piece_set.h>

#include <array>
#include <rapidxml.hpp>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct reader;
struct structured_grid : piece_set {
  std::array<double, 3> whole_extent1;
  std::array<double, 3> whole_extent2;
  explicit structured_grid(reader& r, rapidxml::xml_node<>* node)
      : piece_set{r, node} {}
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
