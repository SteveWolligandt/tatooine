#ifndef TATOOINE_GEOMETRY_VTK_XML_POLY_DATA_H
#define TATOOINE_GEOMETRY_VTK_XML_POLY_DATA_H
//==============================================================================
#include <tatooine/vtk/xml/piece_set.h>

#include <rapidxml.hpp>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct reader;
struct poly_data : piece_set {
  explicit poly_data(reader& r, rapidxml::xml_node<>* node)
      : piece_set{r, node} {}
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
