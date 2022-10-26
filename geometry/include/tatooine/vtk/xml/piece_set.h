#ifndef TATOOINE_GEOMETRY_VTK_XML_PIECE_SET_H
#define TATOOINE_GEOMETRY_VTK_XML_PIECE_SET_H
//==============================================================================
#include <tatooine/vtk/xml/piece.h>

#include <rapidxml.hpp>
#include <vector>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct reader;
struct piece_set {
  std::vector<piece> pieces;
  explicit piece_set(reader& r, rapidxml::xml_node<>* node);
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
