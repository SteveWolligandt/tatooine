#ifndef TATOOINE_GEOMETRY_VTK_XML_IMAGE_DATA_H
#define TATOOINE_GEOMETRY_VTK_XML_IMAGE_DATA_H
//==============================================================================
#include <tatooine/vtk/xml/piece_set.h>

#include <array>
#include <rapidxml.hpp>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct reader;
struct image_data : piece_set {
  std::array<double, 3> whole_extent1;
  std::array<double, 3> whole_extent2;
  std::array<double, 3> origin;
  std::array<double, 3> spacing;
  explicit image_data(reader& r, rapidxml::xml_node<>* node)
      : piece_set{r, node} {}
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
