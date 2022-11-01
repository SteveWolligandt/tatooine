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
  std::array<std::size_t, 3> whole_extent1;
  std::array<std::size_t, 3> whole_extent2;
  explicit structured_grid(reader& r, rapidxml::xml_node<>* node)
      : piece_set{r, node} {
    if (auto const attr = node->first_attribute("WholeExtent");
        attr != nullptr) {
      auto ss = std::stringstream{attr->value()};
      ss >> whole_extent1[0] >> whole_extent2[0] >> whole_extent1[1] >>
          whole_extent2[1] >> whole_extent1[2] >> whole_extent2[2];
    }
  }
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
