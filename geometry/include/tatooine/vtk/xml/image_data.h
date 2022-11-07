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
  std::array<std::size_t, 3> whole_extent1;
  std::array<std::size_t, 3> whole_extent2;
  std::array<double, 3>      origin;
  std::array<double, 3>      spacing;
  explicit image_data(reader& r, rapidxml::xml_node<>* node)
      : piece_set{r, node} {
    if (auto const attr = node->first_attribute("WholeExtent");
        attr != nullptr) {
      auto ss = std::stringstream{attr->value()};
      ss >> whole_extent1[0] >> whole_extent2[0] >> whole_extent1[1] >>
          whole_extent2[1] >> whole_extent1[2] >> whole_extent2[2];
    }
    if (auto const attr = node->first_attribute("Origin"); attr != nullptr) {
      auto ss = std::stringstream{attr->value()};
      ss >> origin[0] >> origin[0] >> origin[1];
    }
    if (auto const attr = node->first_attribute("Spacing"); attr != nullptr) {
      auto ss = std::stringstream{attr->value()};
      ss >> spacing[0] >> spacing[0] >> spacing[1];
    }
  }
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
