#include <tatooine/vtk/xml/piece_set.h>
#include <tatooine/vtk/xml/reader.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
piece_set::piece_set(reader& r, rapidxml::xml_node<>* node) {
  for (auto* piece_node = node->first_node(); piece_node != nullptr;
       piece_node       = piece_node->next_sibling()) {
    pieces.emplace_back(r, piece_node);
  }
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
