#include <tatooine/parse.h>
#include <tatooine/vtk/xml/piece.h>
#include <tatooine/vtk/xml/reader.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
piece::piece(reader& r, rapidxml::xml_node<>* node) {
  auto const num_points_attr = node->first_attribute("NumberOfPoints");
  auto const num_verts_attr  = node->first_attribute("NumberOfVerts");
  auto const num_lines_attr  = node->first_attribute("NumberOfLines");
  auto const num_cells_attr  = node->first_attribute("NumberOfCells");
  auto const num_strips_attr = node->first_attribute("NumberOfStrips");
  auto const num_polys_attr  = node->first_attribute("NumberOfPolys");
  auto const extent_attr     = node->first_attribute("Extent");
  if (num_points_attr != nullptr) {
    num_points = parse<std::size_t>(num_points_attr->value());
  }
  if (num_verts_attr != nullptr) {
    num_vertices = parse<std::size_t>(num_verts_attr->value());
  }
  if (num_lines_attr != nullptr) {
    num_lines = parse<std::size_t>(num_lines_attr->value());
  }
  if (num_cells_attr != nullptr) {
    num_cells = parse<std::size_t>(num_cells_attr->value());
  }
  if (num_strips_attr != nullptr) {
    num_strips = parse<std::size_t>(num_strips_attr->value());
  }
  if (num_polys_attr != nullptr) {
    num_polygons = parse<std::size_t>(num_polys_attr->value());
  }
  if (extent_attr != nullptr) {
    extent1 = std::array<std::size_t, 3>{};
    extent2 = std::array<std::size_t, 3>{};
    auto ss = std::stringstream{extent_attr->value()};
    ss >> extent1->at(0) >> extent2->at(0) >> extent1->at(1) >>
          extent2->at(1) >> extent1->at(2) >> extent2->at(2);
  }

  read_points(r, node);
  read_point_data(r, node);
  read_cell_data(r, node);
  read_vertices(r, node);
  read_lines(r, node);
  read_cells(r, node);
  read_strips(r, node);
  read_polygons(r, node);
}
//------------------------------------------------------------------------------
auto piece::read_data_array_set(reader& r, rapidxml::xml_node<>* node,
                                std::map<std::string, data_array>& set)
    -> void {
  for (auto* data_array_node = node->first_node(); data_array_node != nullptr;
       data_array_node       = data_array_node->next_sibling()) {
    auto da = data_array{r, data_array_node};
    set.insert(std::pair{*da.name(), da});
  }
}
//------------------------------------------------------------------------------
auto piece::read_points(reader& r, rapidxml::xml_node<>* node) -> void {
  auto const points_node = node->first_node("Points");
  if (points_node != nullptr) {
    points = data_array{r, points_node->first_node()};
  }
}
//------------------------------------------------------------------------------
auto piece::read_vertices(reader& r, rapidxml::xml_node<>* node) -> void {
  auto const vertices_node = node->first_node("Vertices");
  if (vertices_node != nullptr) {
    read_data_array_set(r, vertices_node, vertices);
  }
}
//------------------------------------------------------------------------------
auto piece::read_lines(reader& r, rapidxml::xml_node<>* node) -> void {
  auto const lines_node = node->first_node("Lines");
  if (lines_node != nullptr) {
    read_data_array_set(r, lines_node, lines);
  }
}
//------------------------------------------------------------------------------
auto piece::read_cells(reader& r, rapidxml::xml_node<>* node) -> void {
  auto const cells_node = node->first_node("Cells");
  if (cells_node != nullptr) {
    read_data_array_set(r, cells_node, cells);
  }
}
//------------------------------------------------------------------------------
auto piece::read_strips(reader& r, rapidxml::xml_node<>* node) -> void {
  auto const strips_node = node->first_node("Strips");
  if (strips_node != nullptr) {
    read_data_array_set(r, strips_node, strips);
  }
}
//------------------------------------------------------------------------------
auto piece::read_polygons(reader& r, rapidxml::xml_node<>* node) -> void {
  auto const polygons_node = node->first_node("Polys");
  if (polygons_node != nullptr) {
    read_data_array_set(r, polygons_node, polygons);
  }
}
//------------------------------------------------------------------------------
auto piece::read_point_data(reader& r, rapidxml::xml_node<>* node) -> void {
  auto const point_data_node = node->first_node("PointData");
  if (point_data_node != nullptr) {
    read_data_array_set(r, point_data_node, point_data);
  }
}
//------------------------------------------------------------------------------
auto piece::read_cell_data(reader& r, rapidxml::xml_node<>* node) -> void {
  auto const cell_data_node = node->first_node("CellData");
  if (cell_data_node != nullptr) {
    read_data_array_set(r, cell_data_node, cell_data);
  }
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
