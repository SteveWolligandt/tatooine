#ifndef TATOOINE_GEOMETRY_VTK_XML_PIECE_H
#define TATOOINE_GEOMETRY_VTK_XML_PIECE_H
//==============================================================================
#include <tatooine/vtk/xml/data_array.h>

#include <map>
#include <optional>
#include <rapidxml.hpp>
#include <vector>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct reader;
struct piece {
  std::size_t num_points   = {};
  std::size_t num_vertices = {};
  std::size_t num_lines    = {};
  std::size_t num_cells    = {};
  std::size_t num_strips   = {};
  std::size_t num_polygons = {};

  std::optional<std::array<double, 3>> extent1 = {};
  std::optional<std::array<double, 3>> extent2 = {};

  data_array                        points     = {};
  std::map<std::string, data_array> vertices   = {};
  std::map<std::string, data_array> lines      = {};
  std::map<std::string, data_array> cells      = {};
  std::map<std::string, data_array> strips     = {};
  std::map<std::string, data_array> polygons   = {};
  std::map<std::string, data_array> point_data = {};
  std::map<std::string, data_array> cell_data  = {};

  explicit piece(reader& r, rapidxml::xml_node<>* node);
  auto read_data_array_set(reader& r, rapidxml::xml_node<>* node,
                           std::map<std::string, data_array>& set) -> void;
  auto read_points(reader& r, rapidxml::xml_node<>* node) -> void;
  auto read_vertices(reader& r, rapidxml::xml_node<>* node) -> void;
  auto read_lines(reader& r, rapidxml::xml_node<>* node) -> void;
  auto read_cells(reader& r, rapidxml::xml_node<>* node) -> void;
  auto read_strips(reader& r, rapidxml::xml_node<>* node) -> void;
  auto read_polygons(reader& r, rapidxml::xml_node<>* node) -> void;
  auto read_point_data(reader& r, rapidxml::xml_node<>* node) -> void;
  auto read_cell_data(reader& r, rapidxml::xml_node<>* node) -> void;
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
