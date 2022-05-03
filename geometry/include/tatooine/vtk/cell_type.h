#ifndef TATOOINE_VTK_CELL_TYPE_H
#define TATOOINE_VTK_CELL_TYPE_H
//==============================================================================
#include <string>
//==============================================================================
namespace tatooine::vtk {
//==============================================================================
enum class cell_type : std::uint8_t {
  vertex            = 1,
  poly_vertex       = 2,
  line              = 3,
  poly_line         = 4,
  triangle          = 5,
  triangle_strip    = 6,
  polygon           = 7,
  pixel             = 8,
  quad              = 9,
  tetra             = 10,
  voxel             = 11,
  hexahedron        = 12,
  wedge             = 13,
  pyramid           = 14,
  unknown_cell_type = 0,
};
//-----------------------------------------------------------------------------
constexpr auto to_string_view(cell_type const ct) -> std::string_view {
  switch (ct) {
    case cell_type::vertex:
      return "VERTEX";
    case cell_type::poly_vertex:
      return "POLY_VERTEX";
    case cell_type::line:
      return "LINE";
    case cell_type::poly_line:
      return "POLY_LINE";
    case cell_type::triangle:
      return "TRIANGLE";
    case cell_type::triangle_strip:
      return "TRIANGLE_STRIP";
    case cell_type::polygon:
      return "POLYGON";
    case cell_type::pixel:
      return "PIXEL";
    case cell_type::quad:
      return "QUAD";
    case cell_type::tetra:
      return "TETRA";
    case cell_type::voxel:
      return "VOXEL";
    case cell_type::hexahedron:
      return "HEXAHEDRON";
    case cell_type::wedge:
      return "WEDGE";
    case cell_type::pyramid:
      return "PYRAMID";
    default:
    case cell_type::unknown_cell_type:
      return "UNKNOWN";
  }
}
//-----------------------------------------------------------------------------
auto parse_cell_type(std::string const &) -> cell_type;
//==============================================================================
}  // namespace tatooine::vtk
//==============================================================================
#endif
