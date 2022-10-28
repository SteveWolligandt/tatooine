#ifndef TATOOINE_DETAIL_STRUCTURED_GRID_VTS_WRITER_H
#define TATOOINE_DETAIL_STRUCTURED_GRID_VTS_WRITER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/filesystem.h>
#include <tatooine/linspace.h>
#include <tatooine/vtk/xml.h>

#include <array>
#include <vector>
//==============================================================================
namespace tatooine::detail::structured_grid {
//==============================================================================
template <typename Grid, unsigned_integral HeaderType>
struct vts_writer {
  //using vertex_property_type = typename Grid::vertex_property_type;
  //template <typename T>
  //using typed_vertex_property_type =
  //    typename Grid::template typed_vertex_property_type<T>;
  //----------------------------------------------------------------------------
  Grid const& m_grid;
  //----------------------------------------------------------------------------
  auto write(filesystem::path const& path) const {
    auto file = std::ofstream{path};
    if (!file.is_open()) {
      throw std::runtime_error{"Could open file " + path.string() +
                               " for writing."};
    }
    auto offset = std::size_t{};
    write_vtk_file(file, offset);
  }
  //----------------------------------------------------------------------------
 private:
  auto write_vtk_file(std::ofstream& file, auto& offset) const {
    file << "<VTKFile"
         << " type=\"StructuredGrid\""
         << " version=\"1.0\""
         << " byte_order=\"LittleEndian\""
         << " header_type=\"" << tatooine::vtk::xml::to_data_type<HeaderType>()
         << "\">\n";
    write_structured_grid(file, offset);
    write_appended_data(file);
    file << "</VTKFile>";
  }
  //----------------------------------------------------------------------------
  auto write_structured_grid(std::ofstream& file, auto& offset) const {
    file << "  <StructuredGrid WholeExtent=\"" << 0 << " " << m_grid.size(0) - 1
         << " " << 0 << " " << m_grid.size(1) - 1;
    if (m_grid.num_dimensions() == 3) {
      file << " " << 0 << " " << m_grid.size(2) - 1;
    } else if (m_grid.num_dimensions() == 2) {
      file << " " << 0 << " " << 0;
    }
    file << "\">\n";
    write_piece(file, offset);
    file << "  </StructuredGrid>\n";
  }
  //----------------------------------------------------------------------------
  auto write_piece(std::ofstream& file, auto& offset) const {
    file << "    <Piece Extent=\"";
    file << "0 " << m_grid.size(0) - 1;
    file << " 0 " << m_grid.size(1) - 1;
    if (m_grid.num_dimensions() == 3) {
      file << " 0 " << m_grid.size(2) - 1;
    } else if (m_grid.num_dimensions() == 2) {
      file << " 0 0";
    }
    file << "\">\n";
    write_points(file, offset);
    write_point_data(file, offset);
    write_cell_data(file, offset);
    file << "    </Piece>\n";
  }
  //----------------------------------------------------------------------------
  auto write_points(std::ofstream& file, std::size_t& offset) const {
    file << "      <Points>\n";
    file << "        <DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\"" << vtk::xml::to_data_type<typename Grid::real_type>()
         << "\" NumberOfComponents=\"3\"/>\n";
    file << "      </Points>\n";
    auto const num_bytes = HeaderType(sizeof(typename Grid::real_type) * 3 *
                                      m_grid.num_components());
    offset += num_bytes + sizeof(HeaderType);
  }
  //------------------------------------------------------------------------------
  auto write_point_data(std::ofstream& file, std::size_t& offset) const {
    file << "      <PointData>\n";
    // TODO
    file << "      </PointData>\n";
  }
  //------------------------------------------------------------------------------
  auto write_cell_data(std::ofstream& file, std::size_t& /*offset*/) const {
    file << "      <CellData>\n";
    // TODO
    file << "      </CellData>\n";
  }
  //----------------------------------------------------------------------------
  auto write_appended_data(std::ofstream& file) const {
    file << "  <AppendedData encoding=\"raw\">\n   _";
    write_appended_data_points(file);
    write_appended_data_point_data(file);
    write_appended_data_cell_data(file);
    file << "\n  </AppendedData>\n";
  }
  //----------------------------------------------------------------------------
  auto write_appended_data_points(std::ofstream& file) const {}
  //----------------------------------------------------------------------------
  auto write_appended_data_point_data(std::ofstream& file) const {
    //for (auto const& [name, prop] : m_grid.vertex_properties()) {
    //  write_vertex_property_appended_data<
    //      HeaderType, float, vec2f, vec3f, vec4f, mat2f, mat3f, mat4f, double,
    //      vec2d, vec3d, vec4d, mat2d, mat3d, mat4d>(*prop, file);
    //}
  }
  //----------------------------------------------------------------------------
  auto write_appended_data_cell_data(std::ofstream& /*file*/) const {
    // TODO
  }
  ////----------------------------------------------------------------------------
  //template <typename... Ts>
  //auto write_vertex_property_appended_data(vertex_property_type const& prop,
  //                                         std::ofstream& file) const {
  //  invoke([&] {
  //    if (prop.type() == typeid(Ts)) {
  //      // TODO
  //    }
  //  }...);
  //}
};
//==============================================================================
}  // namespace tatooine::detail::structured_grid
//==============================================================================
#endif
