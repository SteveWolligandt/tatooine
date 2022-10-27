#ifndef TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_TRIANGULAR_VTU_WRITER_H
#define TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_TRIANGULAR_VTU_WRITER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/filesystem.h>
#include <tatooine/linspace.h>
#include <tatooine/vtk/xml.h>

#include <array>
#include <vector>
//==============================================================================
namespace tatooine::detail::unstructured_simplicial_grid {
//==============================================================================
template <typename Grid, unsigned_integral HeaderType = std::uint64_t,
          integral          ConnectivityInt = std::int64_t,
          integral          OffsetInt       = std::int64_t,
          unsigned_integral CellTypesInt    = std::uint8_t>
requires(Grid::num_dimensions() == 2 ||
         Grid::num_dimensions() == 3)
struct triangular_vtu_writer {
  static auto constexpr num_dimensions() { return Grid::num_dimensions(); }
  using vertex_property_type = typename Grid::vertex_property_type;
  template <typename T>
  using typed_vertex_property_type =
      typename Grid::template typed_vertex_property_type<T>;
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
         << " type=\"UnstructuredGrid\""
         << " version=\"1.0\""
         << " byte_order=\"LittleEndian\""
         << " header_type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<HeaderType>())
         << "\">\n";
    write_unstructured_grid(file, offset);
    write_appended_data(file);
    file << "</VTKFile>";
  }
  //----------------------------------------------------------------------------
  auto write_unstructured_grid(std::ofstream& file, auto& offset) const {
    file << "  <UnstructuredGrid>\n";
    write_piece(file, offset);
    file << "  </UnstructuredGrid>\n";
  }
  //----------------------------------------------------------------------------
  auto write_piece(std::ofstream& file, auto& offset) const {
    file << "    <Piece"
         << " NumberOfPoints=\"" << m_grid.vertices().size() << "\""
         << " NumberOfCells=\"" << m_grid.simplices().size() << "\""
         << ">\n";
    write_points(file, offset);
    write_cells(file, offset);
    write_point_data(file, offset);
    write_cell_data(file, offset);
    file << "    </Piece>\n";
  }
  //------------------------------------------------------------------------------
  auto write_point_data(std::ofstream& file, std::size_t& offset) const {
    file << "      <PointData>\n";
    for (auto const& [name, prop] : m_grid.vertex_properties()) {
      write_vertex_property_data_array<HeaderType, float, vec2f, vec3f, vec4f,
                                       mat2f, mat3f, mat4f, double, vec2d,
                                       vec3d, vec4d, mat2d, mat3d, mat4d>(
          name, *prop, file, offset);
    }
    file << "      </PointData>\n";
  }
  //------------------------------------------------------------------------------
  auto write_cell_data(std::ofstream& file, std::size_t& /*offset*/) const {
    file << "      <CellData>\n";
    file << "      </CellData>\n";
  }
  //----------------------------------------------------------------------------
  auto write_points(std::ofstream& file, std::size_t& offset) const {
    file << "      <Points>\n"
         << "        <DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<typename Grid::real_type>())
         << "\" NumberOfComponents=\"3\"/>\n"
         << "      </Points>\n";
    auto const num_bytes = HeaderType(sizeof(typename Grid::real_type) * 3 *
                                      m_grid.vertices().size());
    offset += num_bytes + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  auto write_cells(std::ofstream& file, std::size_t& offset) const {
    write_connectivity(file, offset);
    write_offsets(file, offset);
    write_cell_types(file, offset);
  }
  //----------------------------------------------------------------------------
  auto write_connectivity(std::ofstream& file, std::size_t& offset) const {
    // connectivity
    file << "      <Cells>\n"
         << "        <DataArray format=\"appended\" offset=\"" << offset
         << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<ConnectivityInt>())
         << "\" Name=\"connectivity\"/>\n";
    HeaderType num_bytes = m_grid.simplices().size() *
                           m_grid.num_vertices_per_simplex() *
                           sizeof(ConnectivityInt);
    offset += num_bytes + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  auto write_offsets(std::ofstream& file, std::size_t& offset) const {
    // offsets
    file << "        <DataArray format=\"appended\" offset=\"" << offset
         << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<OffsetInt>())
         << "\" Name=\"offsets\"/>\n";
    auto const num_bytes = sizeof(OffsetInt) * (m_grid.simplices().size());
    offset += num_bytes + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  auto write_cell_types(std::ofstream& file, std::size_t& offset) const {
    // types
    file << "        <DataArray format=\"appended\" offset=\"" << offset
         << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<CellTypesInt>())
         << "\" Name=\"types\"/>\n";
    file << "      </Cells>\n";
    auto const num_bytes=
        sizeof(CellTypesInt) * m_grid.simplices().size();
    offset += num_bytes + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  auto write_appended_data(std::ofstream& file) const {
    file << "  <AppendedData encoding=\"raw\">\n   _";
    write_appended_data_points(file);
    write_appended_data_cells(file);
    write_appended_data_point_data(file);
    write_appended_data_cell_data(file);
    file << "\n  </AppendedData>\n";
  }
  //----------------------------------------------------------------------------
  auto write_appended_data_points(std::ofstream& file) const {
    using namespace std::ranges;
    auto const num_bytes = HeaderType(sizeof(typename Grid::real_type) * 3 *
                                      m_grid.vertices().size());
    file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(HeaderType));
    if constexpr (Grid::num_dimensions() == 2) {
      auto point_data = std::vector<vec<typename Grid::real_type, 3>>{};
      point_data.reserve(m_grid.vertices().size());
      auto           position = [this](auto const v) -> auto& { return at(v); };
      constexpr auto to_3d    = [](auto const& p) {
        return vec{p.x(), p.y(), typename Grid::real_type(0)};
      };
      for (auto const v : m_grid.vertices()) {
        point_data.push_back(to_3d(m_grid[v]));
      }

      file.write(reinterpret_cast<char const*>(point_data.data()), num_bytes);
    } else if constexpr (Grid::num_dimensions() == 3) {
      file.write(reinterpret_cast<char const*>(m_grid.vertices().data()),
                 num_bytes);
    }
  }
  //------------------------------------------------------------------------------
  auto write_appended_data_cells(std::ofstream& file) const {
    write_appended_data_cells_connectivity(file);
    write_appended_data_cells_offsets(file);
    write_appended_data_cells_types(file);
  }
  //------------------------------------------------------------------------------
  auto write_appended_data_cells_connectivity(std::ofstream& file) const {
    using namespace std::ranges;
    auto connectivity_data = std::vector<ConnectivityInt>(
        m_grid.simplices().size() * m_grid.num_vertices_per_simplex());
    auto index = [](auto const x) -> ConnectivityInt { return x.index(); };
    copy(m_grid.simplices().data_container() | views::transform(index),
         begin(connectivity_data));
    HeaderType num_bytes = m_grid.simplices().size() *
                           m_grid.num_vertices_per_simplex() *
                           sizeof(ConnectivityInt);
    file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(HeaderType));
    file.write(reinterpret_cast<char const*>(connectivity_data.data()),
               num_bytes);
  }
  //------------------------------------------------------------------------------
  auto write_appended_data_cells_offsets(std::ofstream& file) const {
    auto offsets = std::vector<OffsetInt>(m_grid.simplices().size(),
                                          m_grid.num_vertices_per_simplex());
    for (std::size_t i = 1; i < size(offsets); ++i) {
      offsets[i] += offsets[i - 1];
    };
    auto const num_bytes = sizeof(OffsetInt) * m_grid.simplices().size();
    file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(HeaderType));
    file.write(reinterpret_cast<char const*>(offsets.data()), num_bytes);
  }
  //------------------------------------------------------------------------------
  auto write_appended_data_cells_types(std::ofstream& file) const {
    auto cell_types = std::vector<CellTypesInt>(m_grid.simplices().size(), 5);
    auto const num_bytes = sizeof(CellTypesInt) * m_grid.simplices().size();
    file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(HeaderType));
    file.write(reinterpret_cast<char const*>(cell_types.data()), num_bytes);
  }
  //----------------------------------------------------------------------------
  auto write_appended_data_point_data(std::ofstream& file) const {
    for (auto const& [name, prop] : m_grid.vertex_properties()) {
      write_vertex_property_appended_data<
          HeaderType, float, vec2f, vec3f, vec4f, mat2f, mat3f, mat4f, double,
          vec2d, vec3d, vec4d, mat2d, mat3d, mat4d>(*prop, file);
    }
  }
  //----------------------------------------------------------------------------
  auto write_appended_data_cell_data(std::ofstream& /*file*/) const {}
  //----------------------------------------------------------------------------
  template <typename... Ts>
  auto write_vertex_property_appended_data(vertex_property_type const& prop,
                                           std::ofstream& file) const {
    (
        [&] {
          if (prop.type() == typeid(Ts)) {
            write_vertex_property_appended_data(
                prop.template cast_to_typed<Ts>(), file);
          }
        }(),
        ...);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto write_vertex_property_appended_data(
      typed_vertex_property_type<T> const& prop, std::ofstream& file) const {
    if constexpr (tensor_rank<T> <= 1) {
      auto data = std::vector<T>{};
      for (auto const v : m_grid.vertices()) {
        data.push_back(prop[v]);
      };
      auto const num_bytes =
          HeaderType(sizeof(tensor_value_type<T>) * tensor_num_components<T> *
                     m_grid.vertices().size());
      file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(HeaderType));
      file.write(reinterpret_cast<char const*>(data.data()), num_bytes);
    } else if constexpr (tensor_rank<T> == 2) {
      auto const num_bytes =
          HeaderType(sizeof(tensor_value_type<T>) * tensor_num_components<T> *
                     m_grid.vertices().size() / tensor_dimension<T, 0>);
      for (std::size_t i = 0; i < tensor_dimension<T, 1>; ++i) {
        file.write(reinterpret_cast<char const*>(&num_bytes),
                   sizeof(HeaderType));
        for (auto const v : m_grid.vertices()) {
          auto data_begin = &prop[v](0, i);
          file.write(reinterpret_cast<char const*>(data_begin),
                     sizeof(tensor_value_type<T>) * tensor_dimension<T, 0>);
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  template <typename... Ts>
  auto write_vertex_property_data_array(auto const&                 name,
                                        vertex_property_type const& prop,
                                        std::ofstream&              file,
                                        std::size_t& offset) const {
    (
        [&] {
          if (prop.type() == typeid(Ts)) {
            if constexpr (tensor_rank<Ts> <= 1) {
              file << "        <DataArray"
                   << " Name=\"" << name << "\""
                   << " format=\"appended\""
                   << " offset=\"" << offset << "\""
                   << " type=\""
                   << tatooine::vtk::xml::to_string(
                          tatooine::vtk::xml::to_type<
                              tensor_value_type<Ts>>())
                   << "\" NumberOfComponents=\""
                   << tensor_num_components<Ts> << "\"/>\n";
              offset +=
                  m_grid.vertices().size() * sizeof(Ts) + sizeof(HeaderType);
            } else if constexpr (tensor_rank<Ts> == 2) {
              for (std::size_t i = 0; i < Ts::dimension(1); ++i) {
                file << "<DataArray"
                     << " Name=\"" << name << "_col_" << i << "\""
                     << " format=\"appended\""
                     << " offset=\"" << offset << "\""
                     << " type=\""
                     << vtk::xml::to_string(
                            vtk::xml::to_type<
                                tensor_value_type<Ts>>())
                     << "\" NumberOfComponents=\"" << Ts::dimension(0)
                     << "\"/>\n";
                offset += m_grid.vertices().size() *
                              sizeof(tensor_value_type<Ts>) *
                              tensor_dimension<Ts, 0> +
                          sizeof(HeaderType);
              }
            }
          }
        }(),
        ...);
  }
};
//==============================================================================
}  // namespace tatooine::detail::unstructured_simplicial_grid
//==============================================================================
#endif
