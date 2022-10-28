#ifndef TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_EDGE_VTP_WRITER_H
#define TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_EDGE_VTP_WRITER_H
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
          integral          OffsetInt       = std::int64_t>
requires(Grid::num_dimensions() == 2 || Grid::num_dimensions() == 3)
struct edge_vtp_writer {
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
    auto const num_bytes_points = HeaderType(sizeof(typename Grid::real_type) *
                                             3 * m_grid.vertices().size());
    auto const num_bytes_connectivity = m_grid.simplices().size() *
                                        m_grid.num_vertices_per_simplex() *
                                        sizeof(ConnectivityInt);
    auto const num_bytes_offsets =
        sizeof(OffsetInt) * m_grid.simplices().size();
    file << "<VTKFile"
         << " type=\"PolyData\""
         << " version=\"1.0\""
         << " byte_order=\"LittleEndian\""
         << " header_type=\""
         << vtk::xml::to_string(
                vtk::xml::to_data_type<HeaderType>())
         << "\">\n"
         << "<PolyData>\n"
         << "<Piece"
         << " NumberOfPoints=\"" << m_grid.vertices().size() << "\""
         << " NumberOfPolys=\"0\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\"" << m_grid.simplices().size() << "\""
         << " NumberOfStrips=\"0\""
         << ">\n"
         // Points
         << "<Points>"
         << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::to_string(
                vtk::xml::to_data_type<typename Grid::real_type>())
         << "\" NumberOfComponents=\"3\"/>"
         << "</Points>\n";
    offset += num_bytes_points + sizeof(HeaderType);
    // Lines
    file << "<Lines>\n"
         // Lines - connectivity
         << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_data_type<ConnectivityInt>())
         << "\" Name=\"connectivity\"/>\n";
    offset += num_bytes_connectivity + sizeof(HeaderType);
    // Lines - offsets
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_data_type<OffsetInt>())
         << "\" Name=\"offsets\"/>\n";
    offset += num_bytes_offsets + sizeof(HeaderType);
    file << "</Lines>\n"
         << "</Piece>\n"
         << "</PolyData>\n"
         << "<AppendedData encoding=\"raw\">\n_";
    // Writing vertex data to appended data section

    using namespace std::ranges;
    {
      file.write(reinterpret_cast<char const*>(&num_bytes_points),
                 sizeof(HeaderType));
      if constexpr (Grid::num_dimensions() == 2) {
        auto point_data = std::vector<vec<typename Grid::real_type, 3>>(
            m_grid.vertices().size());
        auto position = [this](auto const v) -> auto& { return m_grid.at(v); };
        constexpr auto to_3d = [](auto const& p) {
          return vec{p.x(), p.y(), typename Grid::real_type(0)};
        };
        copy(m_grid.vertices() | views::transform(position) |
                 views::transform(to_3d),
             begin(point_data));
        file.write(reinterpret_cast<char const*>(point_data.data()),
                   num_bytes_points);
      } else if constexpr (Grid::num_dimensions() == 3) {
        file.write(reinterpret_cast<char const*>(m_grid.vertices().data()),
                   num_bytes_points);
      }
    }

    // Writing lines connectivity data to appended data section
    {
      auto connectivity_data = std::vector<ConnectivityInt>(
          m_grid.simplices().size() * m_grid.num_vertices_per_simplex());
      auto index = [](auto const x) -> ConnectivityInt { return x.index(); };
      copy(m_grid.simplices().data_container() | views::transform(index),
           begin(connectivity_data));
      file.write(reinterpret_cast<char const*>(&num_bytes_connectivity),
                 sizeof(HeaderType));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 num_bytes_connectivity);
    }

    // Writing lines offsets to appended data section
    {
      auto offsets = std::vector<OffsetInt>(m_grid.simplices().size(),
                                            m_grid.num_vertices_per_simplex());
      for (std::size_t i = 1; i < size(offsets); ++i) {
        offsets[i] += offsets[i - 1];
      };
      file.write(reinterpret_cast<char const*>(&num_bytes_offsets),
                 sizeof(HeaderType));
      file.write(reinterpret_cast<char const*>(offsets.data()),
                 num_bytes_offsets);
    }
    file << "\n</AppendedData>\n"
         << "</VTKFile>";
  }
};
//==============================================================================
}  // namespace tatooine::detail::unstructured_simplicial_grid
//==============================================================================
#endif
