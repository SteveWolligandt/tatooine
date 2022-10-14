#ifndef TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_TRIANGULAR_VTP_WRITER_H
#define TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_TRIANGULAR_VTP_WRITER_H
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
struct triangular_vtp_writer {
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
    if (!file.is_open()) {
      throw std::runtime_error{"Could not write " + path.string()};
    }
    auto       offset = std::size_t{};
    auto const num_bytes_points =
        HeaderType(sizeof(Real) * 3 * vertices().size());
    auto const num_bytes_connectivity = simplices().size() *
                                        num_vertices_per_simplex() *
                                        sizeof(ConnectivityInt);
    auto const num_bytes_offsets = sizeof(OffsetInt) * simplices().size();
    file << "<VTKFile"
         << " type=\"PolyData\""
         << " version=\"1.0\""
         << " byte_order=\"LittleEndian\""
         << " HeaderType=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<HeaderType>())
         << "\">\n"
         << "<PolyData>\n"
         << "<Piece"
         << " NumberOfPoints=\"" << vertices().size() << "\""
         << " NumberOfPolys=\"" << simplices().size() << "\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\"0\""
         << " NumberOfStrips=\"0\""
         << ">\n"
         // Points
         << "<Points>"
         << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<Real>())
         << "\" NumberOfComponents=\"3\"/>"
         << "</Points>\n";
    offset += num_bytes_points + sizeof(HeaderType);
    // Polys
    file << "<Polys>\n"
         // Polys - connectivity
         << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<ConnectivityInt>())
         << "\" Name=\"connectivity\"/>\n";
    offset += num_bytes_connectivity + sizeof(HeaderType);
    // Polys - offsets
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<OffsetInt>())
         << "\" Name=\"offsets\"/>\n";
    offset += num_bytes_offsets + sizeof(HeaderType);
    file << "</Polys>\n"
         << "</Piece>\n"
         << "</PolyData>\n"
         << "<AppendedData encoding=\"raw\">\n_";
    // Writing vertex data to appended data section

    using namespace std::ranges;
    {
      file.write(reinterpret_cast<char const*>(&num_bytes_points),
                 sizeof(HeaderType));
      if constexpr (NumDimensions == 2) {
        auto point_data      = std::vector<vec<Real, 3>>(vertices().size());
        auto position        = [this](auto const v) -> auto& { return at(v); };
        constexpr auto to_3d = [](auto const& p) {
          return Vec3<Real>{p.x(), p.y(), 0};
        };
        copy(vertices() | views::transform(position) | views::transform(to_3d),
             begin(point_data));
        file.write(reinterpret_cast<char const*>(point_data.data()),
                   num_bytes_points);
      } else if constexpr (NumDimensions == 3) {
        file.write(reinterpret_cast<char const*>(vertices().data()),
                   num_bytes_points);
      }
    }

    // Writing polys connectivity data to appended data section
    {
      auto connectivity_data = std::vector<ConnectivityInt>(
          simplices().size() * num_vertices_per_simplex());
      auto index = [](auto const x) -> ConnectivityInt { return x.index(); };
      copy(simplices().data_container() | views::transform(index),
           begin(connectivity_data));
      file.write(reinterpret_cast<char const*>(&num_bytes_connectivity),
                 sizeof(HeaderType));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 num_bytes_connectivity);
    }

    // Writing polys offsets to appended data section
    {
      auto offsets = std::vector<OffsetInt>(simplices().size(),
                                            num_vertices_per_simplex());
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
  //------------------};
//==============================================================================
}  // namespace tatooine::detail::unstructured_simplicial_grid
//==============================================================================
#endif
