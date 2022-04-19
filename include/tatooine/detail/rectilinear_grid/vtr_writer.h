#ifndef TATOOINE_DETAIL_RECTILINEAR_GRID_VTR_WRITER_H
#define TATOOINE_DETAIL_RECTILINEAR_GRID_VTR_WRITER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/filesystem.h>
#include <tatooine/linspace.h>
#include <tatooine/vtk/xml.h>

#include <array>
#include <vector>
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
//==============================================================================
template <typename Grid, unsigned_integral HeaderType>
requires(Grid::num_dimensions() == 2 ||
         Grid::num_dimensions() == 3) struct vtr_writer {
  static auto constexpr num_dimensions() { return Grid::num_dimensions(); }
  using vertex_property_type = typename Grid::vertex_property_type;
  template <typename T, bool H>
  using typed_vertex_property_interface_type =
      typename Grid::template typed_vertex_property_interface_type<T, H>;
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
         << " type=\"RectilinearGrid\""
         << " version=\"1.0\""
         << " byte_order=\"LittleEndian\""
         << " header_type=\""
         << tatooine::vtk::xml::data_array::to_string(
                tatooine::vtk::xml::data_array::to_type<HeaderType>())
         << "\">\n";
    write_rectilinear_grid(file, offset);
    write_appended_data(file);
    file << "</VTKFile>";
  }
  //----------------------------------------------------------------------------
  auto write_rectilinear_grid(std::ofstream& file, auto& offset) const {
    file << "  <RectilinearGrid WholeExtent=\"" << 0 << " "
         << m_grid.template size<0>() - 1 << " " << 0 << " "
         << m_grid.template size<1>() - 1;
    if constexpr (num_dimensions() >= 3) {
      file << " " << 0 << " " << m_grid.template size<2>() - 1;
    } else {
      file << " " << 0 << " " << 0;
    }
    file << "\">\n";
    write_piece(file, offset);
    file << "  </RectilinearGrid>\n";
  }
  //----------------------------------------------------------------------------
  auto write_piece(std::ofstream& file, auto& offset) const {
    file << "    <Piece Extent=\"" << 0 << " " << m_grid.template size<0>() - 1
         << " " << 0 << " " << m_grid.template size<1>() - 1 << " ";
    if constexpr (num_dimensions() >= 3) {
      file << 0 << " " << m_grid.template size<2>() - 1;
    } else {
      file << 0 << " " << 0;
    }
    file << "\">\n";
    {
      write_point_data(file, offset);
      write_cell_data(file, offset);
      write_coordinates(file, offset);
    }
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
  auto write_cell_data(std::ofstream& file, std::size_t& offset) const {
    file << "      <CellData>\n";
    file << "      </CellData>\n";
  }
  //----------------------------------------------------------------------------
  auto write_coordinates(std::ofstream& file, std::size_t& offset) const {
    file << "      <Coordinates>\n";
    write_dimension_data_array(m_grid.template dimension<0>(), "x_coordinates",
                               file, offset);
    write_dimension_data_array(m_grid.template dimension<1>(), "y_coordinates",
                               file, offset);
    if constexpr (num_dimensions() >= 3) {
      write_dimension_data_array(m_grid.template dimension<2>(),
                                 "z_coordinates", file, offset);
    } else {
      write_dimension_data_array(std::vector<typename Grid::real_type>{0},
                                 "z_coordinates", file, offset);
    }
    file << "      </Coordinates>\n";
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto write_dimension_data_array(std::vector<T> const& dim,
                                  std::string const& name, std::ofstream& file,
                                  std::size_t& offset) const {
    file << "        <DataArray"
         << " type=\""
         << tatooine::vtk::xml::data_array::to_string(
                tatooine::vtk::xml::data_array::to_type<T>())
         << "\""
         << " Name=\"" << name << "\""
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << "/>\n";
    offset += dim.size() * sizeof(T) + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  template <typename T, std::size_t N>
  auto write_dimension_data_array(std::array<T, N> const& dim,
                                  std::string const& name, std::ofstream& file,
                                  std::size_t& offset) const {
    file << "        <DataArray"
         << " type=\""
         << tatooine::vtk::xml::data_array::to_string(
                tatooine::vtk::xml::data_array::to_type<T>())
         << "\""
         << " Name=\"" << name << "\""
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << "/>\n";
    offset += N * sizeof(T) + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto write_dimension_data_array(tatooine::linspace<T> const& dim,
                                  std::string const& name, std::ofstream& file,
                                  std::size_t& offset) const {
    file << "        <DataArray"
         << " type=\""
         << tatooine::vtk::xml::data_array::to_string(
                tatooine::vtk::xml::data_array::to_type<T>())
         << "\""
         << " Name=\"" << name << "\""
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << "/>\n";
    file.flush();
    offset += dim.size() * sizeof(T) + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  auto write_appended_data(std::ofstream& file) const {
    file << "  <AppendedData encoding=\"raw\">\n   _";
    write_appended_data_point_data(file);
    write_appended_data_cell_data(file);
    write_appended_data_coordinates(file);
    file << "\n  </AppendedData>\n";
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
  auto write_appended_data_cell_data(std::ofstream& file) const {}
  //----------------------------------------------------------------------------
  auto write_appended_data_coordinates(std::ofstream& file) const {
    write_dimension_appended_data(m_grid.template dimension<0>(), file);
    write_dimension_appended_data(m_grid.template dimension<1>(), file);
    if constexpr (num_dimensions() >= 3) {
      write_dimension_appended_data(m_grid.template dimension<2>(), file);
    } else {
      write_dimension_appended_data(std::vector{typename Grid::real_type(0)},
                                    file);
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto write_dimension_appended_data(std::vector<T> const& dim,
                                     std::ofstream&        file) const {
    auto const num_bytes = HeaderType(sizeof(T) * dim.size());
    file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(HeaderType));
    file.write(reinterpret_cast<char const*>(dim.data()), num_bytes);
  }
  //----------------------------------------------------------------------------
  template <typename T, std::size_t N>
  auto write_dimension_appended_data(std::array<T, N> const& dim,
                                     std::ofstream&          file) const {
    auto const num_bytes = HeaderType(sizeof(T) * N);
    file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(HeaderType));
    file.write(reinterpret_cast<char const*>(dim.data()), num_bytes);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto write_dimension_appended_data(tatooine::linspace<T> const& dim,
                                     std::ofstream&               file) const {
    auto data = std::vector<T>(dim.size());
    std::ranges::copy(dim, begin(data));
    auto const num_bytes = HeaderType(sizeof(T) * data.size());
    file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(HeaderType));
    file.flush();
    file.write(reinterpret_cast<char const*>(data.data()), num_bytes);
    file.flush();
  }
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
  template <typename T, bool H>
  auto write_vertex_property_appended_data(
      typed_vertex_property_interface_type<T, H> const& prop,
      std::ofstream&                                    file) const {
    if constexpr (tensor_rank<T> <= 1) {
      auto data = std::vector<T>{};
      m_grid.vertices().iterate_indices(
          [&](auto const... is) { data.push_back(prop(is...)); });
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
        m_grid.vertices().iterate_indices([&](auto const... is) {
          auto data_begin = &prop(is...)(0, i);
          file.write(reinterpret_cast<char const*>(data_begin),
                     sizeof(tensor_value_type<T>) * tensor_dimension<T, 0>);
        });
      }
    } else {
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
              file << "<DataArray"
                   << " Name=\"" << name << "\""
                   << " format=\"appended\""
                   << " offset=\"" << offset << "\""
                   << " type=\""
                   << tatooine::vtk::xml::data_array::to_string(
                          tatooine::vtk::xml::data_array::to_type<
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
                     << vtk::xml::data_array::to_string(
                            vtk::xml::data_array::to_type<
                                tensor_value_type<Ts>>())
                     << "\" NumberOfComponents=\"" << Ts::dimension(0)
                     << "\"/>\n";
                offset += m_grid.vertices().size() * sizeof(tensor_value_type<Ts>) *
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
}  // namespace tatooine::detail::rectilinear_grid
//==============================================================================
#endif
