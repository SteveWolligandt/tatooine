#ifndef TATOOINE_DETAIL_LINE_VTP_WRITER_H
#define TATOOINE_DETAIL_LINE_VTP_WRITER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/filesystem.h>
#include <tatooine/line.h>
#include <tatooine/linspace.h>
#include <tatooine/vtk/xml.h>

#include <array>
#include <vector>
//==============================================================================
namespace tatooine::detail::line {
//==============================================================================
template <typename Line, unsigned_integral HeaderType, integral ConnectivityInt,
          integral OffsetInt>
requires(Line::num_dimensions() == 2 || Line::num_dimensions() == 3)
struct vtp_writer {
  static auto constexpr num_dimensions() { return Line::num_dimensions(); }
  using vertex_property_type = typename Line::vertex_property_type;
  template <typename T>
  using typed_vertex_property_type =
      typename Line::template typed_vertex_property_type<T>;
  //----------------------------------------------------------------------------
  Line const& m_line;
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
  auto write_vtk_file(std::ofstream& file, std::size_t& offset) const {
    file << "<VTKFile"
         << " type=\"PolyData\""
         << " version=\"1.0\""
         << " byte_order=\"LittleEndian\""
         << " header_type=\"" << vtk::xml::to_data_type<HeaderType>()
         << "\">\n";
    write_polydata(file, offset);
    write_appended_data(file);
    file << "</VTKFile>";
  }
  //----------------------------------------------------------------------------
  auto write_polydata(std::ofstream& file, std::size_t& offset) const {
    file << "  <PolyData>\n";
    write_piece(file, offset);
    file << "  </PolyData>\n";
  }
  //----------------------------------------------------------------------------
  auto write_piece(std::ofstream& file, std::size_t& offset) const {
    file << "    <Piece"
         << " NumberOfPoints=\"" << m_line.num_vertices() << "\""
         << " NumberOfLines=\"" << m_line.num_line_segments() << "\""
         << ">\n";
    write_points(file, offset);
    write_lines(file, offset);
    write_point_data(file, offset);
    file << "    </Piece>\n";
  }
  //----------------------------------------------------------------------------
  auto write_points(std::ofstream& file, std::size_t& offset) const {
    auto const num_bytes_points = HeaderType(sizeof(typename Line::real_type) *
                                             3 * m_line.num_vertices());
    file << "      <Points>\n"
         << "        <DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::to_data_type<typename Line::real_type>()
         << "\" NumberOfComponents=\"3\"/>\n"
         << "      </Points>\n";
    offset += num_bytes_points + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  auto write_lines(std::ofstream& file, std::size_t& offset) const {
    file << "      <Lines>\n";
    write_lines_connectivity(file, offset);
    write_lines_offsets(file, offset);
    file << "      </Lines>\n";
  }
  //----------------------------------------------------------------------------
  auto write_lines_connectivity(std::ofstream& file,
                                std::size_t&   offset) const {
    auto const num_bytes =
        m_line.num_line_segments() * 2 * sizeof(ConnectivityInt);
    file << "        <DataArray format=\"appended\" offset=\"" << offset
         << "\" type=\""
         << vtk::xml::to_data_type<ConnectivityInt>()
         << "\" Name=\"connectivity\"/>\n";
    offset += num_bytes + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  auto write_lines_offsets(std::ofstream& file, std::size_t& offset) const {
    auto const num_bytes = sizeof(OffsetInt) * m_line.num_line_segments();
    file << "        <DataArray format=\"appended\" offset=\"" << offset
         << "\" type=\""
         << vtk::xml::to_data_type<OffsetInt>()
         << "\" Name=\"offsets\"/>\n";
    offset += num_bytes + sizeof(HeaderType);
  }
  //----------------------------------------------------------------------------
  auto write_point_data(std::ofstream& file, std::size_t& offset) const {
    file << "      <PointData>\n";
    for (auto const& [name, prop] : m_line.vertex_properties()) {
      write_vertex_property_data_array<HeaderType, float, double, vec2f, vec2d,
                                       vec3f, vec3d, vec4f, vec4d, mat2f, mat2d,
                                       mat3f, mat3d, mat4f, mat4d>(
          name, *prop, file, offset);
    }
    file << "      </PointData>\n";
  }
  //----------------------------------------------------------------------------
  auto write_appended_data(std::ofstream& file) const {
    file << "  <AppendedData encoding=\"raw\">\n_";
    write_vertex_positions_to_appended_data(file);
    write_line_connectivity_to_appended_data(file);
    write_line_offsets_to_appended_data(file);
    for (auto const& [name, prop] : m_line.vertex_properties()) {
      write_vertex_property_appended_data<
          HeaderType, float, vec2f, vec3f, vec4f, mat2f, mat3f, mat4f, double,
          vec2d, vec3d, vec4d, mat2d, mat3d, mat4d>(*prop, file);
    }
    file << "\n  </AppendedData>\n";
  }
  //----------------------------------------------------------------------------
  auto write_vertex_positions_to_appended_data(std::ofstream& file) const {
    auto arr_size = HeaderType(sizeof(typename Line::real_type) * 3 *
                               m_line.num_vertices());
    file.write(reinterpret_cast<char const*>(&arr_size), sizeof(HeaderType));
    auto zero = (typename Line::real_type)(0);
    for (auto const v : m_line.vertices()) {
      if constexpr (num_dimensions() == 2) {
        file.write(reinterpret_cast<char const*>(m_line[v].data()),
                   sizeof(typename Line::real_type) * 2);
        file.write(reinterpret_cast<char const*>(&zero),
                   sizeof(typename Line::real_type));
      } else if constexpr (num_dimensions() == 3) {
        file.write(reinterpret_cast<char const*>(m_line[v].data()),
                   sizeof(typename Line::real_type) * 3);
      }
    }
  }
  //----------------------------------------------------------------------------
  // Writing polys connectivity data to appended data section
  auto write_line_connectivity_to_appended_data(std::ofstream& file) const {
    auto connectivity_data = std::vector<ConnectivityInt>{};
    connectivity_data.reserve(m_line.num_line_segments() * 2);
    for (ConnectivityInt i = 0;
         i < static_cast<ConnectivityInt>(m_line.num_vertices()) - 1; ++i) {
      connectivity_data.push_back(i);
      connectivity_data.push_back(i + 1);
    }
    if (m_line.is_closed()) {
      connectivity_data.push_back(
          static_cast<ConnectivityInt>(m_line.num_vertices() - 1));
      connectivity_data.push_back(0);
    }
    auto const arr_size = static_cast<HeaderType>(connectivity_data.size() *
                                                  sizeof(ConnectivityInt));
    file.write(reinterpret_cast<char const*>(&arr_size), sizeof(HeaderType));
    file.write(reinterpret_cast<char const*>(connectivity_data.data()),
               arr_size);

    // using namespace std::ranges;
    // auto const num_bytes_connectivity =
    //     m_line.num_line_segments() * 2 * sizeof(ConnectivityInt);
    // auto connectivity_data =
    //     std::vector<ConnectivityInt>(m_line.num_line_segments() * 2);
    // auto index = [](auto const x) -> ConnectivityInt { return x.index(); };
    // copy(m_line.simplices().data_container() | views::transform(index),
    //      begin(connectivity_data));
    // file.write(reinterpret_cast<char const*>(&num_bytes_connectivity),
    //            sizeof(HeaderType));
    // file.write(reinterpret_cast<char const*>(connectivity_data.data()),
    //            num_bytes_connectivity);
  }
  //----------------------------------------------------------------------------
  auto write_line_offsets_to_appended_data(std::ofstream& file) const {
    auto offsets = std::vector<OffsetInt>(m_line.num_line_segments(), 2);
    for (std::size_t i = 1; i < size(offsets); ++i) {
      offsets[i] += offsets[i - 1];
    }
    auto const arr_size =
        static_cast<HeaderType>(sizeof(OffsetInt) * m_line.num_line_segments());
    file.write(reinterpret_cast<char const*>(&arr_size), sizeof(HeaderType));
    file.write(reinterpret_cast<char const*>(offsets.data()), arr_size);

    // using namespace std::ranges;
    // auto const num_bytes_offsets =
    //     sizeof(OffsetInt) * m_line.simplices().size();
    // auto offsets = std::vector<OffsetInt>(m_line.simplices().size(),
    //                                       m_line.num_vertices_per_simplex());
    // for (std::size_t i = 1; i < size(offsets); ++i) {
    //   offsets[i] += offsets[i - 1];
    // };
    // file.write(reinterpret_cast<char const*>(&num_bytes_offsets),
    //            sizeof(HeaderType));
    // file.write(reinterpret_cast<char const*>(offsets.data()),
    //            num_bytes_offsets);
  }
  //----------------------------------------------------------------------------
  template <typename... Ts>
  auto write_vertex_property_data_array(auto const&                 name,
                                        vertex_property_type const& prop,
                                        std::ofstream&              file,
                                        std::size_t& offset) const {
    invoke([&] {
      if (prop.type() == typeid(Ts)) {
        if constexpr (tensor_rank<Ts> <= 1) {
          file << "        <DataArray"
               << " Name=\"" << name << "\""
               << " format=\"appended\""
               << " offset=\"" << offset << "\""
               << " type=\""
               << tatooine::vtk::xml::to_data_type<tensor_value_type<Ts>>()
               << "\" NumberOfComponents=\""
               << tensor_num_components<Ts> << "\"/>\n";
          offset += m_line.num_vertices() * sizeof(Ts) + sizeof(HeaderType);
        } else if constexpr (tensor_rank<Ts> == 2) {
          for (std::size_t i = 0; i < Ts::dimension(1); ++i) {
            file << "        <DataArray"
                 << " Name=\"" << name << "_col_" << i << "\""
                 << " format=\"appended\""
                 << " offset=\"" << offset << "\""
                 << " type=\""
                 << vtk::xml::to_data_type<tensor_value_type<Ts>>()
                 << "\" NumberOfComponents=\"" << Ts::dimension(0) << "\"/>\n";
            offset += m_line.num_vertices() * sizeof(tensor_value_type<Ts>) *
                          tensor_dimension<Ts, 0> +
                      sizeof(HeaderType);
          }
        }
      }
    }...);
  }
  //----------------------------------------------------------------------------
  template <typename... Ts>
  auto write_vertex_property_appended_data(vertex_property_type const& prop,
                                           std::ofstream& file) const {
    invoke([&] {
      if (prop.type() == typeid(Ts)) {
        write_vertex_property_appended_data(prop.template cast_to_typed<Ts>(),
                                            file);
      }
    }...);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto write_vertex_property_appended_data(
      typed_vertex_property_type<T> const& prop, std::ofstream& file) const {
    if constexpr (tensor_rank<T> <= 1) {
      auto data = std::vector<T>{};
      for (auto const v : m_line.vertices()) {
        data.push_back(prop[v]);
      }
      auto const num_bytes =
          HeaderType(sizeof(tensor_value_type<T>) * tensor_num_components<T> *
                     m_line.num_vertices());
      file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(HeaderType));
      file.write(reinterpret_cast<char const*>(data.data()), num_bytes);
    } else if constexpr (tensor_rank<T> == 2) {
      auto const num_bytes =
          HeaderType(sizeof(tensor_value_type<T>) * tensor_num_components<T> *
                     m_line.num_vertices() / tensor_dimension<T, 0>);
      for (std::size_t i = 0; i < tensor_dimension<T, 1>; ++i) {
        file.write(reinterpret_cast<char const*>(&num_bytes),
                   sizeof(HeaderType));
        for (auto const v : m_line.vertices()) {
          auto data_begin = &prop[v](0, i);
          file.write(reinterpret_cast<char const*>(data_begin),
                     sizeof(tensor_value_type<T>) * tensor_dimension<T, 0>);
        }
      }
    }
  }
};
//==============================================================================
}  // namespace tatooine::detail::line
//==============================================================================
#endif
