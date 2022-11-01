#ifndef TATOOINE_DETAIL_LINE_VTL_WRITER_H
#define TATOOINE_DETAIL_LINE_VTL_WRITER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/filesystem.h>
#include <tatooine/vtk_legacy.h>
#include <boost/range/algorithm_ext/iota.hpp>

#include <array>
#include <vector>
//==============================================================================
namespace tatooine::detail::line {
//==============================================================================
template <typename Line, unsigned_integral HeaderType = std::uint64_t,
          integral ConnectivityInt = std::int64_t,
          integral OffsetInt       = std::int64_t>
requires(Line::num_dimensions() == 2 ||
         Line::num_dimensions() == 3) struct vtk_writer {
  static auto constexpr num_dimensions() { return Line::num_dimensions(); }
  using vertex_property_type = typename Line::vertex_property_type;
  template <typename T>
  using typed_vertex_property_type =
      typename Line::template typed_vertex_property_type<T>;
  //----------------------------------------------------------------------------
  Line const& m_line;
  //----------------------------------------------------------------------------
  auto write(filesystem::path const& path, std::string const& title) const {
    auto file = std::ofstream{path};
    if (!file.is_open()) {
      throw std::runtime_error{"Could open file " + path.string() +
                               " for writing."};
    }
    auto writer = vtk::legacy_file_writer{path, vtk::dataset_type::polydata};
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();

      // write points
      auto ps = std::vector<std::array<typename Line::real_type, 3>>{};
      ps.reserve(m_line.vertices().size());
      for (auto const& v : m_line.vertices()) {
        auto const& p = m_line[v];
        if constexpr (num_dimensions() == 3) {
          ps.push_back({p(0), p(1), p(2)});
        } else {
          ps.push_back({p(0), p(1), 0});
        }
      }
      writer.write_points(ps);

      // write lines
      auto line_seq = std::vector<std::vector<std::size_t>>(
          1, std::vector<std::size_t>(m_line.vertices().size()));
      boost::iota(line_seq.front(), 0);
      if (m_line.is_closed()) {
        line_seq.front().push_back(0);
      }
      writer.write_lines(line_seq);

      writer.write_point_data(m_line.vertices().size());

      // write properties
      for (auto& [name, prop] : m_line.vertex_properties()) {
        auto const& type = prop->type();
        if (type == typeid(float)) {
          write_prop<float>(writer, name, prop);
        } else if (type == typeid(vec<float, 2>)) {
          write_prop<vec<float, 2>>(writer, name, prop);
        } else if (type == typeid(vec<float, 3>)) {
          write_prop<vec<float, 3>>(writer, name, prop);
        } else if (type == typeid(vec<float, 4>)) {
          write_prop<vec<float, 4>>(writer, name, prop);

        } else if (type == typeid(double)) {
          write_prop<double>(writer, name, prop);
        } else if (type == typeid(vec<double, 2>)) {
          write_prop<vec<double, 2>>(writer, name, prop);
        } else if (type == typeid(vec<double, 3>)) {
          write_prop<vec<double, 3>>(writer, name, prop);
        } else if (type == typeid(vec<double, 4>)) {
          write_prop<vec<double, 4>>(writer, name, prop);
        }
      }
      writer.close();
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  static auto write_prop(
      vtk::legacy_file_writer& writer, std::string const& name,
      std::unique_ptr<vertex_property_type> const& prop) -> void {
    auto const& deque = dynamic_cast<typed_vertex_property_type<T>*>(prop.get())
                            ->internal_container();

    writer.write_scalars(name, std::vector<T>(begin(deque), end(deque)));
  }
};
//==============================================================================
}  // namespace tatooine::detail::line
//==============================================================================
#endif
