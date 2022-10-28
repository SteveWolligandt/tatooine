#ifndef TATOOINE_DETAIL_LINE_OPERATIONS_H
#define TATOOINE_DETAIL_LINE_OPERATIONS_H
//==============================================================================
#include <tatooine/line.h>
#include <tatooine/pow.h>
//==============================================================================
namespace tatooine::detail::line {
//==============================================================================
template <typename T, typename Writer, typename Names>
auto write_properties_to_vtk(Writer& writer, Names const& names,
                                       range_of_lines auto const& lines) -> void {
  std::vector<T> prop_collector;
  for (auto const& [name_to_search, type_to_search] : names) {
    prop_collector.clear();
    for (auto const& l : lines) {
      if (l.has_vertex_property(name_to_search)) {
        try {
          auto const& prop      = l.template vertex_property<T>(name_to_search);
          auto const& prop_data = prop.internal_container();
          std::copy(begin(prop_data), end(prop_data),
                    std::back_inserter(prop_collector));
        } catch (...) {
          for (std::size_t i = 0; i < l.vertices().size(); ++i) {
            prop_collector.push_back(0.0 / 0.0);
          }
        }
      }
    }
    writer.write_scalars(name_to_search, prop_collector);
  }
}
//==============================================================================
}  // namespace tatooine::detail::line
//==============================================================================
namespace tatooine {
//==============================================================================
template <range_of_lines Lines>
auto write_vtp(Lines const& lines, filesystem::path const& path) -> void {
  auto file = std::ofstream{path, std::ios::binary};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not write " + path.string()};
  }
  auto offset              = std::size_t{};
  using header_type        = std::uint32_t;
  using connectivity_int_t = std::int32_t;
  using offset_int_t       = connectivity_int_t;
  using real_type          = typename Lines::value_type::real_type;
  file << "<VTKFile"
       << " type=\"PolyData\""
       << " version=\"1.0\" "
          "byte_order=\"LittleEndian\""
       << " header_type=\""
       << vtk::xml::to_data_type<header_type>()
       << "\">\n";
  file << "  <PolyData>\n";
  for (auto const& l : lines) {
    file << "    <Piece"
         << " NumberOfPoints=\"" << l.vertices().size() << "\""
         << " NumberOfPolys=\"0\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\""
         << (l.vertices().size() - (l.is_closed() ? 0 : 1)) << "\""
         << " NumberOfStrips=\"0\""
         << ">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::to_string(
                vtk::xml::to_data_type<real_type>())
         << "\" NumberOfComponents=\"3\"/>\n";
    auto const num_bytes_points =
        header_type(sizeof(real_type) * 3 * l.vertices().size());
    offset += num_bytes_points + sizeof(header_type);
    file << "      </Points>\n";

    // Lines
    file << "      <Lines>\n";
    // Lines - connectivity
    file << "        <DataArray format=\"appended\" offset=\"" << offset
         << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_data_type<connectivity_int_t>())
         << "\" Name=\"connectivity\"/>\n";
    auto const num_bytes_connectivity =
        (l.vertices().size() - (l.is_closed() ? 0 : 1)) * 2 *
        sizeof(connectivity_int_t);
    offset += num_bytes_connectivity + sizeof(header_type);
    // Lines - offsets
    file << "        <DataArray format=\"appended\" offset=\"" << offset
         << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_data_type<offset_int_t>())
         << "\" Name=\"offsets\"/>\n";
    auto const num_bytes_offsets =
        sizeof(offset_int_t) * (l.vertices().size() - (l.is_closed() ? 0 : 1));
    offset += num_bytes_offsets + sizeof(header_type);
    file << "      </Lines>\n";
    file << "    </Piece>\n";
  }
  file << "  </PolyData>\n";
  file << "  <AppendedData encoding=\"raw\">\n    _";
  // Writing vertex data to appended data section
  for (auto const& l : lines) {
    auto arr_size = header_type{};
    arr_size      = header_type(sizeof(real_type) * 3 * l.vertices().size());
    file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
    auto zero = real_type(0);
    for (auto const v : l.vertices()) {
      if constexpr (Lines::value_type::num_dimensions() == 2) {
        file.write(reinterpret_cast<char const*>(l.at(v).data()),
                   sizeof(real_type) * 2);
        file.write(reinterpret_cast<char const*>(&zero), sizeof(real_type));
      } else if constexpr (Lines::value_type::num_dimensions() == 3) {
        file.write(reinterpret_cast<char const*>(l.at(v).data()),
                   sizeof(real_type) * 3);
      }
    }

    // Writing lines connectivity data to appended data section
    {
      auto connectivity_data = std::vector<connectivity_int_t>{};
      connectivity_data.reserve(
          (l.vertices().size() - (l.is_closed() ? 0 : 1)) * 2);
      for (std::size_t i = 0; i < l.vertices().size() - 1; ++i) {
        connectivity_data.push_back(static_cast<connectivity_int_t>(i));
        connectivity_data.push_back(static_cast<connectivity_int_t>(i + 1));
      }
      if (l.is_closed()) {
        connectivity_data.push_back(
            static_cast<connectivity_int_t>(l.vertices().size()) - 1);
        connectivity_data.push_back(0);
      }
      arr_size = static_cast<header_type>(connectivity_data.size() *
                                          sizeof(connectivity_int_t));
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 arr_size);
    }

    // Writing lines offsets to appended data section
    {
      auto offsets = std::vector<offset_int_t>(
          l.vertices().size() - (l.is_closed() ? 0 : 1), 2);
      for (std::size_t i = 1; i < size(offsets); ++i) {
        offsets[i] += offsets[i - 1];
      }
      arr_size = static_cast<header_type>(sizeof(offset_int_t)) *
                 static_cast<header_type>(l.vertices().size() -
                 static_cast<header_type>((l.is_closed() ? 0 : 1)));
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(offsets.data()), arr_size);
    }
  }
  file << "\n  </AppendedData>\n";
  file << "</VTKFile>";
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <range_of_lines Lines>
auto write_vtk(Lines const& lines, filesystem::path const& path,
               std::string const& title = "tatooine lines") -> void {
  auto writer = vtk::legacy_file_writer{path, vtk::dataset_type::polydata};
  if (writer.is_open()) {
    auto num_pts = std::size_t{};
    for (const auto& l : lines) {
      num_pts += l.vertices().size();
    }
    auto points =
        std::vector<std::array<typename Lines::value_type::real_type, 3>>{};
    auto line_seqs = std::vector<std::vector<std::size_t>>{};
    points.reserve(num_pts);
    line_seqs.reserve(lines.size());

    auto cur_first = std::size_t{};
    for (const auto& l : lines) {
      // add points
      for (const auto& v : l.vertices()) {
        auto const& p = l[v];
        if constexpr (Lines::value_type::num_dimensions() == 3) {
          points.push_back({p(0), p(1), p(2)});
        } else {
          points.push_back({p(0), p(1), 0});
        }
      }

      // add lines
      boost::iota(line_seqs.emplace_back(l.vertices().size()), cur_first);
      if (l.is_closed()) {
        line_seqs.back().push_back(cur_first);
      }
      cur_first += l.vertices().size();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_lines(line_seqs);
    writer.write_point_data(num_pts);

    auto names = std::set<std::pair<std::string, std::type_info const*>>{};
    // collect names
    for (const auto& l : lines) {
      for (auto const& [name, prop] : l.vertex_properties()) {
        names.insert(std::pair{name, &prop->type()});
      }
    }
    detail::line::write_properties_to_vtk<double>(writer, names,
                                                            lines);
    writer.close();
  }
}
//------------------------------------------------------------------------------
template <range_of_lines Lines>
auto write(Lines const& lines, filesystem::path const& path) -> void {
  using line_t   = std::ranges::range_value_t<Lines>;
  auto const ext = path.extension();
  if constexpr (line_t::num_dimensions() == 2 ||
                line_t::num_dimensions() == 3) {
    if (ext == ".vtk") {
      write_vtk(lines, path);
      return;
    } else if (ext == ".vtp") {
      write_vtp(lines, path);
      return;
    }
  }
  throw std::runtime_error("Could not write lines. Unknown file extension: \"" +
                           ext.string() + "\".");
}
//------------------------------------------------------------------------------
template <typename Lines, typename MaxDist>
auto merge(range_of_lines auto lines, MaxDist max_length) {
  using line_t = std::ranges::range_value_t<Lines>;
  auto merged = std::list<line_t> {};
  merged.emplace_back(std::move(lines.back()));
  lines.pop_back();

  while (!lines.empty()) {
    auto min_d   = std::numeric_limits<typename line_t::real_type>::max();
    auto best_it = std::end(lines);
    auto merged_take_front = false;
    auto it_take_front     = false;
    for (auto it = std::begin(lines); it != std::end(lines); ++it) {
      if (const auto d =
              distance(merged.back().front_vertex(), it->front_vertex());
          d < min_d && d < max_length) {
        min_d             = d;
        best_it           = it;
        merged_take_front = true;
        it_take_front     = true;
      }
      if (const auto d =
              distance(merged.back().back_vertex(), it->front_vertex());
          d < min_d && d < max_length) {
        min_d             = d;
        best_it           = it;
        merged_take_front = false;
        it_take_front     = true;
      }
      if (const auto d =
              distance(merged.back().front_vertex(), it->back_vertex());
          d < min_d && d < max_length) {
        min_d             = d;
        best_it           = it;
        merged_take_front = true;
        it_take_front     = false;
      }
      if (const auto d =
              distance(merged.back().back_vertex(), it->back_vertex());
          d < min_d && d < max_length) {
        min_d             = d;
        best_it           = it;
        merged_take_front = false;
        it_take_front     = false;
      }
    }

    if (best_it != end(lines)) {
      if (merged_take_front) {
        if (it_take_front) {
          for (const auto& v : best_it->vertices()) {
            merged.back().push_front(v);
          }
        } else {
          for (const auto& v :
               best_it->vertices() | boost::adaptors::reversed) {
            merged.back().push_front(v);
          }
        }
      } else {
        if (it_take_front) {
          for (const auto& v : best_it->vertices()) {
            merged.back().push_back(v);
          }
        } else {
          for (const auto& v :
               best_it->vertices() | boost::adaptors::reversed) {
            merged.back().push_back(v);
          }
        }
      }
      lines.erase(best_it);
    } else {
      merged.emplace_back(std::move(lines.back()));
      lines.pop_back();
    }
  }

  return merged;
}
//------------------------------------------------------------------------------
auto filter_length(range_of_lines auto const& lines,
                   arithmetic auto const      max_length) {
  auto filtered =
      std::vector<typename std::decay_t<decltype(lines)>::value_type>{};
  for (auto const& l : lines) {
    if (l.arc_length() > max_length) {
      filtered.push_back(l);
    }
  }
  return filtered;
}
//------------------------------------------------------------------------------
template <typename Real>
auto intersections(std::vector<line<Real, 2>> const& lines0,
                   std::vector<line<Real, 2>> const& lines1) {
  static auto const eps = Real(1e-6);
  auto              xs  = std::vector<vec<Real, 2>>{};
  for (auto const& l0 : lines0) {
    for (auto const& l1 : lines1) {
      for (std::size_t i = 0; i < l0.vertices().size() - 1; ++i) {
        for (std::size_t j = 0; j < l1.vertices().size() - 1; ++j) {
          auto const& p0  = l0.vertex_at(i);
          auto const& p1  = l0.vertex_at(i + 1);
          auto const& p2  = l1.vertex_at(j);
          auto const& p3  = l1.vertex_at(j + 1);
          auto const  d01 = p0 - p1;
          auto const  d23 = p2 - p3;
          auto const  d02 = p0 - p2;

          auto const denom = d01.x() * d23.y() - d01.y() * d23.x();
          if (std::abs(denom) < eps) {
            continue;
          }
          auto const inv_denom = 1 / denom;

          auto const nom_t = d02.x() * d23.y() - d02.y() * d23.x();
          auto const t     = nom_t * inv_denom;
          if (0 > t || t > 1) {
            continue;
          }
          auto const nom_u = -(d01.x() * d02.y() - d01.y() * d02.x());
          auto const u     = nom_u * inv_denom;
          if (0 > u || u > 1) {
            continue;
          }
          xs.push_back(p2 - u * d23);
        }
      }
    }
  }
  return xs;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
