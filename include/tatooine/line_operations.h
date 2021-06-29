#ifndef TATOOINE_LINE_OPERATIONS_H
#define TATOOINE_LINE_OPERATIONS_H
//==============================================================================
#include <tatooine/line.h>
//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail {
//==============================================================================
template <typename T, typename Writer, typename Names, typename Lines>
auto write_line_container_properties_to_vtk(Writer& writer, Names const& names,
                                            Lines const& lines) -> void {
  std::vector<T> prop_collector;
  for (auto const& [name_to_search, type_to_search] : names) {
    prop_collector.clear();
    for (auto const& l : lines) {
      if (l.has_vertex_property(name_to_search)) {
        try {
          auto const& prop      = l.template vertex_property<T>(name_to_search);
          auto const& prop_data = prop.container();
          std::copy(begin(prop_data), end(prop_data),
                    std::back_inserter(prop_collector));
        } catch (...) {
          for (size_t i = 0; i < l.num_vertices(); ++i) {
            prop_collector.push_back(0.0 / 0.0);
          }
        }
      }
    }
    writer.write_scalars(name_to_search, prop_collector);
  }
}
//------------------------------------------------------------------------------
template <typename LineCont>
void write_line_container_to_vtk(LineCont const& lines, std::string const& path,
                                 std::string const& title) {
  vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
  if (writer.is_open()) {
    size_t num_pts = 0;
    for (const auto& l : lines) {
      num_pts += l.num_vertices();
    }
    std::vector<std::array<typename LineCont::value_type::real_t, 3>> points;
    std::vector<std::vector<size_t>>                                  line_seqs;
    points.reserve(num_pts);
    line_seqs.reserve(lines.size());

    size_t cur_first = 0;
    for (const auto& l : lines) {
      // add points
      for (const auto v : l.vertices()) {
        auto const& p = l[v];
        if constexpr (LineCont::value_type::num_dimensions() == 3) {
          points.push_back({p(0), p(1), p(2)});
        } else {
          points.push_back({p(0), p(1), 0});
        }
      }

      // add lines
      boost::iota(line_seqs.emplace_back(l.num_vertices()), cur_first);
      if (l.is_closed()) {
        line_seqs.back().push_back(cur_first);
      }
      cur_first += l.num_vertices();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_lines(line_seqs);
    writer.write_point_data(num_pts);

    std::set<std::pair<std::string, std::type_info const*>> names;
    // collect names
    for (const auto& l : lines) {
      for (auto const& [name, prop] : l.vertex_properties()) {
        names.insert(std::pair{name, &prop->type()});
      }
    }
    write_line_container_properties_to_vtk<double>(writer, names, lines);
    writer.close();
  }
}
//------------------------------------------------------------------------------
template <typename Lines, typename MaxDist /*, typename MinAngle*/>
auto merge_line_container(Lines   lines,
                          MaxDist max_dist /*, MinAngle min_angle*/) {
  using line_t = typename std::decay_t<Lines>::value_type;
  std::list<line_t> merged;
  merged.emplace_back(std::move(lines.back()));
  lines.pop_back();

  while (!lines.empty()) {
    auto min_d   = std::numeric_limits<typename line_t::real_t>::max();
    auto best_it = std::end(lines);
    bool merged_take_front = false;
    bool it_take_front     = false;
    for (auto it = std::begin(lines); it != std::end(lines); ++it) {
      if (const auto d =
              distance(merged.back().front_vertex(), it->front_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = true;
        it_take_front     = true;
      }
      if (const auto d =
              distance(merged.back().back_vertex(), it->front_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = false;
        it_take_front     = true;
      }
      if (const auto d =
              distance(merged.back().front_vertex(), it->back_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = true;
        it_take_front     = false;
      }
      if (const auto d =
              distance(merged.back().back_vertex(), it->back_vertex());
          d < min_d && d < max_dist) {
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
#ifdef __cpp_concepts
template <range Lines, arithmetic Real>
#else
template <typename Lines, typename Real,
          enable_if<is_range<Lines>, is_arithmetic<Real>> = true>
#endif
auto filter_length(Lines const& lines, Real length) {
  std::vector<typename std::decay_t<Lines>::value_type> filtered;
  for (auto const& l : lines) {
    if (l.arc_length() > length) {
      filtered.push_back(l);
    }
  }
  return filtered;
}
//==============================================================================
}  // namespace detail
//==============================================================================
template <typename Real, size_t N>
void write_vtk(const std::vector<line<Real, N>>& lines, const std::string& path,
               const std::string& title = "tatooine lines") {
  detail::write_line_container_to_vtk(lines, path, title);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
void write_vtk(const std::list<line<Real, N>>& lines, const std::string& path,
               const std::string& title = "tatooine lines") {
  detail::write_line_container_to_vtk(lines, path, title);
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename MaxDist>
auto merge(const std::vector<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::merge_line_container(lines, max_dist);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename MaxDist>
auto merge(const std::list<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::merge_line_container(lines, max_dist);
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename MaxDist>
auto filter_length(const std::vector<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::filter_length(lines, max_dist);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename MaxDist>
auto filter_length(const std::list<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::filter_length(lines, max_dist);
}
////------------------------------------------------------------------------------
//template <typename Real, size_t N,
//          template <typename> typename InterpolationKernel>
//void write_vtk(
//    const std::vector<parameterized_line<Real, N, InterpolationKernel>>& lines,
//    const std::string& path, const std::string& title = "tatooine lines") {
//  detail::write_line_container_to_vtk(lines, path, title);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Real, size_t N,
//          template <typename> typename InterpolationKernel>
//void write_vtk(
//    const std::list<parameterized_line<Real, N, InterpolationKernel>>& lines,
//    const std::string& path, const std::string& title = "tatooine lines") {
//  detail::write_line_container_to_vtk(lines, path, title);
//}
//==============================================================================
/// \brief      merge line strips
template <typename Real, size_t N>
auto merge(std::vector<line<Real, N>>& lines0,
           std::vector<line<Real, N>>& lines1) -> void {
  Real const eps = 1e-7;
  // move line1 pairs to line0 pairs
  size_t const size_before = size(lines0);
  lines0.resize(size(lines0) + size(lines1));
  std::move(begin(lines1), end(lines1), next(begin(lines0), size_before));
  lines1.clear();

  // merge line0 side
  for (auto line0 = begin(lines0); line0 != end(lines0); ++line0) {
    for (auto line1 = begin(lines0); line1 != end(lines0); ++line1) {
      if (line0 != line1 && !line0->empty() && !line1->empty()) {
        // [line0front, ..., LINE0BACK] -> [LINE1FRONT, ..., line1back]
        if (approx_equal(line0->back_vertex(), line1->front_vertex(), eps)) {
          for (size_t i = line0->num_vertices() - 2; i > 0; --i) {
            line1->push_front(line0->vertex_at(i));
          }
          line0->clear();

          // [line1front, ..., LINE1BACK] -> [LINE0FRONT, ..., line0back]
        } else if (approx_equal(line1->back_vertex(), line0->front_vertex(),
                                eps)) {
          for (size_t i = 1; i < line0->num_vertices(); ++i) {
            line1->push_back(line0->vertex_at(i));
          }
          line0->clear();

          // [LINE1FRONT, ..., line1back] -> [LINE0FRONT, ..., line0back]
        } else if (approx_equal(line1->front_vertex(), line0->front_vertex(),
                                eps)) {
          // -> [line1back, ..., LINE1FRONT] -> [LINE0FRONT, ..., line0back]
          for (size_t i = line0->num_vertices() - 2; i > 0; --i) {
            line1->push_back(line0->vertex_at(i));
          }
          line0->clear();

          // [line0front, ..., LINE0BACK] -> [line1front,..., LINE1BACK]
        } else if (approx_equal(line0->back_vertex(), line1->back_vertex(),
                                eps)) {
          // -> [line1front, ..., LINE1BACK] -> [LINE0BACK, ..., line0front]
          for (size_t i = line0->num_vertices() - 2; i > 0; --i) {
            line1->push_back(line0->vertex_at(i));
          }
          line0->clear();
        }
      }
    }
  }

  // move empty vectors of line0 side at end
  for (unsigned int i = 0; i < lines0.size(); i++) {
    for (unsigned int j = 0; j < i; j++) {
      if (lines0[j].empty() && !lines0[i].empty()) {
        lines0[j] = std::move(lines0[i]);
      }
    }
  }

  // remove empty vectors of line0 side
  for (int i = lines0.size() - 1; i >= 0; i--) {
    if (lines0[i].empty()) {
      lines0.pop_back();
    }
  }
}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
auto line_segments_to_line_strips(
    std::vector<line<Real, N>> const& unmerged_lines) {
  auto merged_lines =
      std::vector<std::vector<line<Real, N>>>(unmerged_lines.size());

  auto unmerged_it = begin(unmerged_lines);
  for (auto& merged_line : merged_lines) {
    merged_line.push_back({*unmerged_it});
    ++unmerged_it;
  }

  auto num_merge_steps =
      static_cast<size_t>(std::ceil(std::log2(unmerged_lines.size())));

  for (size_t i = 0; i < num_merge_steps; i++) {
    size_t offset = std::pow(2, i);

#pragma omp parallel for
    for (size_t j = 0; j < unmerged_lines.size(); j += offset * 2) {
      auto left  = j;
      auto right = j + offset;
      if (right < unmerged_lines.size()) {
        merge(merged_lines[left], merged_lines[right]);
      }
    }
  }
  return merged_lines.front();
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto merge(std::vector<line<Real, N>> const& lines) {
  std::vector<line<Real, N>> merged_lines;
  if (!lines.empty()) {
    auto line_strips = line_segments_to_line_strips(lines);

    for (const auto& line_strip : line_strips) {
      merged_lines.emplace_back();
      for (size_t i = 0; i < line_strip.num_vertices() - 1; i++) {
        merged_lines.back().push_back(line_strip.vertex_at(i));
      }
      if (&line_strip.front_vertex() == &line_strip.back_vertex()) {
        merged_lines.back().set_closed(true);
      } else {
        merged_lines.back().push_back(line_strip.back_vertex());
      }
    }
  }
  return merged_lines;
}
//------------------------------------------------------------------------------
template <typename Real>
auto intersections(const std::vector<line<Real, 2>>& lines0,
                   const std::vector<line<Real, 2>>& lines1) {
  static auto const         eps = 1e-6;
  std::vector<vec<Real, 2>> xs;
  for (auto const& l0 : lines0) {
    for (auto const& l1 : lines1) {
      for (size_t i = 0; i < l0.num_vertices() - 1; ++i) {
        for (size_t j = 0; j < l1.num_vertices() - 1; ++j) {
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
