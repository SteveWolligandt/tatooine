#ifndef TATOOINE_GEOMETRY_DETAIL_LINE_MERGE_H
#define TATOOINE_GEOMETRY_DETAIL_LINE_MERGE_H
//==============================================================================
#include <tatooine/line.h>

#include <vector>
//==============================================================================
namespace tatooine::detail::line {
//==============================================================================
/// \brief Merges two sets of lines
/// Lines from parameter lines1 will be inserted into lines0 or possibly merged
/// intto other lines of lines0.
template <typename Real, std::size_t NumDimensions>
auto merge(std::vector<tatooine::line<Real, NumDimensions>> &lines0,
           std::vector<tatooine::line<Real, NumDimensions>> &lines1) -> void {
  auto const eps = Real(1e-13);
  // move line1 pairs to line0 pairs
  auto const size_before = size(lines0);
  lines0.resize(size(lines0) + size(lines1));
  std::ranges::move(lines1, next(begin(lines0), size_before));
  lines1.clear();
  lines1.shrink_to_fit();

  // merge line0 side
  for (auto line0 = begin(lines0); line0 != end(lines0); ++line0) {
    for (auto line1 = begin(lines0); line1 != end(lines0); ++line1) {
      if (line0 != line1 && !line0->empty() && !line1->empty()) {
        // [line0front, ..., LINE0BACK] -> [LINE1FRONT, ..., line1back]
        if (approx_equal(line0->back_vertex(), line1->front_vertex(), eps)) {
          for (std::size_t i = line0->vertices().size() - 2; i > 0; --i) {
            line1->push_front(line0->vertex_at(i));
          }
          line0->clear();

          // [line1front, ..., LINE1BACK] -> [LINE0FRONT, ..., line0back]
        } else if (approx_equal(line1->back_vertex(), line0->front_vertex(),
                                eps)) {
          for (std::size_t i = 1; i < line0->vertices().size(); ++i) {
            line1->push_back(line0->vertex_at(i));
          }
          line0->clear();

          // [LINE1FRONT, ..., line1back] -> [LINE0FRONT, ..., line0back]
        } else if (approx_equal(line1->front_vertex(), line0->front_vertex(),
                                eps)) {
          // -> [line1back, ..., LINE1FRONT] -> [LINE0FRONT, ..., line0back]
          for (std::size_t i = line0->vertices().size() - 2; i > 0; --i) {
            line1->push_back(line0->vertex_at(i));
          }
          line0->clear();

          // [line0front, ..., LINE0BACK] -> [line1front,..., LINE1BACK]
        } else if (approx_equal(line0->back_vertex(), line1->back_vertex(),
                                eps)) {
          // -> [line1front, ..., LINE1BACK] -> [LINE0BACK, ..., line0front]
          for (std::size_t i = line0->vertices().size() - 2; i > 0; --i) {
            line1->push_back(line0->vertex_at(i));
          }
          line0->clear();
        }
      }
    }
  }

  const auto [first, last] =
      std::ranges::remove_if(lines0, [](auto const &l) { return l.empty(); });
  lines0.erase(first, last);
}
//==============================================================================
} // namespace tatooine::detail::line
//==============================================================================
namespace tatooine {
//==============================================================================
/// \brief Merges a set of lines and combines lines with equal vertex endings.
template <range_of_lines Lines> auto merge(Lines const &unmerged_lines) {
  using line_t = std::ranges::range_value_t<Lines>;
  auto merged_lines = std::vector<std::vector<line_t>>(unmerged_lines.size());

  auto unmerged_it = begin(unmerged_lines);
  for (auto &merged_line : merged_lines) {
    merged_line.push_back({*unmerged_it});
    ++unmerged_it;
  }

  auto num_merge_steps =
      static_cast<std::size_t>(ceil(log2(unmerged_lines.size())));

  for (std::size_t i = 0; i < num_merge_steps; i++) {
    std::size_t offset = tatooine::pow(std::size_t(2), i);

#pragma omp parallel for
    for (std::size_t j = 0; j < unmerged_lines.size(); j += offset * 2) {
      auto left = j;
      auto right = j + offset;
      if (right < unmerged_lines.size()) {
        detail::line::merge(merged_lines[left], merged_lines[right]);
      }
    }
  }
  return merged_lines.front();
}
//------------------------------------------------------------------------------
// template <range_of_lines Lines>
// auto merge(Lines const& unmerged_lines) {
//  using line_t      = std::ranges::range_value_t<Lines>;
//  auto merged_lines = std::vector<line_t>{};
//  if (!unmerged_lines.empty()) {
//    auto line_strips = line_segments_to_line_strips(unmerged_lines);
//
//    for (const auto& line_strip : line_strips) {
//      merged_lines.emplace_back();
//      for (std::size_t i = 0; i < line_strip.vertices().size() - 1; i++) {
//        merged_lines.back().push_back(line_strip.vertex_at(i));
//      }
//      if (&line_strip.front_vertex() == &line_strip.back_vertex()) {
//        merged_lines.back().set_closed(true);
//      } else {
//        merged_lines.back().push_back(line_strip.back_vertex());
//      }
//    }
//  }
//  return merged_lines;
//}
//==============================================================================
} // namespace tatooine
//==============================================================================
#endif
