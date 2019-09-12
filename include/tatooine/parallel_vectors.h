#ifndef TATOOINE_PARALLEL_VECTORS_H
#define TATOOINE_PARALLEL_VECTORS_H

#include <array>
#include <optional>
#include <tuple>
#include <vector>
#include "grid.h"
#include "line.h"
#include "type_traits.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V, typename W>
struct parallel_vectors {
  static_assert(V::tensor_t::num_dimensions() == 1);
  static_assert(W::tensor_t::num_dimensions() == 1);
  static_assert(V::num_dimensions() == 3);
  static_assert(W::num_dimensions() == 3);

  //============================================================================
  using real_t  = promote_t<typename V::real_t, typename W::real_t>;
  using this_t  = parallel_vectors<V, W>;
  using vec3    = tensor<real_t, 3>;
  using pos_t   = vec3;
  using pairs_t = std::vector<std::vector<vec3>>;

  //============================================================================
 private:
  const V&        m_v;
  const W&        m_w;
  grid<real_t, 3> m_grid;

  std::vector<std::vector<std::vector<std::vector<int>>>> m_upper_indices;
  std::vector<std::vector<std::vector<std::vector<int>>>> m_lower_indices;

  //============================================================================
 public:
  parallel_vectors(const V& v, const W& w, const grid<real_t, 3>& g)
      : m_v{v}, m_w{w}, m_grid{g} {}
  parallel_vectors(const V& v, const W& w, const linspace<real_t>& x,
                   const linspace<real_t>& y, const linspace<real_t>& z)
      : m_v{v}, m_w{w}, m_grid{x, y, z} {}

  //----------------------------------------------------------------------------
  /// \brief      merge line strip indices
  void merge(pairs_t& pairs_left, pairs_t& pairs_right) {
    const real_t eps = 1e-7;
    // move right pairs to left pairs
    pairs_left.insert(end(pairs_left), begin(pairs_right), end(pairs_right));
    pairs_right.clear();

    // merge left side
    for (auto left = begin(pairs_left); left != end(pairs_left); ++left) {
      for (auto right = begin(pairs_left); right != end(pairs_left); ++right) {
        if (left != right && !left->empty() && !right->empty()) {
          // [leftfront, ..., LEFTBACK] -> [RIGHTFRONT, ..., rightback]
          if (approx_equal(left->back(), right->front(), eps)) {
            right->insert(begin(*right), begin(*left), prev(end(*left)));
            left->clear();

            // [rightfront, ..., RIGHTBACK] -> [LEFTFRONT, ..., leftback]
          } else if (approx_equal(right->back(), left->front(), eps)) {
            right->insert(end(*right), next(begin(*left)), end(*left));
            left->clear();

            // [RIGHTFRONT, ..., rightback] -> [LEFTFRONT, ..., leftback]
          } else if (approx_equal(right->front(), left->front(), eps)) {
            boost::reverse(*left);
            // -> [leftback, ..., LEFTFRONT] -> [RIGHTFRONT, ..., rightback]
            right->insert(begin(*right), begin(*left), prev(end(*left)));
            left->clear();

            // [leftfront, ..., LEFTBACK] -> [rightfront,..., RIGHTBACK]
          } else if (approx_equal(left->back(), right->back(), eps)) {
            boost::reverse(*left);
            // -> [rightfront, ..., RIGHTBACK] -> [LEFTBACK, ..., leftfront]
            right->insert(end(*right), next(begin(*left)), end(*left));
            left->clear();
          }
        }
      }
    }

    // move empty vectors of left side at end
    for (unsigned int i = 0; i < pairs_left.size(); i++) {
      for (unsigned int j = 0; j < i; j++) {
        if (pairs_left[j].size() == 0 && pairs_left[i].size() > 0) {
          pairs_left[j] = std::move(pairs_left[i]);
        }
      }
    }

    // remove empty vectors of left side
    for (int i = pairs_left.size() - 1; i >= 0; i--) {
      if (pairs_left[i].size() == 0) { pairs_left.pop_back(); }
    }
  }

  //----------------------------------------------------------------------------
  auto line_segments_to_line_strips(
      const std::vector<std::pair<vec3, vec3>>& line_segments) {
    std::vector<std::vector<std::vector<vec3>>> merged_strips(
        line_segments.size());

    auto seg_it = begin(line_segments);
    for (auto& merged_strip : merged_strips) {
      merged_strip.push_back({seg_it->first, seg_it->second});
      ++seg_it;
    }

    auto num_merge_steps =
        static_cast<size_t>(std::ceil(std::log2(line_segments.size())));

    for (size_t i = 0; i < num_merge_steps; i++) {
      size_t offset = std::pow(2, i);

      for (size_t j = 0; j < line_segments.size(); j += offset * 2) {
        auto left  = j;
        auto right = j + offset;
        if (right < line_segments.size()) {
          merge(merged_strips[left], merged_strips[right]);
        }
      }
    }
    return merged_strips.front();
  }

  //----------------------------------------------------------------------------
  std::optional<vec3> pv_on_tri(
      const vec3& p0, const vec3& v0, const vec3& w0,
      const vec3& p1, const vec3& v1, const vec3& w1,
      const vec3& p2, const vec3& v2, const vec3& w2) {
    mat<real_t, 3, 3> v, w, m;
    v.col(0) = v0; v.col(1) = v1; v.col(2) = v2;
    w.col(0) = w0; w.col(1) = w1; w.col(2) = w2;

    if (std::abs(det(v)) > 0) {
      m = gesv(v, w);
    } else if (std::abs(det(w)) > 0) {
      m = gesv(w, v);
    } else {
      return {};
    }

    auto [eigvecs, eigvals] = eigenvectors(m);

    std::vector<vec3> barycentric_coords;
    for (int i = 0; i < 3; i++) {
      if ((std::abs(eigvecs(0, i).imag()) <= 0 &&
           std::abs(eigvecs(1, i).imag()) <= 0 &&
           std::abs(eigvecs(2, i).imag()) <= 0) &&
          ((eigvecs(0, i).real() <= 0 && eigvecs(1, i).real() <= 0 &&
            eigvecs(2, i).real() <= 0) ||
           (eigvecs(0, i).real() >= 0 && eigvecs(1, i).real() >= 0 &&
            eigvecs(2, i).real() >= 0))) {
        const auto bc = real(eigvecs.col(i)) / sum(real(eigvecs.col(i)));
        barycentric_coords.push_back(bc);
      }
    }

    if (barycentric_coords.empty()) {
      return {};

    } else if (barycentric_coords.size() == 1) {
      auto pos = barycentric_coords.front()(0) * p0 +
                 barycentric_coords.front()(1) * p1 +
                 barycentric_coords.front()(2) * p2;
      //if (std::abs(pos(0) * pos(0) + pos(1) * pos(1) - 9) < 1e-1) { return {}; }
      return pos;

    } else {
      // check if all found barycentric coordinates are the same
      //const real_t      eps = 1e-5;
      //for (unsigned int i = 1; i < barycentric_coords.size(); i++) {
      //  for (unsigned int j = 0; j < i; j++) {
      //    if (!approx_equal(barycentric_coords[i], barycentric_coords[j],
      //                      eps)) {
      //      return {};
      //    }
      //  }
      //}
      const auto pos = barycentric_coords.front()(0) * p0 +
                       barycentric_coords.front()(1) * p1 +
                       barycentric_coords.front()(2) * p2;
      return pos;

      //return {};
    }
  }

  //----------------------------------------------------------------------------
  static auto check_tet(const std::optional<vec3>& tri0,
                        const std::optional<vec3>& tri1,
                        const std::optional<vec3>& tri2,
                        const std::optional<vec3>& tri3) {
    std::vector<const std::optional<vec3>*> tris;
    if (tri0) { tris.push_back(&tri0); }
    if (tri1) { tris.push_back(&tri1); }
    if (tri2) { tris.push_back(&tri2); }
    if (tri3) { tris.push_back(&tri3); }

    std::vector<std::pair<vec3, vec3>> lines;
    if (tris.size() == 1) {
      // std::cerr << "only 1 point\n";
    } else if (tris.size() == 2) {
      lines.push_back({*(*tris[0]), *(*tris[1])});
    } else if (tris.size() == 3) {
      // std::cerr << "3 points\n";
    } else if (tris.size() == 4) {
      // std::cerr << "several solutions\n";
    }
    return lines;
  }

  //----------------------------------------------------------------------------
  auto operator()(real_t t = 0) { return calculate(t); }

  //----------------------------------------------------------------------------
  auto calculate(const real_t t = 0) {
    using boost::copy;
    std::vector<std::pair<vec3, vec3>> line_segments;
#ifdef NDEBUG
#pragma omp parallel for collapse(3)
#endif
    for (size_t iz = 0; iz < m_grid.dimension(2).size() - 1; ++iz) {
      for (size_t iy = 0; iy < m_grid.dimension(1).size() - 1; ++iy) {
        for (size_t ix = 0; ix < m_grid.dimension(0).size() - 1; ++ix) {
          const auto& x      = m_grid.dimension(0)[ix];
          const auto& next_x = m_grid.dimension(0)[ix + 1];
          const auto& y      = m_grid.dimension(1)[iy];
          const auto& next_y = m_grid.dimension(1)[iy + 1];
          const auto& z      = m_grid.dimension(2)[iz];
          const auto& next_z = m_grid.dimension(2)[iz + 1];
          std::array  p{
              vec{x, y, z},           vec{next_x, y, z},
              vec{x, next_y, z},      vec{next_x, next_y, z},
              vec{x, y, next_z},      vec{next_x, y, next_z},
              vec{x, next_y, next_z}, vec{next_x, next_y, next_z},
          };
          std::array v{
              m_v(p[0], t), m_v(p[1], t), m_v(p[2], t), m_v(p[3], t),
              m_v(p[4], t), m_v(p[5], t), m_v(p[6], t), m_v(p[7], t),
          };
          std::array w{
              m_w(p[0], t), m_w(p[1], t), m_w(p[2], t), m_w(p[3], t),
              m_w(p[4], t), m_w(p[5], t), m_w(p[6], t), m_w(p[7], t),
          };

          const bool xodd = ix % 2 == 0;
          const bool yodd = iy % 2 == 0;
          const bool zodd = iz % 2 == 0;

          bool turned = xodd;
          if (yodd) { turned = !turned; }
          if (zodd) { turned = !turned; }
          if (turned) {
            // check if there are parallel vectors on any of the tet's triangles
            auto pv012 =
                pv_on_tri(p[0], v[0], w[0], p[1], v[1], w[1], p[2], v[2], w[2]);
            auto pv014 =
                pv_on_tri(p[0], v[0], w[0], p[1], v[1], w[1], p[4], v[4], w[4]);
            auto pv024 =
                pv_on_tri(p[0], v[0], w[0], p[2], v[2], w[2], p[4], v[4], w[4]);
            auto pv124 =
                pv_on_tri(p[1], v[1], w[1], p[2], v[2], w[2], p[4], v[4], w[4]);
            auto pv246 =
                pv_on_tri(p[2], v[2], w[2], p[4], v[4], w[4], p[6], v[6], w[6]);
            auto pv247 =
                pv_on_tri(p[2], v[2], w[2], p[4], v[4], w[4], p[7], v[7], w[7]);
            auto pv267 =
                pv_on_tri(p[2], v[2], w[2], p[6], v[6], w[6], p[7], v[7], w[7]);
            auto pv467 =
                pv_on_tri(p[4], v[4], w[4], p[6], v[6], w[6], p[7], v[7], w[7]);
            auto pv145 =
                pv_on_tri(p[1], v[1], w[1], p[4], v[4], w[4], p[5], v[5], w[5]);
            auto pv147 =
                pv_on_tri(p[1], v[1], w[1], p[4], v[4], w[4], p[7], v[7], w[7]);
            auto pv157 =
                pv_on_tri(p[1], v[1], w[1], p[5], v[5], w[5], p[7], v[7], w[7]);
            auto pv457 =
                pv_on_tri(p[4], v[4], w[4], p[5], v[5], w[5], p[7], v[7], w[7]);
            auto pv123 =
                pv_on_tri(p[1], v[1], w[1], p[2], v[2], w[2], p[3], v[3], w[3]);
            auto pv127 =
                pv_on_tri(p[1], v[1], w[1], p[2], v[2], w[2], p[7], v[7], w[7]);
            auto pv137 =
                pv_on_tri(p[1], v[1], w[1], p[3], v[3], w[3], p[7], v[7], w[7]);
            auto pv237 =
                pv_on_tri(p[2], v[2], w[2], p[3], v[3], w[3], p[7], v[7], w[7]);

            // check the tets themselves
            // 0124
            copy(check_tet(pv012, pv014, pv024, pv124),
                 std::back_inserter(line_segments));
            // 2467
            copy(check_tet(pv246, pv247, pv267, pv467),
                 std::back_inserter(line_segments));
            // 1457
            copy(check_tet(pv145, pv147, pv157, pv457),
                 std::back_inserter(line_segments));
            // 1237
            copy(check_tet(pv123, pv127, pv137, pv237),
                 std::back_inserter(line_segments));
            // 1247
            copy(check_tet(pv124, pv127, pv147, pv247),
                 std::back_inserter(line_segments));
          } else {
            // std::cout << ix << ' ' << iy << ' ' << iz << " not turned\n";
            // check if there are parallel vectors on any of the tets triangles
            auto pv023 =
                pv_on_tri(p[0], v[0], w[0], p[2], v[2], w[2], p[3], v[3], w[3]);
            auto pv026 =
                pv_on_tri(p[0], v[0], w[0], p[2], v[2], w[2], p[6], v[6], w[6]);
            auto pv036 =
                pv_on_tri(p[0], v[0], w[0], p[3], v[3], w[3], p[6], v[6], w[6]);
            auto pv236 =
                pv_on_tri(p[2], v[2], w[2], p[3], v[3], w[3], p[6], v[6], w[6]);
            auto pv013 =
                pv_on_tri(p[0], v[0], w[0], p[1], v[1], w[1], p[3], v[3], w[3]);
            auto pv015 =
                pv_on_tri(p[0], v[0], w[0], p[1], v[1], w[1], p[5], v[5], w[5]);
            auto pv035 =
                pv_on_tri(p[0], v[0], w[0], p[3], v[3], w[3], p[5], v[5], w[5]);
            auto pv135 =
                pv_on_tri(p[1], v[1], w[1], p[3], v[3], w[3], p[5], v[5], w[5]);
            auto pv356 =
                pv_on_tri(p[3], v[3], w[3], p[5], v[5], w[5], p[6], v[6], w[6]);
            auto pv357 =
                pv_on_tri(p[3], v[3], w[3], p[5], v[5], w[5], p[7], v[7], w[7]);
            auto pv367 =
                pv_on_tri(p[3], v[3], w[3], p[6], v[6], w[6], p[7], v[7], w[7]);
            auto pv567 =
                pv_on_tri(p[5], v[5], w[5], p[6], v[6], w[6], p[7], v[7], w[7]);
            auto pv045 =
                pv_on_tri(p[0], v[0], w[0], p[4], v[4], w[4], p[5], v[5], w[5]);
            auto pv046 =
                pv_on_tri(p[0], v[0], w[0], p[4], v[4], w[4], p[6], v[6], w[6]);
            auto pv056 =
                pv_on_tri(p[0], v[0], w[0], p[5], v[5], w[5], p[6], v[6], w[6]);
            auto pv456 =
                pv_on_tri(p[4], v[4], w[4], p[5], v[5], w[5], p[6], v[6], w[6]);

            // check the tets themselves
            // 0236
            copy(check_tet(pv023, pv026, pv036, pv236),
                 std::back_inserter(line_segments));
            // 0135
            copy(check_tet(pv013, pv015, pv035, pv135),
                 std::back_inserter(line_segments));
            // 3567
            copy(check_tet(pv356, pv357, pv367, pv567),
                 std::back_inserter(line_segments));
            // 0456
            copy(check_tet(pv045, pv046, pv056, pv456),
                 std::back_inserter(line_segments));
            // 0356
            copy(check_tet(pv035, pv036, pv056, pv356),
                 std::back_inserter(line_segments));
          }
          turned = !turned;
        }
      }
    }

    // merge single line segments to line strips with a divide and conquer
    std::vector<line<real_t, 3>> lines;
    if (!line_segments.empty()) {
      auto line_strips = line_segments_to_line_strips(line_segments);

      for (const auto& line_strip : line_strips) {
        lines.emplace_back();
        for (size_t i = 0; i < line_strip.size() - 1; i++) {
          lines.back().push_back(line_strip[i]);
        }
        if (&line_strip.front() == &line_strip.back()) {
          lines.back().set_is_closed(true);
        } else {
          lines.back().push_back(line_strip.back());
        }
      }
    }

    return lines;
  }

  //----------------------------------------------------------------------------
  // const auto& grid() const { return m_grid; }
  // auto&       grid() { return m_grid; }

  //----------------------------------------------------------------------------
  auto min() const { return m_grid.min; }
  auto max() const { return m_grid.max; }
  auto res() const { return m_grid.resolution(); }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
