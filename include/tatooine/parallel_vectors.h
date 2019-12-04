#ifndef TATOOINE_PARALLEL_VECTORS_H
#define TATOOINE_PARALLEL_VECTORS_H

#include <omp.h>
#include <array>
#include <optional>
#include <tuple>
#include <vector>
#include "grid.h"
#include "line.h"
#include "openblas.h"
#include "type_traits.h"
#include "hdf5.h"
#include "field.h"

//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail_pv {
//==============================================================================
/// \brief      merge line strip indices
template <typename Real>
void merge(std::vector<std::vector<vec<Real, 3>>>& pairs_left,
           std::vector<std::vector<vec<Real, 3>>>& pairs_right) {
  const Real eps = 1e-7;
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
template <typename Real>
auto line_segments_to_line_strips(
    const std::vector<std::pair<vec<Real, 3>, vec<Real, 3>>>& line_segments) {
  std::vector<std::vector<std::vector<vec<Real, 3>>>> merged_strips(
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
        detail_pv::merge(merged_strips[left], merged_strips[right]);
      }
    }
  }
  return merged_strips.front();
}
//------------------------------------------------------------------------------
template <typename Real>
auto line_segments_to_lines(
    const std::vector<std::pair<vec<Real, 3>, vec<Real, 3>>>& line_segments) {
  std::vector<line<Real, 3>> lines;
  if (!line_segments.empty()) {
    auto line_strips = detail_pv::line_segments_to_line_strips(line_segments);

    for (const auto& line_strip : line_strips) {
      lines.emplace_back();
      for (size_t i = 0; i < line_strip.size() - 1; i++) {
        lines.back().push_back(line_strip[i]);
      }
      if (&line_strip.front() == &line_strip.back()) {
        lines.back().set_closed(true);
      } else {
        lines.back().push_back(line_strip.back());
      }
    }
  }
  return lines;
}
//----------------------------------------------------------------------------
template <typename Real, typename... Preds>
std::optional<vec<Real, 3>> pv_on_tri(
    const vec<Real, 3>& p0, const vec<Real, 3>& v0, const vec<Real, 3>& w0,
    const vec<Real, 3>& p1, const vec<Real, 3>& v1, const vec<Real, 3>& w1,
    const vec<Real, 3>& p2, const vec<Real, 3>& v2, const vec<Real, 3>& w2,
    Preds&&... preds) {
  mat<Real, 3, 3> v, w, m;
  v.col(0) = v0;
  v.col(1) = v1;
  v.col(2) = v2;
  w.col(0) = w0;
  w.col(1) = w1;
  w.col(2) = w2;

  openblas_set_num_threads(1);
  if (std::abs(det(v)) > 0) {
    m = gesv(v, w);
  } else if (std::abs(det(w)) > 0) {
    m = gesv(w, v);
  } else {
    return {};
  }

  auto [eigvecs, eigvals] = eigenvectors(m);
  auto ieig               = imag(eigvecs);
  auto reig               = real(eigvecs);

  std::vector<vec<Real, 3>> barycentric_coords;
  for (int i = 0; i < 3; i++) {
    if ((std::abs(ieig(0, i)) <= 0 && std::abs(ieig(1, i)) <= 0 &&
         std::abs(ieig(2, i)) <= 0) &&
        ((reig(0, i) <= 0 && reig(1, i) <= 0 && reig(2, i) <= 0) ||
         (reig(0, i) >= 0 && reig(1, i) >= 0 && reig(2, i) >= 0))) {
      const vec<Real, 3> bc = real(eigvecs.col(i)) / sum(real(eigvecs.col(i)));
      barycentric_coords.push_back(bc);
    }
  }

  if (barycentric_coords.empty()) {
    return {};

  } else if (barycentric_coords.size() == 1) {
    auto pos = barycentric_coords.front()(0) * p0 +
               barycentric_coords.front()(1) * p1 +
               barycentric_coords.front()(2) * p2;
    if ((preds(pos) && ...)) { return pos; }
    return {};

  } else {
    // check if all found barycentric coordinates are the same
    // const Real      eps = 1e-5;
    // for (unsigned int i = 1; i < barycentric_coords.size(); i++) {
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

    // return {};
  }
}

//----------------------------------------------------------------------------
template <typename Real>
static auto check_tet(const std::optional<vec<Real, 3>>& tri0,
                      const std::optional<vec<Real, 3>>& tri1,
                      const std::optional<vec<Real, 3>>& tri2,
                      const std::optional<vec<Real, 3>>& tri3) {
  std::vector<const std::optional<vec<Real, 3>>*> tris;
  if (tri0) { tris.push_back(&tri0); }
  if (tri1) { tris.push_back(&tri1); }
  if (tri2) { tris.push_back(&tri2); }
  if (tri3) { tris.push_back(&tri3); }

  std::vector<std::pair<vec<Real, 3>, vec<Real, 3>>> lines;
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
//------------------------------------------------------------------------------
bool turned(size_t ix, size_t iy, size_t iz) {
  const bool xodd = ix % 2 == 0;
  const bool yodd = iy % 2 == 0;
  const bool zodd = iz % 2 == 0;

  bool turned = xodd;
  if (yodd) { turned = !turned; }
  if (zodd) { turned = !turned; }
  return turned;
}
//------------------------------------------------------------------------------
template <typename Real, typename... Preds>
auto check_cell(
    std::vector<std::pair<vec<Real, 3>, vec<Real, 3>>>& line_segments,
    const std::array<vec<Real, 3>, 8>& p, const std::array<vec<Real, 3>, 8>& v,
    const std::array<vec<Real, 3>, 8>& w, size_t ix, size_t iy, size_t iz,
    omp_lock_t& writelock, Preds&&... preds) {
  using boost::copy;
  if (turned(ix, iy, iz)) {
    // check if there are parallel vectors on any of the tet's triangles
    auto pv012 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[1], v[1], w[1], p[2], v[2],
                             w[2], std::forward<Preds>(preds)...);
    auto pv014 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[1], v[1], w[1], p[4], v[4],
                             w[4], std::forward<Preds>(preds)...);
    auto pv024 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[2], v[2], w[2], p[4], v[4],
                             w[4], std::forward<Preds>(preds)...);
    auto pv124 =
        detail_pv::pv_on_tri(p[1], v[1], w[1], p[2], v[2], w[2], p[4], v[4],
                             w[4], std::forward<Preds>(preds)...);
    auto pv246 =
        detail_pv::pv_on_tri(p[2], v[2], w[2], p[4], v[4], w[4], p[6], v[6],
                             w[6], std::forward<Preds>(preds)...);
    auto pv247 =
        detail_pv::pv_on_tri(p[2], v[2], w[2], p[4], v[4], w[4], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv267 =
        detail_pv::pv_on_tri(p[2], v[2], w[2], p[6], v[6], w[6], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv467 =
        detail_pv::pv_on_tri(p[4], v[4], w[4], p[6], v[6], w[6], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv145 =
        detail_pv::pv_on_tri(p[1], v[1], w[1], p[4], v[4], w[4], p[5], v[5],
                             w[5], std::forward<Preds>(preds)...);
    auto pv147 =
        detail_pv::pv_on_tri(p[1], v[1], w[1], p[4], v[4], w[4], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv157 =
        detail_pv::pv_on_tri(p[1], v[1], w[1], p[5], v[5], w[5], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv457 =
        detail_pv::pv_on_tri(p[4], v[4], w[4], p[5], v[5], w[5], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv123 =
        detail_pv::pv_on_tri(p[1], v[1], w[1], p[2], v[2], w[2], p[3], v[3],
                             w[3], std::forward<Preds>(preds)...);
    auto pv127 =
        detail_pv::pv_on_tri(p[1], v[1], w[1], p[2], v[2], w[2], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv137 =
        detail_pv::pv_on_tri(p[1], v[1], w[1], p[3], v[3], w[3], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv237 =
        detail_pv::pv_on_tri(p[2], v[2], w[2], p[3], v[3], w[3], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);

#ifdef NDEBUG
    omp_set_lock(&writelock);

#endif
    // check the tets themselves
    // 0124
    copy(detail_pv::check_tet(pv012, pv014, pv024, pv124),
         std::back_inserter(line_segments));
    // 2467
    copy(detail_pv::check_tet(pv246, pv247, pv267, pv467),
         std::back_inserter(line_segments));
    // 1457
    copy(detail_pv::check_tet(pv145, pv147, pv157, pv457),
         std::back_inserter(line_segments));
    // 1237
    copy(detail_pv::check_tet(pv123, pv127, pv137, pv237),
         std::back_inserter(line_segments));
    // 1247
    copy(detail_pv::check_tet(pv124, pv127, pv147, pv247),
         std::back_inserter(line_segments));
#ifdef NDEBUG
    omp_unset_lock(&writelock);
#endif
  } else {
    // check if there are parallel vectors on any of the tets triangles
    auto pv023 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[2], v[2], w[2], p[3], v[3],
                             w[3], std::forward<Preds>(preds)...);
    auto pv026 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[2], v[2], w[2], p[6], v[6],
                             w[6], std::forward<Preds>(preds)...);
    auto pv036 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[3], v[3], w[3], p[6], v[6],
                             w[6], std::forward<Preds>(preds)...);
    auto pv236 =
        detail_pv::pv_on_tri(p[2], v[2], w[2], p[3], v[3], w[3], p[6], v[6],
                             w[6], std::forward<Preds>(preds)...);
    auto pv013 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[1], v[1], w[1], p[3], v[3],
                             w[3], std::forward<Preds>(preds)...);
    auto pv015 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[1], v[1], w[1], p[5], v[5],
                             w[5], std::forward<Preds>(preds)...);
    auto pv035 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[3], v[3], w[3], p[5], v[5],
                             w[5], std::forward<Preds>(preds)...);
    auto pv135 =
        detail_pv::pv_on_tri(p[1], v[1], w[1], p[3], v[3], w[3], p[5], v[5],
                             w[5], std::forward<Preds>(preds)...);
    auto pv356 =
        detail_pv::pv_on_tri(p[3], v[3], w[3], p[5], v[5], w[5], p[6], v[6],
                             w[6], std::forward<Preds>(preds)...);
    auto pv357 =
        detail_pv::pv_on_tri(p[3], v[3], w[3], p[5], v[5], w[5], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv367 =
        detail_pv::pv_on_tri(p[3], v[3], w[3], p[6], v[6], w[6], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv567 =
        detail_pv::pv_on_tri(p[5], v[5], w[5], p[6], v[6], w[6], p[7], v[7],
                             w[7], std::forward<Preds>(preds)...);
    auto pv045 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[4], v[4], w[4], p[5], v[5],
                             w[5], std::forward<Preds>(preds)...);
    auto pv046 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[4], v[4], w[4], p[6], v[6],
                             w[6], std::forward<Preds>(preds)...);
    auto pv056 =
        detail_pv::pv_on_tri(p[0], v[0], w[0], p[5], v[5], w[5], p[6], v[6],
                             w[6], std::forward<Preds>(preds)...);
    auto pv456 =
        detail_pv::pv_on_tri(p[4], v[4], w[4], p[5], v[5], w[5], p[6], v[6],
                             w[6], std::forward<Preds>(preds)...);
#ifdef NDEBUG
    omp_set_lock(&writelock);
#endif
    // check the tets themselves
    // 0236
    copy(detail_pv::check_tet(pv023, pv026, pv036, pv236),
         std::back_inserter(line_segments));
    // 0135
    copy(detail_pv::check_tet(pv013, pv015, pv035, pv135),
         std::back_inserter(line_segments));
    // 3567
    copy(detail_pv::check_tet(pv356, pv357, pv367, pv567),
         std::back_inserter(line_segments));
    // 0456
    copy(detail_pv::check_tet(pv045, pv046, pv056, pv456),
         std::back_inserter(line_segments));
    // 0356
    copy(detail_pv::check_tet(pv035, pv036, pv056, pv356),
         std::back_inserter(line_segments));
#ifdef NDEBUG
    omp_unset_lock(&writelock);
#endif
  }
}
//------------------------------------------------------------------------------
/// Framework for calculating PV Operator.
/// \param getv function for getting value for V field
/// \param getw function for getting value for W field
template <typename Real, typename GetV, typename GetW, typename GridReal,
          typename... Preds>
auto calc(GetV&& getv, GetW&& getw, const grid<GridReal, 3>& g,
          Preds&&... preds) {
  using vec3 = vec<Real, 3>;
  std::vector<std::pair<vec3, vec3>> line_segments;
#ifdef NDEBUG
  omp_lock_t writelock;
  omp_init_lock(&writelock);
#pragma omp parallel for collapse(3)
#endif
  for (size_t iz = 0; iz < g.dimension(2).size() - 1; ++iz) {
    for (size_t iy = 0; iy < g.dimension(1).size() - 1; ++iy) {
      for (size_t ix = 0; ix < g.dimension(0).size() - 1; ++ix) {
        const auto& x0 = g.dimension(0)[ix];
        const auto& x1 = g.dimension(0)[ix + 1];
        const auto& y0 = g.dimension(1)[iy];
        const auto& y1 = g.dimension(1)[iy + 1];
        const auto& z0 = g.dimension(2)[iz];
        const auto& z1 = g.dimension(2)[iz + 1];
        std::array  p{
            vec{x0, y0, z0}, vec{x1, y0, z0}, vec{x0, y1, z0}, vec{x1, y1, z0},
            vec{x0, y0, z1}, vec{x1, y0, z1}, vec{x0, y1, z1}, vec{x1, y1, z1},
        };
        check_cell(line_segments, p, getv(p, ix, iy, iz), getw(p, ix, iy, iz),
                   ix, iy, iz, writelock, std::forward<Preds>(preds)...);
      }
    }
  }
  return line_segments_to_lines(line_segments);
}
//==============================================================================
}  // namespace detail_pv
//==============================================================================
template <typename V, typename W, typename VReal, typename WReal,
          typename TReal, typename GridReal, typename... Preds,
          enable_if_arithmetic<TReal> = true>
auto parallel_vectors(const field<V, VReal, 3, 3>& vf,
                      const field<W, WReal, 3, 3>& wf, TReal t,
                      const grid<GridReal, 3>& g, Preds&&... preds) {
  return detail_pv::calc<promote_t<VReal, WReal>>(
      // get v data by evaluating V field
      [&vf, t](const auto& p, auto /*ix*/, auto /*iy*/, auto /*iz*/) {
        return std::array{vf(p[0], t), vf(p[1], t), vf(p[2], t), vf(p[3], t),
                          vf(p[4], t), vf(p[5], t), vf(p[6], t), vf(p[7], t)};
      },
      // get w data by evaluating W field
      [&wf, t](const auto& p, auto /*ix*/, auto /*iy*/, auto /*iz*/) {
        return std::array{wf(p[0], t), wf(p[1], t), wf(p[2], t), wf(p[3], t),
                          wf(p[4], t), wf(p[5], t), wf(p[6], t), wf(p[7], t)};
      },
      g, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename V, typename W, typename VReal, typename WReal,
          typename GridReal, typename... Preds>
auto parallel_vectors(const field<V, VReal, 3, 3>& v,
                      const field<W, WReal, 3, 3>& w,
                      const grid<GridReal, 3>&     g, Preds&&... preds) {
  return parallel_vectors(v, w, 0, g, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename V, typename W, typename VReal, typename WReal,
          typename XReal, typename YReal, typename ZReal, typename TReal,
          typename... Preds, enable_if_arithmetic<TReal> = true>
auto parallel_vectors(const field<V, VReal, 3, 3>& v,
                      const field<W, WReal, 3, 3>& w, TReal t,
                      const linspace<XReal>& x, const linspace<YReal>& y,
                      const linspace<ZReal>& z, Preds&&... preds) {
  return parallel_vectors(v, w, t, grid{x, y, z},
                          std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename V, typename W, typename VReal, typename WReal,
          typename XReal, typename YReal, typename ZReal, typename... Preds>
auto parallel_vectors(const field<V, VReal, 3, 3>& v,
                      const field<W, WReal, 3, 3>& w, const linspace<XReal>& x,
                      const linspace<YReal>& y, const linspace<ZReal>& z,
                      Preds&&... preds) {
  return parallel_vectors(v, w, 0, grid{x, y, z},
                          std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename VReal, typename WReal, typename GridReal, typename... Preds>
auto parallel_vectors(const h5::multi_array<vec<VReal, 3>>& vf,
                      const h5::multi_array<vec<WReal, 3>>& wf,
                      const grid<GridReal, 3>& g, Preds&&... preds) {
  assert(vf.num_dimensions() == 3);
  assert(wf.num_dimensions() == 3);
  assert(vf.resolution(0) == wf.resolution(0));
  assert(vf.resolution(1) == wf.resolution(1));
  assert(vf.resolution(2) == wf.resolution(2));
  assert(vf.resolution(0) == g.dimension(0).size());
  assert(vf.resolution(1) == g.dimension(1).size());
  assert(vf.resolution(2) == g.dimension(2).size());

  return detail_pv::calc<promote_t<VReal, WReal>>(
      [&vf](const auto& /*p*/, auto ix, auto iy, auto iz) {
        return std::array{vf(ix,   iy,   iz),
                          vf(ix+1, iy,   iz),
                          vf(ix,   iy+1, iz),
                          vf(ix+1, iy+1, iz),
                          vf(ix,   iy,   iz+1),
                          vf(ix+1, iy,   iz+1),
                          vf(ix,   iy+1, iz+1),
                          vf(ix+1, iy+1, iz+1)};
      },
      [&wf](const auto& /*p*/, auto ix, auto iy, auto iz) {
        return std::array{wf(ix,   iy,   iz),
                          wf(ix+1, iy,   iz),
                          wf(ix,   iy+1, iz),
                          wf(ix+1, iy+1, iz),
                          wf(ix,   iy,   iz+1),
                          wf(ix+1, iy,   iz+1),
                          wf(ix,   iy+1, iz+1),
                          wf(ix+1, iy+1, iz+1)};
      },
      g, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
