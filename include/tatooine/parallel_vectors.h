#ifndef TATOOINE_PARALLEL_VECTORS_H
#define TATOOINE_PARALLEL_VECTORS_H


#ifdef NDEBUG
#include <mutex>
#endif
#include <array>
#include <optional>
#include <tuple>
#include <vector>

#include "field.h"
#include "nested_for_loop.h"
#include "grid.h"
#include "line.h"
#include "openblas.h"
#include "type_traits.h"

//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail {
//==============================================================================
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

  const auto [eigvecs, eigvals] = eigenvectors(m);
  const auto ieig               = imag(eigvecs);
  const auto reig               = real(eigvecs);

  std::vector<vec<Real, 3>> barycentric_coords;
  for (size_t i = 0; i < 3; ++i) {
    if ((ieig(0, i) == 0 && ieig(1, i) == 0 && ieig(2, i) == 0) &&
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

  std::vector<line<Real, 3>> lines;
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
/// Framework for calculating PV Operator.
/// \param getv function for getting value for V field
/// \param getw function for getting value for W field
template <typename Real, typename GetV, typename GetW, typename GridReal,
          typename... Preds>
auto calc(GetV&& getv, GetW&& getw, const grid<GridReal, 3>& g,
          Preds&&... preds) {
  std::vector<line<Real, 3>> line_segments;

#ifdef NDEBUG
  std::mutex mutex;
#endif

  auto check_cell =
      [&line_segments, &getv, &getw, &g, &mutex, &preds...](size_t ix, size_t iy, size_t iz) {
        using boost::copy;
        std::array p{
            g(ix, iy, iz),         g(ix + 1, iy, iz),
            g(ix, iy + 1, iz),     g(ix + 1, iy + 1, iz),
            g(ix, iy, iz + 1),     g(ix + 1, iy, iz + 1),
            g(ix, iy + 1, iz + 1), g(ix + 1, iy + 1, iz + 1),
        };
        const decltype(auto) v0 = getv(ix  , iy  , iz  , p[0]);
        const decltype(auto) v1 = getv(ix+1, iy  , iz  , p[1]);
        const decltype(auto) v2 = getv(ix  , iy+1, iz  , p[2]);
        const decltype(auto) v3 = getv(ix+1, iy+1, iz  , p[3]);
        const decltype(auto) v4 = getv(ix  , iy  , iz+1, p[4]);
        const decltype(auto) v5 = getv(ix+1, iy  , iz+1, p[5]);
        const decltype(auto) v6 = getv(ix  , iy+1, iz+1, p[6]);
        const decltype(auto) v7 = getv(ix+1, iy+1, iz+1, p[7]);
        const decltype(auto) w0 = getw(ix  , iy  , iz  , p[0]);
        const decltype(auto) w1 = getw(ix+1, iy  , iz  , p[1]);
        const decltype(auto) w2 = getw(ix  , iy+1, iz  , p[2]);
        const decltype(auto) w3 = getw(ix+1, iy+1, iz  , p[3]);
        const decltype(auto) w4 = getw(ix  , iy  , iz+1, p[4]);
        const decltype(auto) w5 = getw(ix+1, iy  , iz+1, p[5]);
        const decltype(auto) w6 = getw(ix  , iy+1, iz+1, p[6]);
        const decltype(auto) w7 = getw(ix+1, iy+1, iz+1, p[7]);
        if (turned(ix, iy, iz)) {
          // check if there are parallel vectors on any of the tet's triangles
          [[maybe_unused]] auto pv012 =
              detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[2],
                                   v2, w2, std::forward<Preds>(preds)...);
          [[maybe_unused]] auto pv014 =
              detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[4],
                                   v4, w4, std::forward<Preds>(preds)...);
          auto pv024 =
              detail::pv_on_tri(p[0], v0, w0, p[2], v2, w2, p[4],
                                   v4, w4, std::forward<Preds>(preds)...);
          auto pv124 =
              detail::pv_on_tri(p[1], v1, w1, p[2], v2, w2, p[4],
                                   v4, w4, std::forward<Preds>(preds)...);
          auto pv246 =
              detail::pv_on_tri(p[2], v2, w2, p[4], v4, w4, p[6],
                                   v6, w6, std::forward<Preds>(preds)...);
          auto pv247 =
              detail::pv_on_tri(p[2], v2, w2, p[4], v4, w4, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv267 =
              detail::pv_on_tri(p[2], v2, w2, p[6], v6, w6, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv467 =
              detail::pv_on_tri(p[4], v4, w4, p[6], v6, w6, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv145 =
              detail::pv_on_tri(p[1], v1, w1, p[4], v4, w4, p[5],
                                   v5, w5, std::forward<Preds>(preds)...);
          auto pv147 =
              detail::pv_on_tri(p[1], v1, w1, p[4], v4, w4, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv157 =
              detail::pv_on_tri(p[1], v1, w1, p[5], v5, w5, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv457 =
              detail::pv_on_tri(p[4], v4, w4, p[5], v5, w5, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv123 =
              detail::pv_on_tri(p[1], v1, w1, p[2], v2, w2, p[3],
                                   v3, w3, std::forward<Preds>(preds)...);
          auto pv127 =
              detail::pv_on_tri(p[1], v1, w1, p[2], v2, w2, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv137 =
              detail::pv_on_tri(p[1], v1, w1, p[3], v3, w3, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv237 =
              detail::pv_on_tri(p[2], v2, w2, p[3], v3, w3, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);

#ifdef NDEBUG
          {
            std::lock_guard lock{mutex};
#endif
            // check the tets themselves
            // 0124
            copy(detail::check_tet(pv012, pv014, pv024, pv124),
                 std::back_inserter(line_segments));
            // 2467
            copy(detail::check_tet(pv246, pv247, pv267, pv467),
                 std::back_inserter(line_segments));
            // 1457
            copy(detail::check_tet(pv145, pv147, pv157, pv457),
                 std::back_inserter(line_segments));
            // 1237
            copy(detail::check_tet(pv123, pv127, pv137, pv237),
                 std::back_inserter(line_segments));
            // 1247
            copy(detail::check_tet(pv124, pv127, pv147, pv247),
                 std::back_inserter(line_segments));
#ifdef NDEBUG
          }
#endif
        } else {
          // check if there are parallel vectors on any of the tets triangles
          auto pv023 =
              detail::pv_on_tri(p[0], v0, w0, p[2], v2, w2, p[3],
                                   v3, w3, std::forward<Preds>(preds)...);
          auto pv026 =
              detail::pv_on_tri(p[0], v0, w0, p[2], v2, w2, p[6],
                                   v6, w6, std::forward<Preds>(preds)...);
          auto pv036 =
              detail::pv_on_tri(p[0], v0, w0, p[3], v3, w3, p[6],
                                   v6, w6, std::forward<Preds>(preds)...);
          auto pv236 =
              detail::pv_on_tri(p[2], v2, w2, p[3], v3, w3, p[6],
                                   v6, w6, std::forward<Preds>(preds)...);
          auto pv013 =
              detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[3],
                                   v3, w3, std::forward<Preds>(preds)...);
          auto pv015 =
              detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[5],
                                   v5, w5, std::forward<Preds>(preds)...);
          auto pv035 =
              detail::pv_on_tri(p[0], v0, w0, p[3], v3, w3, p[5],
                                   v5, w5, std::forward<Preds>(preds)...);
          auto pv135 =
              detail::pv_on_tri(p[1], v1, w1, p[3], v3, w3, p[5],
                                   v5, w5, std::forward<Preds>(preds)...);
          auto pv356 =
              detail::pv_on_tri(p[3], v3, w3, p[5], v5, w5, p[6],
                                   v6, w6, std::forward<Preds>(preds)...);
          auto pv357 =
              detail::pv_on_tri(p[3], v3, w3, p[5], v5, w5, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv367 =
              detail::pv_on_tri(p[3], v3, w3, p[6], v6, w6, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv567 =
              detail::pv_on_tri(p[5], v5, w5, p[6], v6, w6, p[7],
                                   v7, w7, std::forward<Preds>(preds)...);
          auto pv045 =
              detail::pv_on_tri(p[0], v0, w0, p[4], v4, w4, p[5],
                                   v5, w5, std::forward<Preds>(preds)...);
          auto pv046 =
              detail::pv_on_tri(p[0], v0, w0, p[4], v4, w4, p[6],
                                   v6, w6, std::forward<Preds>(preds)...);
          auto pv056 =
              detail::pv_on_tri(p[0], v0, w0, p[5], v5, w5, p[6],
                                   v6, w6, std::forward<Preds>(preds)...);
          auto pv456 =
              detail::pv_on_tri(p[4], v4, w4, p[5], v5, w5, p[6],
                                   v6, w6, std::forward<Preds>(preds)...);
#ifdef NDEBUG
          {
            std::lock_guard lock{mutex};
#endif
            // check the tets themselves
            // 0236
            copy(detail::check_tet(pv023, pv026, pv036, pv236),
                 std::back_inserter(line_segments));
            // 0135
            copy(detail::check_tet(pv013, pv015, pv035, pv135),
                 std::back_inserter(line_segments));
            // 3567
            copy(detail::check_tet(pv356, pv357, pv367, pv567),
                 std::back_inserter(line_segments));
            // 0456
            copy(detail::check_tet(pv045, pv046, pv056, pv456),
                 std::back_inserter(line_segments));
            // 0356
            copy(detail::check_tet(pv035, pv036, pv056, pv356),
                 std::back_inserter(line_segments));
#ifdef NDEBUG
          }
#endif
        }
      };
#ifdef NDEBUG
  parallel_nested_for(check_cell, g.size(0) - 1, g.size(1) - 1, g.size(2) - 1);
#else
  nested_for(check_cell, g.size(0) - 1, g.size(1) - 1, g.size(2) - 1);
#endif
  return merge_lines(line_segments);
}
//==============================================================================
}  // namespace detail
//==============================================================================
template <typename V, typename W, typename VReal, typename WReal,
          typename TReal, typename GridReal, typename... Preds,
          enable_if_arithmetic<TReal> = true>
auto parallel_vectors(const field<V, VReal, 3, 3>& vf,
                      const field<W, WReal, 3, 3>& wf,
                      const grid<GridReal, 3>& g, TReal t, Preds&&... preds) {
  return detail::calc<promote_t<VReal, WReal>>(
      // get v data by evaluating V field
      [&vf, t](auto /*ix*/, auto /*iy*/, auto /*iz*/, const auto& p) {
        return vf(p, t);
      },
      // get w data by evaluating W field
      [&wf, t](auto /*ix*/, auto /*iy*/, auto /*iz*/, const auto& p) {
        return wf(p, t);
      },
      g, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename V, typename W, typename VReal, typename WReal,
          typename GridReal, typename... Preds>
auto parallel_vectors(const field<V, VReal, 3, 3>& v,
                      const field<W, WReal, 3, 3>& w,
                      const grid<GridReal, 3>&     g, Preds&&... preds) {
  return parallel_vectors(v, w, g, 0, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename V, typename W, typename VReal, typename WReal,
          typename XReal, typename YReal, typename ZReal, typename TReal,
          typename... Preds, enable_if_arithmetic<TReal> = true>
auto parallel_vectors(const field<V, VReal, 3, 3>& v,
                      const field<W, WReal, 3, 3>& w,
                      const linspace<XReal>& x, const linspace<YReal>& y,
                      const linspace<ZReal>& z, TReal t, Preds&&... preds) {
  return parallel_vectors(v, w, grid{x, y, z}, t,
                          std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename V, typename W, typename VReal, typename WReal,
          typename XReal, typename YReal, typename ZReal, typename... Preds>
auto parallel_vectors(const field<V, VReal, 3, 3>& v,
                      const field<W, WReal, 3, 3>& w, const linspace<XReal>& x,
                      const linspace<YReal>& y, const linspace<ZReal>& z,
                      Preds&&... preds) {
  return parallel_vectors(v, w, grid{x, y, z}, 0, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename VReal, typename VIndexing, typename WReal,
          typename WIndexing, typename BBReal, typename... Preds>
auto parallel_vectors(
    const dynamic_multidim_array<vec<VReal, 3>, VIndexing>&        vf,
    const dynamic_multidim_array<vec<WReal, 3>, WIndexing>& wf,
    const boundingbox<BBReal, 3>& bb, Preds&&... preds) {
  assert(vf.num_dimensions() == 3);
  assert(wf.num_dimensions() == 3);
  assert(vf.size(0) == wf.size(0));
  assert(vf.size(1) == wf.size(1));
  assert(vf.size(2) == wf.size(2));

  return detail::calc<promote_t<VReal, WReal>>(
      [&vf](auto ix, auto iy, auto iz, const auto & /*p*/) -> const auto& {
        return vf(ix, iy, iz);
      },
      [&wf](auto ix, auto iy, auto iz, const auto & /*p*/) -> const auto& {
        return wf(ix, iy, iz);
      },
      grid{linspace{bb.min(0), bb.max(0), vf.size(0)},
           linspace{bb.min(1), bb.max(1), vf.size(1)},
           linspace{bb.min(2), bb.max(2), vf.size(2)}},
      std::forward<Preds>(preds)...);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
