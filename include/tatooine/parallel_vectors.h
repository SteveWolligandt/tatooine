#ifndef TATOOINE_PARALLEL_VECTORS_H
#define TATOOINE_PARALLEL_VECTORS_H
//==============================================================================
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
#include <tatooine/cache_alignment.h>

#include <mutex>
#endif
#include <tatooine/dynamic_multidim_array.h>
#include <tatooine/field.h>
#include <tatooine/for_loop.h>
#include <tatooine/line.h>
#include <tatooine/rectilinear_grid.h>
//#include <tatooine/openblas.h>
#include <tatooine/type_traits.h>

#include <array>
#include <boost/range/adaptors.hpp>
#include <boost/range/numeric.hpp>
#include <optional>
#include <tuple>
#include <vector>
//==============================================================================
/** \page parallel_vectors_page Parallel Vectors

\section pv_cube_sec Cube Setup

A cube's vertices are numbered like this where
- `0 -> 1` corresponds to `x`-direction,
- `0 -> 2` corresponds to `y`-direction and
- `0 -> 4` corresponds to `z`-direction.

```
     6---------7
    /.        /|
   / .       / |
  2---------3  |
  |  4. . . |. 5
  | .       | /
  |.        |/
  0---------1
```
A cube can be decomposed into 5 tetrahedrons in 2 ways. The first one is called
'non-turned' and the other one is called 'turned'.

\subsection pv_cube_tet_sec Tetrahedrons per cube
The tetrahedrons are setup like this:
- turned tetrahedrons per cube:
  - `[0]`: `0236` (with triangles `023`, `026`, `036`, `236`)
  - `[1]`: `0135` (with triangles `013`, `015`, `035`, `135`)
  - `[2]`: `3567` (with triangles `356`, `357`, `367`, `567`)
  - `[3]`: `0456` (with triangles `045`, `046`, `056`, `456`)
  - `[4]`: `0356` (with triangles `035`, `036`, `056`, `356`)
- non-turned tetrahedrons per cube:
  - `[0]`: `0124` (with triangles `012`, `014`, `024`, `124`)
  - `[1]`: `1457` (with triangles `145`, `147`, `157`, `457`)
  - `[2]`: `2467` (with triangles `246`, `247`, `267`, `467`)
  - `[3]`: `1237` (with triangles `123`, `127`, `137`, `237`)
  - `[4]`: `1247` (with triangles `124`, `127`, `147`, `247`)
  
\subsection pv_cube_x_sec Faces with constant x
- turned inner tet order:
  - `[0]`: `046` / `157`
  - `[1]`: `026` / `137`
- non-turned inner tet order:
  - `[0]`: `024` / `135`
  - `[1]`: `246` / `357`

\subsection pv_cube_y_sec Faces with constant y
- turned inner tet order:
  - `[0]`: `015` / `237`
  - `[1]`: `045` / `267`
- non-turned inner tet order:
  - `[0]`: `014` / `236`
  - `[1]`: `145` / `367`

\subsection pv_cube_z_sec Faces with constant z
- turned inner tet order:
  - `[0]`: `013` / `457`
  - `[1]`: `023` / `467`
- non-turned inner tet order:
  - `[0]`: `012` / `456`
  - `[1]`: `123` / `567`

\subsection pv_cube_z_sec Inner tetrahedron's faces
- turned inner tet order:
  - `[0]`: `035`
  - `[1]`: `036`
  - `[2]`: `356`
  - `[3]`: `056`
- non-turned inner tet order:
  - `[0]`: `124`
  - `[1]`: `127`
  - `[2]`: `147`
  - `[3]`: `247`
*/
//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail {
static constexpr auto num_tets_per_cube = std::size_t(5);
//==============================================================================
/// \return Position where v and w are parallel otherwise nothing.
#ifdef __cpp_concepts
template <typename Real, invocable<vec<Real, 3>>... Preds>
#else
template <typename Real, typename... Preds,
          enable_if<is_invocable<Preds, vec<Real, 3>>...> = true>
#endif
constexpr auto pv_on_tri(vec<Real, 3> const& p0, vec<Real, 3> const& v0,
                         vec<Real, 3> const& w0, vec<Real, 3> const& p1,
                         vec<Real, 3> const& v1, vec<Real, 3> const& w1,
                         vec<Real, 3> const& p2, vec<Real, 3> const& v2,
                         vec<Real, 3> const& w2, Preds&&... preds)
    -> std::optional<vec<Real, 3>> {
  mat<Real, 3, 3> V, W, M;
  V.col(0) = v0;
  V.col(1) = v1;
  V.col(2) = v2;
  W.col(0) = w0;
  W.col(1) = w1;
  W.col(2) = w2;

  if (std::abs(det(V)) > 0) {
    M = solve(V, W);
  } else if (std::abs(det(W)) > 0) {
    M = solve(W, V);
  } else {
    return {};
  }

  auto const [eigvecs, eigvals] = eigenvectors(M);
  auto const ieig               = imag(eigvecs);
  auto const reig               = real(eigvecs);

  auto barycentric_coords = std::vector<vec<Real, 3>>{};
  for (std::size_t i = 0; i < 3; ++i) {
    if ((ieig(0, i) == 0 && ieig(1, i) == 0 && ieig(2, i) == 0) &&
        ((reig(0, i) <= 0 && reig(1, i) <= 0 && reig(2, i) <= 0) ||
         (reig(0, i) >= 0 && reig(1, i) >= 0 && reig(2, i) >= 0))) {
      vec<Real, 3> const bc = real(eigvecs.col(i)) / sum(real(eigvecs.col(i)));
      barycentric_coords.push_back(bc);
    }
  }

  if (barycentric_coords.empty()) {
    return {};
  }

  if (barycentric_coords.size() == 1) {
    auto pos = barycentric_coords.front()(0) * p0 +
               barycentric_coords.front()(1) * p1 +
               barycentric_coords.front()(2) * p2;
    if constexpr (sizeof...(Preds) > 0) {
      if ((preds(pos) && ...)) {
        return pos;
      }
      return {};
    } else {
      return pos;
    }
  } else {
    // check if all found barycentric coordinates are the same
    Real const eps = 1e-5;
    for (std::size_t i = 1; i < barycentric_coords.size(); i++) {
      for (std::size_t j = 0; j < i; j++) {
        if (!approx_equal(barycentric_coords[i], barycentric_coords[j], eps)) {
          return {};
        }
      }
    }
    auto const pos = barycentric_coords.front()(0) * p0 +
                     barycentric_coords.front()(1) * p1 +
                     barycentric_coords.front()(2) * p2;
    return pos;
  }
}
//------------------------------------------------------------------------------
template <typename Dim0, typename Dim1, typename Dim2>
auto extract_indices(rectilinear_grid<Dim0, Dim1, Dim2> const& g, auto const idx){
  auto const cube_idx        = idx / num_tets_per_cube;
  auto       aux_idx         = cube_idx;
  auto const tet_idx_in_cube = idx % num_tets_per_cube;
  auto const iz = cube_idx / (g.template size<0>() * g.template size<1>());
  aux_idx -= iz * g.template size<0>() * g.template size<1>();
  auto const iy = aux_idx / g.template size<0>();
  auto const ix = aux_idx - iy * g.template size<0>();
  return std::tuple{ix, iy, iz, cube_idx, tet_idx_in_cube};
}
//------------------------------------------------------------------------------
auto constexpr turned(std::size_t const ix, std::size_t const iy,
                      std::size_t const iz) -> bool {
  auto const xodd = ix % 2 == 0 ? 1 : -1;
  auto const yodd = iy % 2 == 0 ? 1 : -1;
  auto const zodd = iz % 2 == 0 ? 1 : -1;
  return xodd * yodd * zodd < 0;
}
//----------------------------------------------------------------------------
template <typename Real>
static auto check_pv_in_tet(std::optional<vec<Real, 3>> const& tri0,
                            std::optional<vec<Real, 3>> const& tri1,
                            std::optional<vec<Real, 3>> const& tri2,
                            std::optional<vec<Real, 3>> const& tri3) {
  return tri0.has_value() || tri1.has_value() || tri2.has_value() ||
         tri3.has_value();
}
//----------------------------------------------------------------------------
template <typename Real>
static auto extract_pv_lines_from_tet(
    std::optional<vec<Real, 3>> const& tri0,
    std::optional<vec<Real, 3>> const& tri1,
    std::optional<vec<Real, 3>> const& tri2,
    std::optional<vec<Real, 3>> const& tri3,
    std::vector<line<Real, 3>>& lines,
    std::mutex& mutex) {
  std::vector<std::optional<vec<Real, 3>> const*> tris;
  if (tri0) {
    tris.push_back(&tri0);
  }
  if (tri1) {
    tris.push_back(&tri1);
  }
  if (tri2) {
    tris.push_back(&tri2);
  }
  if (tri3) {
    tris.push_back(&tri3);
  }

  if (tris.size() == 1) {
    // std::cerr << "only 1 point\n";
  } else if (tris.size() == 2) {
    auto  lock = std::lock_guard{mutex};
    auto& l    = lines.emplace_back();
    l.push_back(*(*tris[0]));
    l.push_back(*(*tris[1]));
  } else if (tris.size() == 3) {
    // std::cerr << "3 points\n";
  } else if (tris.size() == 4) {
    // std::cerr << "several solutions\n";
  }
}
template <typename Real, typename Dim0, typename Dim1, typename Dim2,
          typename GetV, typename GetW>
static auto extract_pv_lines_from_tet(
    rectilinear_grid<Dim0, Dim1, Dim2> const& g, GetV&& getv, GetW&& getw,
    vec<Real, 3> const& p0, vec<Real, 3> const& p1, vec<Real, 3> const& p2,
    vec<Real, 3> const& p3, std::size_t const num_sub_divisions,
    std::vector<line<Real, 3>>& lines, std::mutex& mutex) {
  if (num_sub_divisions == 0) {
    auto v0 = getv(p0);
    auto v1 = getv(p1);
    auto v2 = getv(p2);
    auto v3 = getv(p3);
    auto w0 = getw(p0);
    auto w1 = getw(p1);
    auto w2 = getw(p2);
    auto w3 = getw(p3);
    if (num_sub_division == 0) {
      extract_pv_lines_from_tet(pv_on_tri(p0, v0, w0, p1, v1, w1, p2, v2, w2),
                                pv_on_tri(p0, v0, w0, p1, v1, w1, p3, v3, w3),
                                pv_on_tri(p0, v0, w0, p2, v2, w2, p3, v3, w3),
                                pv_on_tri(p1, v1, w1, p2, v2, w2, p3, v3, w3), lines, mutex);
    }
  } else {
    auto const p01 = (p0+p1)/2;
    auto const p02 = (p0+p2)/2;
    auto const p03 = (p0+p3)/2;
    auto const p12 = (p1+p2)/2;
    auto const p13 = (p1+p3)/2;
    auto const p23 = (p2+p3)/2;
    extract_pv_lines_from_tet(g, getv, getw, p0, p01, p02, p03, num_subdivisions-1, lines, mutex);
    extract_pv_lines_from_tet(g, getv, getw, p1, p01, p12, p13, num_subdivisions-1, lines, mutex);
    extract_pv_lines_from_tet(g, getv, getw, p2, p02, p12, p23, num_subdivisions-1, lines, mutex);
    extract_pv_lines_from_tet(g, getv, getw, p3, p03, p13, p23, num_subdivisions-1, lines, mutex);
    extract_pv_lines_from_tet(g, getv, getw, p01, p02, p03, p, num_subdivisions-1, lines, mutex);
  }
}
//------------------------------------------------------------------------------
template <typename Real, typename Dim0, typename Dim1, typename Dim2, typename GetV,
          typename GetW>
static auto extract_pv_lines_from_tet(
    rectilinear_grid<Dim0, Dim1, Dim2> const& g, GetV& getv, GetW& getw,
    std::size_t const tet_idx, std::size_t const num_sub_divisions,
    std::vector<line<Real, 3>>& lines,
    std::mutex& mutex) {
    if (num_sub_division == 0) {
extract_pv_lines_from_tet
    }
}
//------------------------------------------------------------------------------
/// Framework for calculating PV Operator.
/// \param getv function for getting value for V field
/// \param getw function for getting value for W field
#ifdef __cpp_concepts
template <typename Real, typename GetV, typename GetW, indexable_space XDomain,
          indexable_space YDomain, indexable_space ZDomain,
          invocable<vec<Real, 3>>... Preds>
#else
template <typename Real, typename GetV, typename GetW, typename XDomain,
          typename YDomain, typename ZDomain, typename... Preds,
          enable_if<is_invocable<Preds, vec<Real, 3>>...> = true>
#endif
auto calc_parallel_vectors(GetV&& getv, GetW&& getw,
                           rectilinear_grid<XDomain, YDomain, ZDomain> const& g,
                           Preds&&... preds) -> std::vector<line<Real, 3>> {
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
  auto constexpr policy = execution_policy::parallel;
  auto mutex            = std::mutex{};
#else
  auto constexpr policy = execution_policy::sequential;
#endif

  auto tets_with_pvs = std::vector<std::size_t>{};
  using boost::copy;
  using vec3 = vec<Real, 3>;
  auto iz    = std::size_t(0);
  auto x_faces = dynamic_multidim_array<std::array<std::optional<vec3>, 2>>{
      g.template size<0>(), g.template size<1>() - 1};
  auto update_x_faces = [&](std::size_t const ix, std::size_t const iy) {
    auto const gv = g.vertices();
    auto const p  = std::array{
        gv(ix, iy, iz),         // 0
        gv(ix, iy, iz + 1),     // 4
        gv(ix, iy + 1, iz),     // 2
        gv(ix, iy + 1, iz + 1)  // 6
    };

    decltype(auto) v0 = getv(ix, iy, iz, p[0]);
    if (isnan(v0)) {
      return;
    }
    decltype(auto) v4 = getv(ix, iy, iz + 1, p[1]);
    if (isnan(v4)) {
      return;
    }
    decltype(auto) v2 = getv(ix, iy + 1, iz, p[2]);
    if (isnan(v2)) {
      return;
    }
    decltype(auto) v6 = getv(ix, iy + 1, iz + 1, p[3]);
    if (isnan(v6)) {
      return;
    }
    decltype(auto) w0 = getw(ix, iy, iz, p[0]);
    if (isnan(w0)) {
      return;
    }
    decltype(auto) w4 = getw(ix, iy, iz + 1, p[1]);
    if (isnan(w4)) {
      return;
    }
    decltype(auto) w2 = getw(ix, iy + 1, iz, p[2]);
    if (isnan(w2)) {
      return;
    }
    decltype(auto) w6 = getw(ix, iy + 1, iz + 1, p[3]);
    if (isnan(w6)) {
      return;
    }
    if (turned(ix, iy, iz)) {
      x_faces(ix, iy)[0] =  // 046
          detail::pv_on_tri(p[0], v0, w0, p[1], v4, w4, p[3], v6, w6,
                            std::forward<Preds>(preds)...);
      x_faces(ix, iy)[1] =  // 026
          detail::pv_on_tri(p[0], v0, w0, p[3], v6, w6, p[2], v2, w2,
                            std::forward<Preds>(preds)...);
    } else {
      x_faces(ix, iy)[0] =  // 024
          detail::pv_on_tri(p[0], v0, w0, p[1], v4, w4, p[2], v2, w2,
                            std::forward<Preds>(preds)...);
      x_faces(ix, iy)[1] =  // 246
          detail::pv_on_tri(p[1], v4, w4, p[3], v6, w6, p[2], v2, w2,
                            std::forward<Preds>(preds)...);
    }
  };
  auto y_faces = dynamic_multidim_array<std::array<std::optional<vec3>, 2>>{
      g.template size<0>() - 1, g.template size<1>()};
  auto update_y_faces = [&g, &getv, &getw, &y_faces, &iz, &preds...](
                            std::size_t const ix, std::size_t const iy) {
    auto const gv = g.vertices();
    auto const p0 = gv(ix, iy, iz);
    auto const p1 = gv(ix + 1, iy, iz);
    auto const p4 = gv(ix, iy, iz + 1);
    auto const p5 = gv(ix + 1, iy, iz + 1);

    decltype(auto) v0 = getv(ix, iy, iz, p0);
    if (isnan(v0)) {
      return;
    }
    decltype(auto) v1 = getv(ix + 1, iy, iz, p1);
    if (isnan(v1)) {
      return;
    }
    decltype(auto) v4 = getv(ix, iy, iz + 1, p4);
    if (isnan(v4)) {
      return;
    }
    decltype(auto) v5 = getv(ix + 1, iy, iz + 1, p5);
    if (isnan(v5)) {
      return;
    }
    decltype(auto) w0 = getw(ix, iy, iz, p0);
    if (isnan(w0)) {
      return;
    }
    decltype(auto) w1 = getw(ix + 1, iy, iz, p1);
    if (isnan(w1)) {
      return;
    }
    decltype(auto) w4 = getw(ix, iy, iz + 1, p4);
    if (isnan(w4)) {
      return;
    }
    decltype(auto) w5 = getw(ix + 1, iy, iz + 1, p5);
    if (isnan(w5)) {
      return;
    }
    if (turned(ix, iy, iz)) {
      y_faces(ix, iy)[0] =  // 015
          detail::pv_on_tri(p0, v0, w0, p1, v1, w1, p5, v5, w5,
                            std::forward<Preds>(preds)...);
      y_faces(ix, iy)[1] =  // 045
          detail::pv_on_tri(p0, v0, w0, p5, v5, w5, p4, v4, w4,
                            std::forward<Preds>(preds)...);
    } else {
      y_faces(ix, iy)[0] =  // 014
          detail::pv_on_tri(p0, v0, w0, p1, v1, w1, p4, v4, w4,
                            std::forward<Preds>(preds)...);
      y_faces(ix, iy)[1] =  // 145
          detail::pv_on_tri(p1, v1, w1, p5, v5, w5, p4, v4, w4,
                            std::forward<Preds>(preds)...);
    }
  };
  auto z_faces = dynamic_multidim_array<std::array<std::optional<vec3>, 2>>{
      2, g.template size<0>() - 1, g.template size<1>() - 1};
  auto update_z = [&z_faces, &g, &getv, &getw, &preds...](
                      std::size_t const ix, std::size_t const iy,
                      std::size_t const iz, std::size_t const write_iz) {
    assert(write_iz == 0 || write_iz == 1);
    auto const gv = g.vertices();
    auto const p0 = gv(ix, iy, iz);
    auto const p1 = gv(ix + 1, iy, iz);
    auto const p2 = gv(ix, iy + 1, iz);
    auto const p3 = gv(ix + 1, iy + 1, iz);

    decltype(auto) v0 = getv(ix, iy, iz, p0);
    if (isnan(v0)) {
      return;
    }
    decltype(auto) v1 = getv(ix + 1, iy, iz, p1);
    if (isnan(v1)) {
      return;
    }
    decltype(auto) v2 = getv(ix, iy + 1, iz, p2);
    if (isnan(v2)) {
      return;
    }
    decltype(auto) v3 = getv(ix + 1, iy + 1, iz, p3);
    if (isnan(v3)) {
      return;
    }
    decltype(auto) w0 = getw(ix, iy, iz, p0);
    if (isnan(w0)) {
      return;
    }
    decltype(auto) w1 = getw(ix + 1, iy, iz, p1);
    if (isnan(w1)) {
      return;
    }
    decltype(auto) w2 = getw(ix, iy + 1, iz, p2);
    if (isnan(w2)) {
      return;
    }
    decltype(auto) w3 = getw(ix + 1, iy + 1, iz, p3);
    if (isnan(w3)) {
      return;
    }
    if (turned(ix, iy, iz)) {
      z_faces(write_iz, ix, iy)[0] =  // 013
          detail::pv_on_tri(p0, v0, w0, p1, v1, w1, p3, v3, w3,
                            std::forward<Preds>(preds)...);
      z_faces(write_iz, ix, iy)[1] =  // 023
          detail::pv_on_tri(p0, v0, w0, p3, v3, w3, p2, v2, w2,
                            std::forward<Preds>(preds)...);
    } else {
      z_faces(write_iz, ix, iy)[0] =  // 012
          detail::pv_on_tri(p0, v0, w0, p1, v1, w1, p2, v2, w2,
                            std::forward<Preds>(preds)...);
      z_faces(write_iz, ix, iy)[1] =  // 123
          detail::pv_on_tri(p1, v1, w1, p3, v3, w3, p2, v2, w2,
                            std::forward<Preds>(preds)...);
    }
  };
  auto move_zs = [&]() {
    tatooine::for_loop(
        [&](auto const... is) { z_faces(0, is...) = z_faces(1, is...); },
        policy, g.template size<0>() - 1, g.template size<1>() - 1);
  };
  // initialize front constant-z-face
  tatooine::for_loop([&](auto const... is) { update_z(is..., 0, 0); }, policy,
                     g.template size<0>() - 1, g.template size<1>() - 1);
  auto update_z_faces = [&](std::size_t const ix, std::size_t const iy) {
    update_z(ix, iy, iz + 1, 1);
  };

  auto inner_faces = dynamic_multidim_array<std::array<std::optional<vec3>, 4>>{
      g.template size<0>() - 1, g.template size<1>() - 1};
  auto update_inner_faces = [&](std::size_t ix, std::size_t iy) {
    auto const gv = g.vertices();
    if (turned(ix, iy, iz)) {
      auto const p0 = gv(ix, iy, iz);
      auto const p3 = gv(ix + 1, iy + 1, iz);
      auto const p5 = gv(ix + 1, iy, iz + 1);
      auto const p6 = gv(ix, iy + 1, iz + 1);

      decltype(auto) v0 = getv(ix, iy, iz, p0);
      if (isnan(v0)) {
        return;
      }
      decltype(auto) v3 = getv(ix + 1, iy + 1, iz, p3);
      if (isnan(v3)) {
        return;
      }
      decltype(auto) v5 = getv(ix + 1, iy, iz + 1, p5);
      if (isnan(v5)) {
        return;
      }
      decltype(auto) v6 = getv(ix, iy + 1, iz + 1, p6);
      if (isnan(v6)) {
        return;
      }
      decltype(auto) w0 = getw(ix, iy, iz, p0);
      if (isnan(w0)) {
        return;
      }
      decltype(auto) w3 = getw(ix + 1, iy + 1, iz, p3);
      if (isnan(w3)) {
        return;
      }
      decltype(auto) w5 = getw(ix + 1, iy, iz + 1, p5);
      if (isnan(w5)) {
        return;
      }
      decltype(auto) w6 = getw(ix, iy + 1, iz + 1, p6);
      if (isnan(w6)) {
        return;
      }
      inner_faces(ix, iy)[0] =  // 035
          detail::pv_on_tri(p0, v0, w0, p5, v5, w5, p3, v3, w3,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy)[1] =  // 036
          detail::pv_on_tri(p0, v0, w0, p3, v3, w3, p6, v6, w6,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy)[2] =  // 356
          detail::pv_on_tri(p3, v3, w3, p5, v5, w5, p6, v6, w6,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy)[3] =  // 056
          detail::pv_on_tri(p0, v0, w0, p6, v6, w6, p5, v5, w5,
                            std::forward<Preds>(preds)...);
    } else {
      auto const p1 = gv(ix + 1, iy, iz);
      auto const p2 = gv(ix, iy + 1, iz);
      auto const p4 = gv(ix, iy, iz + 1);
      auto const p7 = gv(ix + 1, iy + 1, iz + 1);

      decltype(auto) v1 = getv(ix + 1, iy, iz, p1);
      if (isnan(v1)) {
        return;
      }
      decltype(auto) v2 = getv(ix, iy + 1, iz, p2);
      if (isnan(v2)) {
        return;
      }
      decltype(auto) v4 = getv(ix, iy, iz + 1, p4);
      if (isnan(v4)) {
        return;
      }
      decltype(auto) v7 = getv(ix + 1, iy + 1, iz + 1, p7);
      if (isnan(v7)) {
        return;
      }
      decltype(auto) w1 = getw(ix + 1, iy, iz, p1);
      if (isnan(w1)) {
        return;
      }
      decltype(auto) w2 = getw(ix, iy + 1, iz, p2);
      if (isnan(w2)) {
        return;
      }
      decltype(auto) w4 = getw(ix, iy, iz + 1, p4);
      if (isnan(w4)) {
        return;
      }
      decltype(auto) w7 = getw(ix + 1, iy + 1, iz + 1, p7);
      if (isnan(w7)) {
        return;
      }
      inner_faces(ix, iy)[0] =  // 124
          detail::pv_on_tri(p1, v1, w1, p4, v4, w4, p2, v2, w2,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy)[1] =  // 127
          detail::pv_on_tri(p1, v1, w1, p7, v7, w7, p2, v2, w2,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy)[2] =  // 147
          detail::pv_on_tri(p1, v1, w1, p4, v4, w4, p7, v7, w7,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy)[3] =  // 247
          detail::pv_on_tri(p2, v2, w2, p7, v7, w7, p4, v4, w4,
                            std::forward<Preds>(preds)...);
    }
  };
  auto precompute_candidates = [&](auto const ix, auto const iy) {
    auto const cube_idx = (ix + iy * g.template size<0>() +
                           iz * g.template size<0>() * g.template size<1>()) *
                          num_tets_per_cube;
    auto const thread_id = omp_get_thread_num();
    if (turned(ix, iy, iz)) {
      // 0236
      if (check_pv_in_tet(x_faces(ix, iy)[1],      // 026
                          y_faces(iy + 1, ix)[0],  // 236
                          z_faces(0, ix, iy)[1],   // 023
                          inner_faces(ix, iy)[1])  // 036
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx);
      }

      // 0135
      if (check_pv_in_tet(x_faces(ix + 1, iy)[0],  // 135
                          y_faces(ix, iy)[0],      // 015
                          z_faces(0, ix, iy)[0],   // 013
                          inner_faces(ix, iy)[0])  // 035
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx + 1);
      }
      // 3567
      if (check_pv_in_tet(x_faces(ix + 1, iy)[1],  // 357
                          y_faces(ix, iy + 1)[1],  // 367
                          z_faces(1, ix, iy)[1],   // 567
                          inner_faces(ix, iy)[2])  // 356
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx + 2);
      }
      // 0456
      if (check_pv_in_tet(x_faces(ix, iy)[0],      // 046
                          y_faces(ix, iy)[1],      // 045
                          z_faces(1, ix, iy)[0],   // 456
                          inner_faces(ix, iy)[3])  // 056
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx + 3);
      }
      // 0356
      if (check_pv_in_tet(inner_faces(ix, iy)[0],  // 035
                                    inner_faces(ix, iy)[1],  // 036
                                    inner_faces(ix, iy)[2],  // 356
                                    inner_faces(ix, iy)[3])  // 056
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx + 4);
      }
    } else {
      // 0124
      if (check_pv_in_tet(x_faces(ix, iy)[0],      // 024
                                    y_faces(ix, iy)[0],      // 014
                                    z_faces(0, ix, iy)[0],   // 012
                                    inner_faces(ix, iy)[0])  // 124
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx);
      }
      // 1457
      if (check_pv_in_tet(x_faces(ix + 1, iy)[0],  // 157
                                    y_faces(ix, iy)[1],      // 145
                                    z_faces(1, ix, iy)[0],   // 457
                                    inner_faces(ix, iy)[2])  // 147
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx + 1);
      }
      // 2467
      if (check_pv_in_tet(x_faces(ix, iy)[1],      // 246
                                    y_faces(ix, iy + 1)[1],  // 267
                                    z_faces(1, ix, iy)[1],   // 467
                                    inner_faces(ix, iy)[3])  // 247
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx + 2);
      }
      // 1237
      if (check_pv_in_tet(x_faces(ix + 1, iy)[1],  // 137
                                    y_faces(ix, iy + 1)[0],  // 237
                                    z_faces(0, ix, iy)[1],   // 123
                                    inner_faces(ix, iy)[1])  // 127
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx + 3);
      }
      // 1247
      if (check_pv_in_tet(inner_faces(ix, iy)[0],  // 124
                                    inner_faces(ix, iy)[1],  // 127
                                    inner_faces(ix, iy)[2],  // 147
                                    inner_faces(ix, iy)[3])  // 247
      ) {
        auto l = std::lock_guard{mutex};
        std::cout << ix << ", " << iy << ", " << iz << '\n';
        tets_with_pvs.push_back(cube_idx + 4);
      }
    }
  };
  for (; iz < g.template size<2>() - 1; ++iz) {
    tatooine::for_loop(update_x_faces, policy, g.template size<0>(),
                       g.template size<1>() - 1);
    tatooine::for_loop(update_y_faces, policy, g.template size<0>() - 1,
                       g.template size<1>());
    tatooine::for_loop(update_z_faces, policy, g.template size<0>() - 1,
                       g.template size<1>() - 1);
    tatooine::for_loop(update_inner_faces, policy, g.template size<0>() - 1,
                       g.template size<1>() - 1);
    tatooine::for_loop(precompute_candidates, policy, g.template size<0>() - 1,
                       g.template size<1>() - 1);
    if (iz < g.template size<2>() - 2) {
      move_zs();
    }
  }


  std::cout << "==========================================\n";
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
  auto num_threads = std::size_t{};
#pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
  }
  auto lines = std::vector<aligned<std::vector<line<Real, 3>>>>(num_threads);
#else
  auto lines = std::vector<line<Real, 3>>{};
#endif


  for(auto const tet_idx : tets_with_pvs) {
    auto const [ix, iy, iz, cube_idx, tet_idx_in_cube] =
        extract_indices(tet_idx);

    extract_pv_lines_from_tet()
  }

#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
  using namespace boost::adaptors;
  auto const s = [](auto const& l) { return l->size(); };
  auto       l = std::vector<line<Real, 3>>{};
  l.reserve(boost::accumulate(lines | transformed(s), std::size_t(0)));
  for (auto& li : lines) {
    std::move(begin(*li), end(*li), std::back_inserter(l));
  }
  return l;
#else
  return lines;
#endif
}
//==============================================================================
}  // namespace detail
//==============================================================================
/// This is an implementation of \cite Peikert1999TheV.
#ifdef __cpp_concepts
template <typename VReal, typename WReal, typename XDomain, typename YDomain,
          typename ZDomain, arithmetic TReal,
          invocable<vec<common_type<VReal, WReal>, 3>>... Preds>
#else
template <typename VReal, typename WReal, typename TReal, typename XDomain,
          typename YDomain, typename ZDomain, typename... Preds,
          enable_if<is_arithmetic<TReal> &&
                    (is_invocable<Preds, vec<common_type<VReal, WReal>, 3>> &&
                     ...)> = true>
#endif
auto parallel_vectors(polymorphic::vectorfield<VReal, 3> const&          vf,
                      polymorphic::vectorfield<WReal, 3> const&          wf,
                      rectilinear_grid<XDomain, YDomain, ZDomain> const& g,
                      TReal const t, Preds&&... preds)
    -> std::vector<line<common_type<VReal, WReal>, 3>> {
  return detail::calc_parallel_vectors<common_type<VReal, WReal>>(
      // get v data by evaluating V field
      [&vf, t](auto const /*ix*/, auto const /*iy*/, auto const /*iz*/,
               auto const& p) { return vf(p, t); },
      // get w data by evaluating W field
      [&wf, t](auto const /*ix*/, auto const /*iy*/, auto const /*iz*/,
               auto const& p) { return wf(p, t); },
      g, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
/// This is an implementation of \cite Peikert1999TheV.
#ifdef __cpp_concepts
template <typename V, typename W, typename VReal, typename WReal,
          typename XDomain, typename YDomain, typename ZDomain,
          arithmetic TReal,
          invocable<vec<common_type<VReal, WReal>, 3>>... Preds>
#else
template <
    typename V, typename W, typename VReal, typename WReal, typename TReal,
    typename XDomain, typename YDomain, typename ZDomain, typename... Preds,
    enable_if<is_arithmetic<TReal> &&
              (is_invocable<Preds, vec<common_type<VReal, WReal>, 3>> && ...)> =
        true>
#endif
auto parallel_vectors(vectorfield<V, VReal, 3> const&                    vf,
                      vectorfield<W, WReal, 3> const&                    wf,
                      rectilinear_grid<XDomain, YDomain, ZDomain> const& g,
                      TReal const t, Preds&&... preds)
    -> std::vector<line<common_type<VReal, WReal>, 3>> {
  return detail::calc_parallel_vectors<common_type<VReal, WReal>>(
      // get v data by evaluating V field
      [&vf, t](auto const /*ix*/, auto const /*iy*/, auto const /*iz*/,
               auto const& p) { return vf(p, t); },
      // get w data by evaluating W field
      [&wf, t](auto const /*ix*/, auto const /*iy*/, auto const /*iz*/,
               auto const& p) { return wf(p, t); },
      g, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
/// This is an implementation of \cite Peikert1999TheV.
#ifdef __cpp_concepts
template <typename V, typename W, typename VReal, typename WReal,
          typename XDomain, typename YDomain, typename ZDomain,
          invocable<vec<common_type<VReal, WReal>, 3>>... Preds>
#else
template <
    typename V, typename W, typename VReal, typename WReal, typename XDomain,
    typename YDomain, typename ZDomain, typename... Preds,
    enable_if<is_invocable<Preds, vec<common_type<VReal, WReal>, 3>>...> = true>
#endif
auto parallel_vectors(vectorfield<V, VReal, 3> const&                    v,
                      vectorfield<W, WReal, 3> const&                    w,
                      rectilinear_grid<XDomain, YDomain, ZDomain> const& g,
                      Preds&&... preds)
    -> std::vector<line<common_type<VReal, WReal>, 3>> {
  return parallel_vectors(v, w, g, 0, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
/// This is an implementation of \cite Peikert1999TheV.
#ifdef __cpp_concepts
template <typename V, typename W, typename VReal, typename WReal,
          typename XReal, typename YReal, typename ZReal, arithmetic TReal,
          invocable<vec<common_type<VReal, WReal>, 3>>... Preds>
#else
template <
    typename V, typename W, typename VReal, typename WReal, typename XReal,
    typename YReal, typename ZReal, typename TReal, typename... Preds,
    enable_if<is_arithmetic<TReal>,
              (is_invocable<Preds, vec<common_type<VReal, WReal>, 3>> && ...)> =
        true>
#endif
auto parallel_vectors(vectorfield<V, VReal, 3> const& v,
                      vectorfield<W, WReal, 3> const& w,
                      linspace<XReal> const& x, linspace<YReal> const& y,
                      linspace<ZReal> const& z, TReal const t, Preds&&... preds)
    -> std::vector<line<common_type<VReal, WReal>, 3>> {
  return parallel_vectors(v, w, rectilinear_grid{x, y, z}, t,
                          std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
/// This is an implementation of \cite Peikert1999TheV.
#ifdef __cpp_concepts
template <typename V, typename W, typename VReal, typename WReal,
          typename XReal, typename YReal, typename ZReal,
          invocable<vec<common_type<VReal, WReal>, 3>>... Preds>
#else
template <typename V, typename W, typename VReal, typename WReal,
          typename XReal, typename YReal, typename ZReal, typename... Preds,
          enable_if<(is_invocable<Preds, vec<common_type<VReal, WReal>, 3>> &&
                     ...)> = true>
#endif
auto parallel_vectors(vectorfield<V, VReal, 3> const& v,
                      vectorfield<W, WReal, 3> const& w,
                      linspace<XReal> const& x, linspace<YReal> const& y,
                      linspace<ZReal> const& z, Preds&&... preds)
    -> std::vector<line<common_type<VReal, WReal>, 3>> {
  return parallel_vectors(v, w, rectilinear_grid{x, y, z}, 0,
                          std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
/// This is an implementation of \cite Peikert1999TheV.
#ifdef __cpp_concepts
template <typename VReal, typename VIndexing, typename WReal,
          typename WIndexing, typename AABBReal,
          invocable<vec<common_type<VReal, WReal>, 3>>... Preds>
#else
template <
    typename VReal, typename VIndexing, typename WReal, typename WIndexing,
    typename AABBReal, typename... Preds,
    enable_if<is_invocable<Preds, vec<common_type<VReal, WReal>, 3>>...> = true>
#endif
auto parallel_vectors(
    dynamic_multidim_array<vec<VReal, 3>, VIndexing> const& vf,
    dynamic_multidim_array<vec<WReal, 3>, WIndexing> const& wf,
    axis_aligned_bounding_box<AABBReal, 3> const& bb, Preds&&... preds)
    -> std::vector<line<common_type<VReal, WReal>, 3>> {
  assert(vf.num_dimensions() == 3);
  assert(wf.num_dimensions() == 3);
  assert(vf.size(0) == wf.size(0));
  assert(vf.size(1) == wf.size(1));
  assert(vf.size(2) == wf.size(2));

  return detail::calc_parallel_vectors<common_type<VReal, WReal>>(
      [&vf](auto ix, auto iy, auto iz, auto const& /*p*/) -> auto const& {
        return vf(ix, iy, iz);
      },
      [&wf](auto ix, auto iy, auto iz, auto const& /*p*/) -> auto const& {
        return wf(ix, iy, iz);
      },
      rectilinear_grid{linspace{bb.min(0), bb.max(0), vf.size(0)},
                       linspace{bb.min(1), bb.max(1), vf.size(1)},
                       linspace{bb.min(2), bb.max(2), vf.size(2)}},
      std::forward<Preds>(preds)...);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
