#ifndef TATOOINE_PARALLEL_VECTORS_H
#define TATOOINE_PARALLEL_VECTORS_H
//==============================================================================
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
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
#include <optional>
#include <tuple>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail {
//==============================================================================
/// \return Position where v and w are parallel otherwise nothing.
#ifdef __cpp_concepts
template <typename Real, invocable<vec<Real, 3>>... Preds>
#else
template <typename Real, typename... Preds,
          enable_if<is_invocable<Preds, vec<Real, 3>>...> = true>
#endif
auto pv_on_tri(vec<Real, 3> const& p0, vec<Real, 3> const& v0,
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

  std::vector<vec<Real, 3>> barycentric_coords;
  for (size_t i = 0; i < 3; ++i) {
    if ((ieig(0, i) == 0 && ieig(1, i) == 0 && ieig(2, i) == 0) &&
        ((reig(0, i) <= 0 && reig(1, i) <= 0 && reig(2, i) <= 0) ||
         (reig(0, i) >= 0 && reig(1, i) >= 0 && reig(2, i) >= 0))) {
      vec<Real, 3> const bc = real(eigvecs.col(i)) / sum(real(eigvecs.col(i)));
      barycentric_coords.push_back(bc);
    }
  }

  if (barycentric_coords.empty()) {
    return {};

  } else if (barycentric_coords.size() == 1) {
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
    for (unsigned int i = 1; i < barycentric_coords.size(); i++) {
      for (unsigned int j = 0; j < i; j++) {
        if (!approx_equal(barycentric_coords[i], barycentric_coords[j], eps)) {
          return {};
        }
      }
    }
    auto const pos = barycentric_coords.front()(0) * p0 +
                     barycentric_coords.front()(1) * p1 +
                     barycentric_coords.front()(2) * p2;
    return pos;

    // return {};
  }
}

//----------------------------------------------------------------------------
template <typename Real>
static auto check_tet(std::optional<vec<Real, 3>> const& tri0,
                      std::optional<vec<Real, 3>> const& tri1,
                      std::optional<vec<Real, 3>> const& tri2,
                      std::optional<vec<Real, 3>> const& tri3,
                      std::vector<line<Real, 3>>& lines, std::mutex& mutex) {
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
//----------------------------------------------------------------------------
template <typename Real>
static auto check_tet(std::optional<vec<Real, 3>> const& tri0,
                      std::optional<vec<Real, 3>> const& tri1,
                      std::optional<vec<Real, 3>> const& tri2,
                      std::optional<vec<Real, 3>> const& tri3,
                      std::vector<line<Real, 3>>&        lines) {
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
    auto& l = lines.emplace_back();
    l.push_back(*(*tris[0]));
    l.push_back(*(*tris[1]));
  } else if (tris.size() == 3) {
    // std::cerr << "3 points\n";
  } else if (tris.size() == 4) {
    // std::cerr << "several solutions\n";
  }
}
//------------------------------------------------------------------------------
auto constexpr turned(size_t const ix, size_t const iy, size_t const iz)
    -> bool {
  auto const xodd = ix % 2 == 0 ? 1 : -1;
  auto const yodd = iy % 2 == 0 ? 1 : -1;
  auto const zodd = iz % 2 == 0 ? 1 : -1;
  return xodd * yodd * zodd < 0;
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
  using boost::copy;

  // turned inner tet order:
  //   [0]: 046 / 157 | [1]: 026 / 137
  // non-turned inner tet order:
  //   [0]: 024 / 135 | [1]: 246 / 357
  auto x_faces =
      dynamic_multidim_array<std::array<std::optional<vec<Real, 3>>, 2>>{
          g.template size<0>(), g.template size<1>() - 1,
          g.template size<2>() - 1};
  auto check_faces_x = [&g, &getv, &getw, &x_faces, &preds...](
                           size_t const iy, size_t const iz, size_t const ix) {
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
      x_faces(ix, iy, iz)[0] = // 046
          detail::pv_on_tri(p[0], v0, w0, p[1], v4, w4, p[3], v6, w6,
                            std::forward<Preds>(preds)...);
      x_faces(ix, iy, iz)[1] = // 026
          detail::pv_on_tri(p[0], v0, w0, p[3], v6, w6, p[2], v2, w2,
                            std::forward<Preds>(preds)...);
    } else {
      x_faces(ix, iy, iz)[0] = // 024
          detail::pv_on_tri(p[0], v0, w0, p[1], v4, w4, p[2], v2, w2,
                            std::forward<Preds>(preds)...);
      x_faces(ix, iy, iz)[1] =  // 246
          detail::pv_on_tri(p[1], v4, w4, p[3], v6, w6, p[2], v2, w2,
                            std::forward<Preds>(preds)...);
    }
  };
  // turned inner tet order:
  //   [0]: 015 / 237 | [1]: 045 / 267
  // non-turned inner tet order:
  //   [0]: 014 / 236 | [1]: 145 / 367
  auto y_faces =
      dynamic_multidim_array<std::array<std::optional<vec<Real, 3>>, 2>>{
          g.template size<1>(), g.template size<0>() - 1,
          g.template size<2>() - 1};
  auto check_faces_y = [&g, &getv, &getw, &y_faces, &preds...](
                           size_t const ix, size_t const iz, size_t const iy) {
    auto const gv = g.vertices();
    auto const p  = std::array{
        gv(ix, iy, iz),         // 0
        gv(ix + 1, iy, iz),     // 1
        gv(ix, iy, iz + 1),     // 4
        gv(ix + 1, iy, iz + 1)  // 5
    };

    decltype(auto) v0 = getv(ix, iy, iz, p[0]);
    if (isnan(v0)) {
      return;
    }
    decltype(auto) v1 = getv(ix + 1, iy, iz, p[1]);
    if (isnan(v1)) {
      return;
    }
    decltype(auto) v4 = getv(ix, iy, iz + 1, p[2]);
    if (isnan(v4)) {
      return;
    }
    decltype(auto) v5 = getv(ix + 1, iy, iz + 1, p[3]);
    if (isnan(v5)) {
      return;
    }
    decltype(auto) w0 = getw(ix, iy, iz, p[0]);
    if (isnan(w0)) {
      return;
    }
    decltype(auto) w1 = getw(ix + 1, iy, iz, p[1]);
    if (isnan(w1)) {
      return;
    }
    decltype(auto) w4 = getw(ix, iy, iz + 1, p[2]);
    if (isnan(w4)) {
      return;
    }
    decltype(auto) w5 = getw(ix + 1, iy, iz + 1, p[3]);
    if (isnan(w5)) {
      return;
    }
    if (turned(ix, iy, iz)) {
      y_faces(iy, ix, iz)[0] = // 015
          detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[3], v5, w5,
                            std::forward<Preds>(preds)...);
      y_faces(iy, ix, iz)[1] = // 045
          detail::pv_on_tri(p[0], v0, w0, p[3], v5, w5, p[2], v4, w4,
                            std::forward<Preds>(preds)...);
    } else {
      y_faces(iy, ix, iz)[0] = // 014
          detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[2], v4, w4,
                            std::forward<Preds>(preds)...);
      y_faces(iy, ix, iz)[1] = // 145
          detail::pv_on_tri(p[1], v1, w1, p[3], v5, w5, p[2], v4, w4,
                            std::forward<Preds>(preds)...);
    }
  };
  // turned inner tet order:
  //   [0]: 013 / 457 | [1]: 023 / 467
  // non-turned inner tet order:
  //   [0]: 012 / 456 | [1]: 123 / 567
  auto z_faces =
      dynamic_multidim_array<std::array<std::optional<vec<Real, 3>>, 2>>{
          g.template size<2>(), g.template size<0>() - 1,
          g.template size<1>() - 1};
  auto check_faces_z = [&g, &getv, &getw, &z_faces, &preds...](
                           size_t const ix, size_t const iy, size_t const iz) {
    auto const gv = g.vertices();
    auto const p  = std::array{
        gv(ix, iy, iz),         // 0
        gv(ix + 1, iy, iz),     // 1
        gv(ix, iy + 1, iz),     // 2
        gv(ix + 1, iy + 1, iz)  // 3
    };

    decltype(auto) v0 = getv(ix, iy, iz, p[0]);
    if (isnan(v0)) {
      return;
    }
    decltype(auto) v1 = getv(ix + 1, iy, iz, p[1]);
    if (isnan(v1)) {
      return;
    }
    decltype(auto) v2 = getv(ix, iy + 1, iz, p[2]);
    if (isnan(v2)) {
      return;
    }
    decltype(auto) v3 = getv(ix + 1, iy + 1, iz, p[3]);
    if (isnan(v3)) {
      return;
    }
    decltype(auto) w0 = getw(ix, iy, iz, p[0]);
    if (isnan(w0)) {
      return;
    }
    decltype(auto) w1 = getw(ix + 1, iy, iz, p[1]);
    if (isnan(w1)) {
      return;
    }
    decltype(auto) w2 = getw(ix, iy + 1, iz, p[2]);
    if (isnan(w2)) {
      return;
    }
    decltype(auto) w3 = getw(ix + 1, iy + 1, iz, p[3]);
    if (isnan(w3)) {
      return;
    }
    if (turned(ix, iy, iz)) {
      z_faces(iz, ix, iy)[0] = // 013
          detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[3], v3, w3,
                            std::forward<Preds>(preds)...);
      z_faces(iz, ix, iy)[1] = // 023
          detail::pv_on_tri(p[0], v0, w0, p[3], v3, w3, p[2], v2, w2,
                            std::forward<Preds>(preds)...);
    } else {
      z_faces(iz, ix, iy)[0] = // 012
          detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[2], v2, w2,
                            std::forward<Preds>(preds)...);
      z_faces(iz, ix, iy)[1] = // 123
          detail::pv_on_tri(p[1], v1, w1, p[3], v3, w3, p[2], v2, w2,
                            std::forward<Preds>(preds)...);
    }
  };
  // turned inner tet order:
  //   [0]: 035 | [1]: 036 | [2]: 356 | [3]: 056
  // non-turned inner tet order:
  //   [0]: 124 | [1]: 127 | [2]: 147 | [3]: 247
  auto inner_faces =
      dynamic_multidim_array<std::array<std::optional<vec<Real, 3>>, 4>>{
          g.template size<0>() - 1, g.template size<1>() - 1,
          g.template size<2>() - 1};
  auto check_inner_faces = [&g, &getv, &getw, &inner_faces, &preds...](
                               size_t ix, size_t iy, size_t iz) {
    auto const gv = g.vertices();
    if (turned(ix, iy, iz)) {
      auto const p = std::array{
          gv(ix, iy, iz),          // 0
          gv(ix + 1, iy + 1, iz),  // 3
          gv(ix + 1, iy, iz + 1),  // 5
          gv(ix, iy + 1, iz + 1)   // 6
      };

      decltype(auto) v0 = getv(ix, iy, iz, p[0]);
      if (isnan(v0)) {
        return;
      }
      decltype(auto) v3 = getv(ix + 1, iy + 1, iz, p[1]);
      if (isnan(v3)) {
        return;
      }
      decltype(auto) v5 = getv(ix + 1, iy, iz + 1, p[2]);
      if (isnan(v5)) {
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
      decltype(auto) w3 = getw(ix + 1, iy + 1, iz, p[1]);
      if (isnan(w3)) {
        return;
      }
      decltype(auto) w5 = getw(ix + 1, iy, iz + 1, p[2]);
      if (isnan(w5)) {
        return;
      }
      decltype(auto) w6 = getw(ix, iy + 1, iz + 1, p[3]);
      if (isnan(w6)) {
        return;
      }
      inner_faces(ix, iy, iz)[0] = // 035
          detail::pv_on_tri(p[0], v0, w0, p[2], v5, w5, p[1], v3, w3,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy, iz)[1] = // 036
          detail::pv_on_tri(p[0], v0, w0, p[1], v3, w3, p[3], v6, w6,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy, iz)[2] = // 356
          detail::pv_on_tri(p[1], v3, w3, p[2], v5, w5, p[3], v6, w6,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy, iz)[3] = // 056
          detail::pv_on_tri(p[0], v0, w0, p[3], v6, w6, p[2], v5, w5,
                            std::forward<Preds>(preds)...);
    } else {
      auto const p = std::array{
          gv(ix + 1, iy, iz),         // 1
          gv(ix, iy + 1, iz),         // 2
          gv(ix, iy, iz + 1),         // 4
          gv(ix + 1, iy + 1, iz + 1)  // 7
      };

      decltype(auto) v1 = getv(ix + 1, iy, iz, p[0]);
      if (isnan(v1)) {
        return;
      }
      decltype(auto) v2 = getv(ix, iy + 1, iz, p[1]);
      if (isnan(v2)) {
        return;
      }
      decltype(auto) v4 = getv(ix, iy, iz + 1, p[2]);
      if (isnan(v4)) {
        return;
      }
      decltype(auto) v7 = getv(ix + 1, iy + 1, iz + 1, p[3]);
      if (isnan(v7)) {
        return;
      }
      decltype(auto) w1 = getw(ix + 1, iy, iz, p[0]);
      if (isnan(w1)) {
        return;
      }
      decltype(auto) w2 = getw(ix, iy + 1, iz, p[1]);
      if (isnan(w2)) {
        return;
      }
      decltype(auto) w4 = getw(ix, iy, iz + 1, p[2]);
      if (isnan(w4)) {
        return;
      }
      decltype(auto) w7 = getw(ix + 1, iy + 1, iz + 1, p[3]);
      if (isnan(w7)) {
        return;
      }
      inner_faces(ix, iy, iz)[0] = // 124
          detail::pv_on_tri(p[0], v1, w1, p[2], v4, w4, p[1], v2, w2,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy, iz)[1] = // 127
          detail::pv_on_tri(p[0], v1, w1, p[3], v7, w7, p[1], v2, w2,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy, iz)[2] = // 147
          detail::pv_on_tri(p[0], v1, w1, p[2], v4, w4, p[3], v7, w7,
                            std::forward<Preds>(preds)...);
      inner_faces(ix, iy, iz)[3] = // 247
          detail::pv_on_tri(p[1], v2, w2, p[3], v7, w7, p[2], v4, w4,
                            std::forward<Preds>(preds)...);
    }
  };
  auto lines = std::vector<line<Real, 3>>{};
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
  auto mutex = std::mutex{};
#endif
  auto compute_line_segments = [&](auto const ix, auto const iy,
                                   auto const iz) {
    if (turned(ix, iy, iz)) {
      // 0236
      check_tet(x_faces(ix, iy, iz)[1],      // 026
                y_faces(iy + 1, ix, iz)[0],  // 236
                z_faces(iz, ix, iy)[1],      // 023
                inner_faces(ix, iy, iz)[1],  // 036
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
      // 0135
      check_tet(x_faces(ix + 1, iy, iz)[0],  // 135
                y_faces(iy, ix, iz)[0],      // 015
                z_faces(iz, ix, iy)[0],      // 013
                inner_faces(ix, iy, iz)[0],  // 035
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
      // 3567
      check_tet(x_faces(ix + 1, iy, iz)[1],  // 357
                y_faces(iy + 1, ix, iz)[1],  // 367
                z_faces(iz + 1, ix, iy)[1],  // 567
                inner_faces(ix, iy, iz)[2],  // 356
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
      // 0456
      check_tet(x_faces(ix, iy, iz)[0],      // 046
                y_faces(iy, ix, iz)[1],      // 045
                z_faces(iz + 1, ix, iy)[0],  // 456
                inner_faces(ix, iy, iz)[3],  // 056
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
      // 0356
      check_tet(inner_faces(ix, iy, iz)[0],  // 035
                inner_faces(ix, iy, iz)[1],  // 036
                inner_faces(ix, iy, iz)[2],  // 356
                inner_faces(ix, iy, iz)[3],  // 056
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
    } else {
      // 0124
      check_tet(x_faces(ix, iy, iz)[0],      // 024
                y_faces(iy, ix, iz)[0],      // 014
                z_faces(iz, ix, iy)[0],      // 012
                inner_faces(ix, iy, iz)[0],  // 124
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
      // 1457
      check_tet(x_faces(ix + 1, iy, iz)[0],  // 157
                y_faces(iy, ix, iz)[1],      // 145
                z_faces(iz + 1, ix, iy)[0],  // 457
                inner_faces(ix, iy, iz)[2],  // 147
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
      // 2467
      check_tet(x_faces(ix, iy, iz)[1],      // 246
                y_faces(iy + 1, ix, iz)[1],  // 267
                z_faces(iz + 1, ix, iy)[1],  // 467
                inner_faces(ix, iy, iz)[3],  // 247
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
      // 1237
      check_tet(x_faces(ix + 1, iy, iz)[1],  // 137
                y_faces(iy + 1, ix, iz)[0],  // 237
                z_faces(iz, ix, iy)[1],      // 123
                inner_faces(ix, iy, iz)[1],  // 127
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
      // 1247
      check_tet(inner_faces(ix, iy, iz)[0],  // 124
                inner_faces(ix, iy, iz)[1],  // 127
                inner_faces(ix, iy, iz)[2],  // 147
                inner_faces(ix, iy, iz)[3],  // 247
                lines
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
                ,
                mutex
#endif
      );
    }
  };
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
  auto constexpr policy = execution_policy::parallel;
#else
  auto constexpr policy = execution_policy::sequential;
#endif
  tatooine::for_loop(check_faces_x, policy, g.template size<1>() - 1,
                     g.template size<2>() - 1, g.template size<0>());
  tatooine::for_loop(check_faces_y, policy, g.template size<0>() - 1,
                     g.template size<2>() - 1, g.template size<1>());
  tatooine::for_loop(check_faces_z, policy, g.template size<0>() - 1,
                     g.template size<1>() - 1, g.template size<2>());
  tatooine::for_loop(check_inner_faces, policy, g.template size<0>() - 1,
                     g.template size<1>() - 1, g.template size<2>() - 1);
  tatooine::for_loop(compute_line_segments, policy, g.template size<0>() - 1,
                     g.template size<1>() - 1, g.template size<2>() - 1);
  return lines;
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
