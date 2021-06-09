#ifndef TATOOINE_PARALLEL_VECTORS_H
#define TATOOINE_PARALLEL_VECTORS_H
//==============================================================================
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
#include <mutex>
#endif
#include <tatooine/field.h>
#include <tatooine/for_loop.h>
#include <tatooine/grid.h>
#include <tatooine/line.h>
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
     Real const       eps = 1e-5;
     for (unsigned int i = 1; i < barycentric_coords.size(); i++) {
      for (unsigned int j = 0; j < i; j++) {
        if (!approx_equal(barycentric_coords[i], barycentric_coords[j],
                          eps)) {
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
                      std::vector<line<Real, 3>> & lines) {
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
auto turned(size_t const ix, size_t const iy, size_t const iz) -> bool {
  bool const xodd = ix % 2 == 0;
  bool const yodd = iy % 2 == 0;
  bool const zodd = iz % 2 == 0;

  bool turned = xodd;
  if (yodd) {
    turned = !turned;
  }
  if (zodd) {
    turned = !turned;
  }
  return turned;
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
                           grid<XDomain, YDomain, ZDomain> const& g,
                           Preds&&... preds) -> std::vector<line<Real, 3>> {
  std::vector<line<Real, 3>> line_segments;

#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
  std::mutex mutex;

  auto check_cell = [&line_segments, &getv, &getw, &g, &mutex, &preds...](
                        size_t ix, size_t iy, size_t iz) {
#else
  auto check_cell = [&line_segments, &getv, &getw, &g, &preds...](
                        size_t ix, size_t iy, size_t iz) {
#endif
    using boost::copy;
    auto gv = g.vertices();
    std::array p{
        gv(ix, iy, iz),         gv(ix + 1, iy, iz),         gv(ix, iy + 1, iz),
        gv(ix + 1, iy + 1, iz), gv(ix, iy, iz + 1),         gv(ix + 1, iy, iz + 1),
        gv(ix, iy + 1, iz + 1), gv(ix + 1, iy + 1, iz + 1),
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
    decltype(auto) v4 = getv(ix, iy, iz + 1, p[4]);
    if (isnan(v4)) {
      return;
    }
    decltype(auto) v5 = getv(ix + 1, iy, iz + 1, p[5]);
    if (isnan(v5)) {
      return;
    }
    decltype(auto) v6 = getv(ix, iy + 1, iz + 1, p[6]);
    if (isnan(v6)) {
      return;
    }
    decltype(auto) v7 = getv(ix + 1, iy + 1, iz + 1, p[7]);
    if (isnan(v7)) {
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
    decltype(auto) w4 = getw(ix, iy, iz + 1, p[4]);
    if (isnan(w4)) {
      return;
    }
    decltype(auto) w5 = getw(ix + 1, iy, iz + 1, p[5]);
    if (isnan(w5)) {
      return;
    }
    decltype(auto) w6 = getw(ix, iy + 1, iz + 1, p[6]);
    if (isnan(w6)) {
      return;
    }
    decltype(auto) w7 = getw(ix + 1, iy + 1, iz + 1, p[7]);
    if (isnan(w7)) {
      return;
    }
    if (turned(ix, iy, iz)) {
      // check if there are parallel vectors on any of the tet's triangles
      auto pv012 = detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[2], v2, w2,
                                     std::forward<Preds>(preds)...);
      auto pv014 = detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[4], v4, w4,
                                     std::forward<Preds>(preds)...);
      auto pv024 = detail::pv_on_tri(p[0], v0, w0, p[2], v2, w2, p[4], v4, w4,
                                     std::forward<Preds>(preds)...);
      auto pv124 = detail::pv_on_tri(p[1], v1, w1, p[2], v2, w2, p[4], v4, w4,
                                     std::forward<Preds>(preds)...);
      auto pv246 = detail::pv_on_tri(p[2], v2, w2, p[4], v4, w4, p[6], v6, w6,
                                     std::forward<Preds>(preds)...);
      auto pv247 = detail::pv_on_tri(p[2], v2, w2, p[4], v4, w4, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv267 = detail::pv_on_tri(p[2], v2, w2, p[6], v6, w6, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv467 = detail::pv_on_tri(p[4], v4, w4, p[6], v6, w6, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv145 = detail::pv_on_tri(p[1], v1, w1, p[4], v4, w4, p[5], v5, w5,
                                     std::forward<Preds>(preds)...);
      auto pv147 = detail::pv_on_tri(p[1], v1, w1, p[4], v4, w4, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv157 = detail::pv_on_tri(p[1], v1, w1, p[5], v5, w5, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv457 = detail::pv_on_tri(p[4], v4, w4, p[5], v5, w5, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv123 = detail::pv_on_tri(p[1], v1, w1, p[2], v2, w2, p[3], v3, w3,
                                     std::forward<Preds>(preds)...);
      auto pv127 = detail::pv_on_tri(p[1], v1, w1, p[2], v2, w2, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv137 = detail::pv_on_tri(p[1], v1, w1, p[3], v3, w3, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv237 = detail::pv_on_tri(p[2], v2, w2, p[3], v3, w3, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);

#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
      {
        std::lock_guard lock{mutex};
#endif
        // check the tets themselves
        // 0124
        detail::check_tet(pv012, pv014, pv024, pv124, line_segments);
        // 2467
        detail::check_tet(pv246, pv247, pv267, pv467, line_segments);
        // 1457
        detail::check_tet(pv145, pv147, pv157, pv457, line_segments);
        // 1237
        detail::check_tet(pv123, pv127, pv137, pv237, line_segments);
        // 1247
        detail::check_tet(pv124, pv127, pv147, pv247, line_segments);
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
      }
#endif
    } else {
      // check if there are parallel vectors on any of the tets triangles
      auto pv023 = detail::pv_on_tri(p[0], v0, w0, p[2], v2, w2, p[3], v3, w3,
                                     std::forward<Preds>(preds)...);
      auto pv026 = detail::pv_on_tri(p[0], v0, w0, p[2], v2, w2, p[6], v6, w6,
                                     std::forward<Preds>(preds)...);
      auto pv036 = detail::pv_on_tri(p[0], v0, w0, p[3], v3, w3, p[6], v6, w6,
                                     std::forward<Preds>(preds)...);
      auto pv236 = detail::pv_on_tri(p[2], v2, w2, p[3], v3, w3, p[6], v6, w6,
                                     std::forward<Preds>(preds)...);
      auto pv013 = detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[3], v3, w3,
                                     std::forward<Preds>(preds)...);
      auto pv015 = detail::pv_on_tri(p[0], v0, w0, p[1], v1, w1, p[5], v5, w5,
                                     std::forward<Preds>(preds)...);
      auto pv035 = detail::pv_on_tri(p[0], v0, w0, p[3], v3, w3, p[5], v5, w5,
                                     std::forward<Preds>(preds)...);
      auto pv135 = detail::pv_on_tri(p[1], v1, w1, p[3], v3, w3, p[5], v5, w5,
                                     std::forward<Preds>(preds)...);
      auto pv356 = detail::pv_on_tri(p[3], v3, w3, p[5], v5, w5, p[6], v6, w6,
                                     std::forward<Preds>(preds)...);
      auto pv357 = detail::pv_on_tri(p[3], v3, w3, p[5], v5, w5, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv367 = detail::pv_on_tri(p[3], v3, w3, p[6], v6, w6, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv567 = detail::pv_on_tri(p[5], v5, w5, p[6], v6, w6, p[7], v7, w7,
                                     std::forward<Preds>(preds)...);
      auto pv045 = detail::pv_on_tri(p[0], v0, w0, p[4], v4, w4, p[5], v5, w5,
                                     std::forward<Preds>(preds)...);
      auto pv046 = detail::pv_on_tri(p[0], v0, w0, p[4], v4, w4, p[6], v6, w6,
                                     std::forward<Preds>(preds)...);
      auto pv056 = detail::pv_on_tri(p[0], v0, w0, p[5], v5, w5, p[6], v6, w6,
                                     std::forward<Preds>(preds)...);
      auto pv456 = detail::pv_on_tri(p[4], v4, w4, p[5], v5, w5, p[6], v6, w6,
                                     std::forward<Preds>(preds)...);
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
      {
        std::lock_guard lock{mutex};
#endif
        // check the tets themselves
        // 0236
        detail::check_tet(pv023, pv026, pv036, pv236, line_segments);
        // 0135
        detail::check_tet(pv013, pv015, pv035, pv135, line_segments);
        // 3567
        detail::check_tet(pv356, pv357, pv367, pv567, line_segments);
        // 0456
        detail::check_tet(pv045, pv046, pv056, pv456, line_segments);
        // 0356
        detail::check_tet(pv035, pv036, pv056, pv356, line_segments);
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
      }
#endif
    }
  };
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
#pragma omp parallel for collapse(3)
#endif
  for (size_t iz = 0; iz < g.size(2) - 1; ++iz) {
    for (size_t iy = 0; iy < g.size(1) - 1; ++iy) {
      for (size_t ix = 0; ix < g.size(0) - 1; ++ix) {
        check_cell(ix, iy, iz);
      }
    }
  }
  // tatooine::for_loop(check_cell, tag::parallel,g.template size<0>() - 1,
  //                            g.template size<1>() - 1,
  //                            g.template size<2>() - 1);
  return line_segments;
  //return merge(line_segments);
}
//==============================================================================
}  // namespace detail
//==============================================================================
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
auto parallel_vectors(polymorphic::vectorfield<VReal, 3> const&   vf,
                      polymorphic::vectorfield<WReal, 3> const&   wf,
                      grid<XDomain, YDomain, ZDomain> const& g, TReal const t,
                      Preds&&... preds)
    -> std::vector<line<common_type<VReal, WReal>, 3>> {
  return detail::calc_parallel_vectors<common_type<VReal, WReal>>(
      // get v data by evaluating V field
      [&vf, t](auto /*ix*/, auto /*iy*/, auto /*iz*/, auto const& p) {
        if (vf.in_domain(p, t)) {
          return vf(p, t);
        }
        return typename std::decay_t<decltype(vf)>::tensor_t{
            tag::fill{VReal(0) / VReal(0)}};
      },
      // get w data by evaluating W field
      [&wf, t](auto /*ix*/, auto /*iy*/, auto /*iz*/, auto const& p) {
        if (wf.in_domain(p, t)) {
          return wf(p, t);
        }
        return typename std::decay_t<decltype(wf)>::tensor_t{
            tag::fill{WReal(0) / WReal(0)}};
      },
      g, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
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
auto parallel_vectors(vectorfield<V, VReal, 3> const&        vf,
                      vectorfield<W, WReal, 3> const&        wf,
                      grid<XDomain, YDomain, ZDomain> const& g, TReal const t,
                      Preds&&... preds)
    -> std::vector<line<common_type<VReal, WReal>, 3>> {
  return detail::calc_parallel_vectors<common_type<VReal, WReal>>(
      // get v data by evaluating V field
      [&vf, t](auto /*ix*/, auto /*iy*/, auto /*iz*/, auto const& p) {
        if (vf.in_domain(p, t)) {
          return vf(p, t);
        }
        return typename std::decay_t<decltype(vf)>::tensor_t{
            tag::fill{VReal(0) / VReal(0)}};
      },
      // get w data by evaluating W field
      [&wf, t](auto /*ix*/, auto /*iy*/, auto /*iz*/, auto const& p) {
        if (wf.in_domain(p, t)) {
          return wf(p, t);
        }
        return typename std::decay_t<decltype(wf)>::tensor_t{
            tag::fill{WReal(0) / WReal(0)}};
      },
      g, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
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
auto parallel_vectors(vectorfield<V, VReal, 3> const&           v,
                      vectorfield<W, WReal, 3> const&           w,
                      grid<XDomain, YDomain, ZDomain> const& g,
                      Preds&&... preds)
    -> std::vector<line<common_type<VReal, WReal>, 3>> {
  return parallel_vectors(v, w, g, 0, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
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
  return parallel_vectors(v, w, grid{x, y, z}, t,
                          std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
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
  return parallel_vectors(v, w, grid{x, y, z}, 0,
                          std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
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
      grid{linspace{bb.min(0), bb.max(0), vf.size(0)},
           linspace{bb.min(1), bb.max(1), vf.size(1)},
           linspace{bb.min(2), bb.max(2), vf.size(2)}},
      std::forward<Preds>(preds)...);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
