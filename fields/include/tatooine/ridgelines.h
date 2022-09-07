#ifndef TATOOINE_FIELDS_RIDGELINES_H
#define TATOOINE_FIELDS_RIDGELINES_H
//==============================================================================
#include <tatooine/edgeset.h>
#include <tatooine/isolines.h>
#include <tatooine/rectilinear_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// Implementation of \cite Peikert2008ridges without filters.
///
/// working_grid needs to have the same dimensions as data.grid()
template <typename Grid, arithmetic T, bool HasNonConstReference,
          typename DomainX, typename DomainY>
requires(Grid::num_dimensions() == 2)
auto ridgelines(detail::rectilinear_grid::typed_vertex_property_interface<
                    Grid, T, HasNonConstReference> const& data,
                rectilinear_grid<DomainX, DomainY>&       working_grid,
                execution_policy_tag auto const           exec) {
  using real_type    = typename Grid::real_type;
  using edgeset_type = Edgeset2<real_type>;
  auto& g            = working_grid.sample_to_vertex_property(diff<1>(data),
                                                              "[ridgelines]_g", exec);
  auto& c            = working_grid.sample_to_vertex_property(
      [&](integral auto const... is) {
        auto const& g_ = g(is...);
        return vec{-g_(1), g_(0)};
      },
      "[ridgelines]_c", exec);
  auto& hessian = working_grid.sample_to_vertex_property(
      diff<2>(data), "[ridgelines]_H", exec);

  auto& Hg = working_grid.sample_to_vertex_property(
      [&](integral auto const... is) { return hessian(is...) * g(is...); },
      "[ridgelines]_Hg", exec);
  auto& Hc = working_grid.sample_to_vertex_property(
      [&](integral auto const... is) { return hessian(is...) * c(is...); },
      "[ridgelines]_Hc", exec);
  working_grid.sample_to_vertex_property(
      [&](integral auto const... is) {
        auto i       = std::size_t{};
        auto max_abs = -std::numeric_limits<real_type>::max();
        for (std::size_t j = 0; j < 2; ++j) {
          if (auto const a = gcem::abs(g(is...)(j)); a > max_abs) {
            max_abs = a;
            i       = j;
          }
        }
        return Hc(is...)(i) / g(is...)(i);
      },
      "[ridgelines]_lambda_g", exec);
  auto& lambda_c = working_grid.sample_to_vertex_property(
      [&](integral auto const... is) {
        auto i       = std::size_t{};
        auto max_abs = -std::numeric_limits<real_type>::max();
        for (std::size_t j = 0; j < 2; ++j) {
          if (auto const a = gcem::abs(c(is...)(j)); a > max_abs) {
            max_abs = a;
            i       = j;
          }
        }
        return Hc(is...)(i) / c(is...)(i);
      },
      "[ridgelines]_lambda_c", exec);
  auto const compute_d = [&](integral auto const... is) {
    auto const& g_  = g(is...);
    auto const& Hg_ = Hg(is...);
    return g_(0) * Hg_(1) - g_(1) * Hg_(0);
  };
  auto& d = working_grid.sample_to_vertex_property(
      compute_d, "[ridgelines]_det(g|Hg)", exec);

  auto const raw        = isolines(d, 0);
  auto       ridgelines = edgeset_type{};
  auto       lc         = lambda_c.linear_sampler();
  for (auto const e : raw.simplices()) {
    auto [v0, v1] = raw[e];
    if (lc(raw[v0]) <= 0 && lc(raw[v1]) <= 0) {
      auto vr0 = ridgelines.insert_vertex(raw[v0]);
      auto vr1 = ridgelines.insert_vertex(raw[v1]);
      ridgelines.insert_edge(vr0, vr1);
    }
  }
  return ridgelines;
}
//------------------------------------------------------------------------------
/// Implementation of \cite Peikert2008ridges without filters.
template <typename Grid, arithmetic T, bool HasNonConstReference,
          typename DomainX, typename DomainY>
requires(Grid::num_dimensions() == 2)
auto ridgelines(detail::rectilinear_grid::typed_vertex_property_interface<
                    Grid, T, HasNonConstReference> const& data,
                execution_policy_tag auto const           exec) {
  return ridgelines(data, data.grid.copy_without_properties(), exec);
}
//------------------------------------------------------------------------------
/// Implementation of \cite Peikert2008ridges without filters.
///
/// working_grid needs to have the same dimensions as data.grid()
template <typename Grid, arithmetic T, bool HasNonConstReference,
          typename DomainX, typename DomainY>
requires(Grid::num_dimensions() == 2)
auto ridgelines(detail::rectilinear_grid::typed_vertex_property_interface<
                    Grid, T, HasNonConstReference> const& data,
                rectilinear_grid<DomainX, DomainY>&       working_grid) {
  return ridgelines(data, working_grid, execution_policy::sequential);
}
//------------------------------------------------------------------------------
/// Implementation of \cite Peikert2008ridges without filters.
template <typename Grid, arithmetic T, bool HasNonConstReference>
requires(Grid::num_dimensions() == 2)
auto ridgelines(detail::rectilinear_grid::typed_vertex_property_interface<
                Grid, T, HasNonConstReference> const& data) {
  return ridgelines(data, data.grid.copy_without_properties(),
                    execution_policy::sequential);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
