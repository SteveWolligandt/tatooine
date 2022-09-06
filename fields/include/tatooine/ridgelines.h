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
template <typename Grid, arithmetic T, bool HasNonConstReference>
requires (Grid::num_dimensions() == 2)
auto ridgelines(detail::rectilinear_grid::typed_vertex_property_interface<
                    Grid, T, HasNonConstReference> const& data,
                execution_policy_tag auto const           exec) {
  using real_type    = typename Grid::real_type;
  using edgeset_type = Edgeset2<real_type>;
  auto  helper_grid  = data.grid().copy_without_properties();
  auto& g =
      helper_grid.sample_to_vertex_property(diff<1>(data), "g", exec);
  auto& c = helper_grid.sample_to_vertex_property(
      [&](integral auto const... is) {
        auto const& g_ = g(is...);
        return vec{-g_(1), g_(0)};
      },
      "c", exec);
  auto& hessian =
      helper_grid.sample_to_vertex_property(diff<2>(data), "H", exec);

  auto& Hg = helper_grid.sample_to_vertex_property(
      [&](integral auto const... is) { return hessian(is...) * g(is...); },
      "Hg", exec);
  auto& Hc = helper_grid.sample_to_vertex_property(
      [&](integral auto const... is) { return hessian(is...) * c(is...); },
      "Hc", exec);
  auto& lambda_g = helper_grid.sample_to_vertex_property(
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
      "lambda_g", exec);
  auto& lambda_c = helper_grid.sample_to_vertex_property(
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
      "lambda_c", exec);
  auto const compute_d = [&](integral auto const... is) {
    auto const& g_  = g(is...);
    auto const& Hg_ = Hg(is...);
    return g_(0) * Hg_(1) - g_(1) * Hg_(0);
  };
  auto& d = helper_grid.sample_to_vertex_property(compute_d, "det(g|Hg)", exec);

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

  helper_grid.write("helper.vtr");
  return ridgelines;
}
//------------------------------------------------------------------------------
/// Implementation of \cite Peikert2008ridges without filters.
template <typename Grid, arithmetic T, bool HasNonConstReference>
auto ridgelines(detail::rectilinear_grid::typed_vertex_property_interface<
                Grid, T, HasNonConstReference> const& data) {
  return ridgelines(data, execution_policy::sequential);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
