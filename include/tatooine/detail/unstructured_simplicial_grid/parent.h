#ifndef TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_PARENT_H
#define TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_PARENT_H
//==============================================================================
#include <tatooine/detail/unstructured_simplicial_grid/simplex_at_return_type.h>
#include <tatooine/pointset.h>
//==============================================================================
namespace tatooine::detail::unstructured_simplicial_grid {
//==============================================================================
using tatooine::pointset;
//==============================================================================
template <typename Mesh, floating_point Real, std::size_t NumDimensions,
          std::size_t SimplexDim>
struct parent : pointset<Real, NumDimensions> {
  using parent_type = pointset<Real, NumDimensions>;
  using typename pointset<Real, NumDimensions>::vertex_handle;
  using hierarchy_type = hierarchy<Mesh, Real, NumDimensions, SimplexDim>;
  using const_simplex_at_return_type =
      detail::unstructured_simplicial_grid::simplex_at_return_type<
          vertex_handle const&, SimplexDim + 1>;
  using simplex_at_return_type =
      detail::unstructured_simplicial_grid::simplex_at_return_type<
          vertex_handle&, SimplexDim + 1>;
  using parent_type::parent_type;
};
//==============================================================================
template <typename Mesh, floating_point Real>
struct parent<Mesh, Real, 3, 2> : pointset<Real, 3>,
                                  ray_intersectable<Real, 3> {
  using parent_pointset_type          = pointset<Real, 3>;
  using parent_ray_intersectable_type = ray_intersectable<Real, 3>;
  using real_type                  = Real;
  using typename pointset<real_type, 3>::vertex_handle;
  using hierarchy_type = hierarchy<Mesh, real_type, 3, 2>;
  using const_simplex_at_return_type =
      detail::unstructured_simplicial_grid::simplex_at_return_type<
          vertex_handle const&, 3>;
  using simplex_at_return_type =
      detail::unstructured_simplicial_grid::simplex_at_return_type<
          vertex_handle&, 3>;

  using parent_pointset_type::parent_pointset_type;
  using typename parent_ray_intersectable_type::intersection_type;
  using typename parent_ray_intersectable_type::optional_intersection_type;
  using typename parent_ray_intersectable_type::ray_type;
  //----------------------------------------------------------------------------
  virtual ~parent() = default;
  auto as_grid() const -> auto const& {
    return *dynamic_cast<Mesh const*>(this);
  }
  //----------------------------------------------------------------------------
  auto check_intersection(ray_type const& r, real_type const min_t = 0) const
      -> optional_intersection_type override {
    constexpr double eps          = 1e-6;
    auto const&      grid         = as_grid();
    auto             global_min_t = std::numeric_limits<real_type>::max();
    auto             inters       = optional_intersection_type{};
    if (!grid.m_hierarchy) {
      grid.build_hierarchy();
    }
    auto const possible_simplices =
        grid.m_hierarchy->collect_possible_intersections(r);
    for (auto const simplex_handle : possible_simplices) {
      auto const [vi0, vi1, vi2] = grid.simplex_at(simplex_handle);
      auto const& v0             = grid.at(vi0);
      auto const& v1             = grid.at(vi1);
      auto const& v2             = grid.at(vi2);
      auto const  v0v1           = v1 - v0;
      auto const  v0v2           = v2 - v0;
      auto const  pvec           = cross(r.direction(), v0v2);
      auto const  det            = dot(v0v1, pvec);
      // r and triangle are parallel if det is close to 0
      if (std::abs(det) < eps) {
        continue;
      }
      auto const inv_det = 1 / det;

      auto const tvec = r.origin() - v0;
      auto const u    = dot(tvec, pvec) * inv_det;
      if (u < 0 || u > 1) {
        continue;
      }

      auto const qvec = cross(tvec, v0v1);
      auto const v    = dot(r.direction(), qvec) * inv_det;
      if (v < 0 || u + v > 1) {
        continue;
      }

      auto const t                 = dot(v0v2, qvec) * inv_det;
      auto const barycentric_coord = vec<real_type, 3>{1 - u - v, u, v};
      if (t > min_t) {
        auto const pos = barycentric_coord(0) * v0 + barycentric_coord(1) * v1 +
                         barycentric_coord(2) * v2;

        if (t < global_min_t) {
          global_min_t = t;
          inters =
              intersection_type{this, r, t, pos, normalize(cross(v0v1, v2 - v1))};
        }
      }
    }

    return inters;
  }
};
//==============================================================================
}  // namespace tatooine::detail::unstructured_simplicial_grid
//==============================================================================
#endif
