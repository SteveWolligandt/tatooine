#ifndef TATOOINE_AXIS_ALIGNED_BOUNDING_BOX_H
#define TATOOINE_AXIS_ALIGNED_BOUNDING_BOX_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/ray_intersectable.h>
#include <tatooine/separating_axis_theorem.h>
#include <tatooine/tensor.h>
#include <tatooine/type_traits.h>
#include <tatooine/vtk_legacy.h>

#include <limits>
#include <ostream>
//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail {
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
// template <typename AABB, typename Real, std::size_t NumDimensions>
// struct aabb_ray_intersectable_parent {};
template <typename AABB, typename Real, std::size_t NumDimensions>
struct aabb_ray_intersectable_parent : ray_intersectable<Real, NumDimensions> {
  using parent_type = ray_intersectable<Real, NumDimensions>;
  using typename parent_type::intersection_type;
  using typename parent_type::optional_intersection_type;
  using typename parent_type::ray_type;
  //============================================================================
  auto as_aabb() const -> auto const& {
    return *dynamic_cast<AABB const*>(this);
  }
  //============================================================================
  // ray_intersectable overrides
  //============================================================================
  auto check_intersection(ray_type const& r, Real const = 0) const
      -> optional_intersection_type override {
    auto const& aabb = as_aabb();
    enum Quadrant { right, left, middle };
    auto coord           = vec<Real, NumDimensions>{};
    auto inside          = true;
    auto quadrant        = make_array<Quadrant, NumDimensions>();
    auto which_plane     = std::size_t(0);
    auto max_t           = make_array<Real, NumDimensions>();
    auto candidate_plane = make_array<Real, NumDimensions>();

    // Find candidate planes; this loop can be avoided if rays cast all from the
    // eye(assume perpsective view)
    for (std::size_t i = 0; i < NumDimensions; i++)
      if (r.origin(i) < aabb.min(i)) {
        quadrant[i]        = left;
        candidate_plane[i] = aabb.min(i);
        inside             = false;
      } else if (r.origin(i) > aabb.max(i)) {
        quadrant[i]        = right;
        candidate_plane[i] = aabb.max(i);
        inside             = false;
      } else {
        quadrant[i] = middle;
      }

    // Ray origin inside bounding box
    if (inside) {
      return intersection_type{this, r, Real(0), r.origin(),
                               vec<Real, NumDimensions>::zeros()};
    }

    // Calculate T distances to candidate planes
    for (std::size_t i = 0; i < NumDimensions; i++)
      if (quadrant[i] != middle && r.direction(i) != 0) {
        max_t[i] = (candidate_plane[i] - r.origin(i)) / r.direction(i);
      } else {
        max_t[i] = -1;
      }

    // Get largest of the max_t's for final choice of intersection
    which_plane = 0;
    for (std::size_t i = 1; i < NumDimensions; i++)
      if (max_t[which_plane] < max_t[i]) {
        which_plane = i;
      }

    // Check final candidate actually inside box
    if (max_t[which_plane] < 0) {
      return {};
    }
    for (std::size_t i = 0; i < NumDimensions; i++)
      if (which_plane != i) {
        coord(i) = r.origin(i) + max_t[which_plane] * r.direction(i);
        if (coord(i) < aabb.min(i) || coord(i) > aabb.max(i)) {
          return {};
        }
      } else {
        coord(i) = candidate_plane[i];
      }
    return intersection_type{this, r, max_t[which_plane], coord,
                             vec<Real, NumDimensions>::zeros()};
  }
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
}  // namespace detail
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Real, std::size_t NumDimensions>
struct axis_aligned_bounding_box
    : detail::aabb_ray_intersectable_parent<
          axis_aligned_bounding_box<Real, NumDimensions>, Real, NumDimensions> {
  static_assert(is_arithmetic<Real>);
  //============================================================================
  using real_type = Real;
  using this_type = axis_aligned_bounding_box<Real, NumDimensions>;
  using vec_type  = vec<Real, NumDimensions>;
  using pos_type  = vec_type;

  static constexpr auto num_dimensions() { return NumDimensions; }
  static constexpr auto infinite() {
    return this_type{pos_type::ones() * -std::numeric_limits<real_type>::max(),
                     pos_type::ones() * std::numeric_limits<real_type>::max()};
  };
  //============================================================================
 private:
  pos_type m_min;
  pos_type m_max;
  //============================================================================
 public:
  constexpr axis_aligned_bounding_box()
      : m_min{pos_type::ones() * std::numeric_limits<real_type>::max()},
        m_max{pos_type::ones() * -std::numeric_limits<real_type>::max()} {}
  constexpr axis_aligned_bounding_box(axis_aligned_bounding_box const& other) =
      default;
  constexpr axis_aligned_bounding_box(
      axis_aligned_bounding_box&& other) noexcept = default;
  constexpr auto operator           =(axis_aligned_bounding_box const& other)
      -> axis_aligned_bounding_box& = default;
  constexpr auto operator=(axis_aligned_bounding_box&& other) noexcept
      -> axis_aligned_bounding_box& = default;
  ~axis_aligned_bounding_box()      = default;
  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  constexpr axis_aligned_bounding_box(vec<Real0, NumDimensions>&& min,
                                      vec<Real1, NumDimensions>&& max) noexcept
      : m_min{std::move(min)}, m_max{std::move(max)} {}
  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  constexpr axis_aligned_bounding_box(vec<Real0, NumDimensions> const& min,
                                      vec<Real1, NumDimensions> const& max)
      : m_min{min}, m_max{max} {}
  //----------------------------------------------------------------------------
  template <typename Tensor0, typename Tensor1, typename Real0, typename Real1>
  constexpr axis_aligned_bounding_box(
      base_tensor<Tensor0, Real0, NumDimensions> const& min,
      base_tensor<Tensor1, Real1, NumDimensions> const& max)
      : m_min{min}, m_max{max} {}
  //============================================================================
  auto constexpr min() const -> auto const& { return m_min; }
  auto constexpr min() -> auto& { return m_min; }
  auto constexpr min(std::size_t i) const -> auto const& { return m_min(i); }
  auto constexpr min(std::size_t i) -> auto& { return m_min(i); }
  //----------------------------------------------------------------------------
  auto constexpr max() const -> auto const& { return m_max; }
  auto constexpr max() -> auto& { return m_max; }
  auto constexpr max(std::size_t i) const -> auto const& { return m_max(i); }
  auto constexpr max(std::size_t i) -> auto& { return m_max(i); }
  //----------------------------------------------------------------------------
  auto constexpr extents() const { return m_max - m_min; }
  auto constexpr extent(std::size_t i) const { return m_max(i) - m_min(i); }
  //----------------------------------------------------------------------------
  auto constexpr center() const { return (m_max + m_min) * Real(0.5); }
  auto constexpr center(std::size_t const i) const {
    return (m_max(i) + m_min(i)) * Real(0.5);
  }
  //----------------------------------------------------------------------------
  auto constexpr is_inside(pos_type const& p) const {
    for (std::size_t i = 0; i < NumDimensions; ++i) {
      if (p(i) < m_min(i) || m_max(i) < p(i)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  /// x3 +--------------+ x2
  ///    |              |
  ///    |              |
  /// x0 +--------------+ x1
  constexpr auto is_rectangle_inside(vec<Real, 2> x0, vec<Real, 2> x1,
                                     vec<Real, 2> x2, vec<Real, 2> x3) const
      requires(NumDimensions == 2) {
    auto const c = center();
    auto const e = extents() / 2;

    x0 -= c;
    x1 -= c;
    x2 -= c;
    x3 -= c;

    // edges of rectangle
    auto const f0 = x1 - x0;  // normal of f1
    auto const f1 = x3 - x0;  // normal of f0

    // normals of aabb
    vec_type constexpr u0{1, 0};
    vec_type constexpr u1{0, 1};

    auto is_separating_axis = [&](auto const& axis) {
      // Project all 4 vertices of the rectangle onto the seperating axis
      auto const p0 = dot(x0, axis);
      auto const p1 = dot(x1, axis);
      auto const p2 = dot(x2, axis);
      auto const p3 = dot(x3, axis);
      // Project the AABB onto the seperating axis.
      // We don't care about the end points of the projection just the length of
      // the half-size of the aabb. That is, we're only casting the extents onto
      // the seperating axis, not the aabb center. We don't need to cast the
      // center, because we know that the aabb is at origin compared to the
      // triangle!
      auto r =
          e.x() * std::abs(dot(u0, axis)) + e.y() * std::abs(dot(u1, axis));
      return tatooine::max(-tatooine::max(p0, p1, p2, p3),
                           tatooine::min(p0, p1, p2, p3)) > r;
    };
    if (is_separating_axis(u0)) {
      return false;
    }
    if (is_separating_axis(u1)) {
      return false;
    }
    if (is_separating_axis(f0)) {
      return false;
    }
    if (is_separating_axis(f1)) {
      return false;
    }
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto is_simplex_inside(vec<Real, 2> const& x0,
                                   vec<Real, 2> const& x1,
                                   vec<Real, 2> const& x2) const
      requires(NumDimensions == 2) {
    auto const c = center();
    auto const e = extents();
    auto const x0_centered = x0 - c;
    auto const x1_centered = x1 - c;
    auto const x2_centered = x2 - c;
    constexpr auto u0 = vec_type {1, 0};
    constexpr auto u1 = vec_type {0, 1};
    auto is_separating_axis = [&](auto const& axis) {
      // Project all 4 vertices of the rectangle onto the seperating axis
      auto const p0 = dot(x0_centered, axis);
      auto const p1 = dot(x1_centered, axis);
      auto const p2 = dot(x2_centered, axis);
      // Project the AABB onto the seperating axis.
      // We don't care about the end points of the projection just the length of
      // the half-size of the aabb. That is, we're only casting the extents onto
      // the seperating axis, not the aabb center. We don't need to cast the
      // center, because we know that the aabb is at origin compared to the
      // triangle!
      auto const r =
          e.x() * std::abs(dot(u0, axis)) + e.y() * std::abs(dot(u1, axis));
      auto const m =
          tatooine::max(-tatooine::max(p0, p1, p2), tatooine::min(p0, p1, p2));
      return m > r;
    };
    if (is_separating_axis(u0)) {
      return false;
    }
    if (is_separating_axis(u1)) {
      return false;
    }
    if (is_separating_axis(vec_type{x0_centered(1) - x1_centered(1), x1_centered(0) - x0_centered(0)})) {
      return false;
    }
    if (is_separating_axis(vec_type{x1_centered(1) - x2_centered(1), x2_centered(0) - x1_centered(0)})) {
      return false;
    }
    if (is_separating_axis(vec_type{x2_centered(1) - x0_centered(1), x0_centered(0) - x2_centered(0)})) {
      return false;
    }
    return true;
  }
  //----------------------------------------------------------------------------
  /// from here:
  /// https://gdbooks.gitbooks.io/3dcollisions/content/Chapter4/aabb-triangle.html
  constexpr auto is_simplex_inside(vec<Real, 3> x0, vec<Real, 3> x1,
                                   vec<Real, 3> x2) const
      requires(NumDimensions == 3) {
    auto const c = center();
    auto const e = extents() / 2;

    x0 -= c;
    x1 -= c;
    x2 -= c;

    auto const f0 = x1 - x0;
    auto const f1 = x2 - x1;
    auto const f2 = x0 - x2;

    vec_type constexpr u0{1, 0, 0};
    vec_type constexpr u1{0, 1, 0};
    vec_type constexpr u2{0, 0, 1};

    auto is_separating_axis = [&](auto const& axis) {
      auto const p0 = dot(x0, axis);
      auto const p1 = dot(x1, axis);
      auto const p2 = dot(x2, axis);
      auto       r  = e.x() * std::abs(dot(u0, axis)) +
               e.y() * std::abs(dot(u1, axis)) +
               e.z() * std::abs(dot(u2, axis));
      return tatooine::max(-tatooine::max(p0, p1, p2),
                           tatooine::min(p0, p1, p2)) > r;
    };

    if (is_separating_axis(cross(u0, f0))) {
      return false;
    }
    if (is_separating_axis(cross(u0, f1))) {
      return false;
    }
    if (is_separating_axis(cross(u0, f2))) {
      return false;
    }
    if (is_separating_axis(cross(u1, f0))) {
      return false;
    }
    if (is_separating_axis(cross(u1, f1))) {
      return false;
    }
    if (is_separating_axis(cross(u1, f2))) {
      return false;
    }
    if (is_separating_axis(cross(u2, f0))) {
      return false;
    }
    if (is_separating_axis(cross(u2, f1))) {
      return false;
    }
    if (is_separating_axis(cross(u2, f2))) {
      return false;
    }
    if (is_separating_axis(u0)) {
      return false;
    }
    if (is_separating_axis(u1)) {
      return false;
    }
    if (is_separating_axis(u2)) {
      return false;
    }
    if (is_separating_axis(cross(f0, f1))) {
      return false;
    }
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto is_simplex_inside(vec<Real, 3> x0, vec<Real, 3> x1,
                                   vec<Real, 3> x2, vec<Real, 3> x3) const
      requires(NumDimensions == 3) {
    auto const c = center();
    auto const e = extents() / 2;

    x0 -= c;
    x1 -= c;
    x2 -= c;
    x3 -= c;

    auto const f0 = x1 - x0;
    auto const f1 = x2 - x1;
    auto const f2 = x0 - x2;
    auto const f3 = x3 - x1;
    auto const f4 = x2 - x3;
    auto const f5 = x3 - x0;

    vec_type constexpr u0{1, 0, 0};
    vec_type constexpr u1{0, 1, 0};
    vec_type constexpr u2{0, 0, 1};

    auto is_separating_axis = [&](auto const axis) {
      auto const p0 = dot(x0, axis);
      auto const p1 = dot(x1, axis);
      auto const p2 = dot(x2, axis);
      auto const p3 = dot(x3, axis);
      auto       r  = e.x() * std::abs(dot(u0, axis)) +
               e.y() * std::abs(dot(u1, axis)) +
               e.z() * std::abs(dot(u2, axis));
      return tatooine::max(-tatooine::max(p0, p1, p2, p3),
                           tatooine::min(p0, p1, p2, p3)) > r;
    };

    if (is_separating_axis(cross(u0, f0))) {
      return false;
    }
    if (is_separating_axis(cross(u0, f1))) {
      return false;
    }
    if (is_separating_axis(cross(u0, f2))) {
      return false;
    }
    if (is_separating_axis(cross(u0, f3))) {
      return false;
    }
    if (is_separating_axis(cross(u0, f4))) {
      return false;
    }
    if (is_separating_axis(cross(u0, f5))) {
      return false;
    }
    if (is_separating_axis(cross(u1, f0))) {
      return false;
    }
    if (is_separating_axis(cross(u1, f1))) {
      return false;
    }
    if (is_separating_axis(cross(u1, f2))) {
      return false;
    }
    if (is_separating_axis(cross(u1, f3))) {
      return false;
    }
    if (is_separating_axis(cross(u1, f4))) {
      return false;
    }
    if (is_separating_axis(cross(u1, f5))) {
      return false;
    }
    if (is_separating_axis(cross(u2, f0))) {
      return false;
    }
    if (is_separating_axis(cross(u2, f1))) {
      return false;
    }
    if (is_separating_axis(cross(u2, f2))) {
      return false;
    }
    if (is_separating_axis(cross(u2, f3))) {
      return false;
    }
    if (is_separating_axis(cross(u2, f4))) {
      return false;
    }
    if (is_separating_axis(cross(u2, f5))) {
      return false;
    }
    if (is_separating_axis(u0)) {
      return false;
    }
    if (is_separating_axis(u1)) {
      return false;
    }
    if (is_separating_axis(u2)) {
      return false;
    }
    if (is_separating_axis(cross(f0, f1))) {
      return false;
    }
    if (is_separating_axis(cross(f3, f4))) {
      return false;
    }
    if (is_separating_axis(cross(-f0, f5))) {
      return false;
    }
    if (is_separating_axis(cross(f5, -f2))) {
      return false;
    }
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto operator+=(pos_type const& point) {
    for (std::size_t i = 0; i < point.num_components(); ++i) {
      m_min(i) = std::min(m_min(i), point(i));
      m_max(i) = std::max(m_max(i), point(i));
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto reset() {
    for (std::size_t i = 0; i < NumDimensions; ++i) {
      m_min(i) = std::numeric_limits<Real>::max();
      m_max(i) = -std::numeric_limits<Real>::max();
    }
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto add_dimension(Real const min,
                                             Real const max) const {
    axis_aligned_bounding_box<Real, NumDimensions + 1> addeddim;
    for (std::size_t i = 0; i < NumDimensions; ++i) {
      addeddim.m_min(i) = m_min(i);
      addeddim.m_max(i) = m_max(i);
    }
    addeddim.m_min(NumDimensions) = min;
    addeddim.m_max(NumDimensions) = max;
    return addeddim;
  }
  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64>
  auto random_point(RandomEngine&& random_engine = RandomEngine{
                        std::random_device{}()}) const {
    pos_type p;
    for (std::size_t i = 0; i < NumDimensions; ++i) {
      std::uniform_real_distribution<Real> distribution{m_min(i), m_max(i)};
      p(i) = distribution(random_engine);
    }
    return p;
  }
  //----------------------------------------------------------------------------
  auto write_vtk(filesystem::path const& path) {
    vtk::legacy_file_writer f{path, vtk::dataset_type::polydata};
    f.write_header();
    std::vector<vec<real_type, 3>>        positions;
    std::vector<std::vector<std::size_t>> indices;

    positions.push_back(vec{min(0), min(1), min(2)});
    positions.push_back(vec{max(0), min(1), min(2)});
    positions.push_back(vec{max(0), max(1), min(2)});
    positions.push_back(vec{min(0), max(1), min(2)});
    positions.push_back(vec{min(0), min(1), max(2)});
    positions.push_back(vec{max(0), min(1), max(2)});
    positions.push_back(vec{max(0), max(1), max(2)});
    positions.push_back(vec{min(0), max(1), max(2)});
    indices.push_back({0, 1, 2, 3, 0});
    indices.push_back({4, 5, 6, 7, 4});
    indices.push_back({0, 4});
    indices.push_back({1, 5});
    indices.push_back({2, 6});
    indices.push_back({3, 7});
    f.write_points(positions);
    f.write_lines(indices);
  }
};
template <typename Real>
using AABB2 = axis_aligned_bounding_box<Real, 2>;
template <typename Real>
using AABB3 = axis_aligned_bounding_box<Real, 3>;
template <typename Real, std::size_t NumDimensions>
using aabb = axis_aligned_bounding_box<Real, NumDimensions>;

using aabb2d = aabb<double, 2>;
using aabb2f = aabb<float, 2>;
using aabb2  = aabb<real_number, 2>;

using aabb3d = aabb<double, 3>;
using aabb3f = aabb<float, 3>;
using aabb3  = aabb<real_number, 3>;
//==============================================================================
// deduction guides
//==============================================================================
template <typename Real0, typename Real1, std::size_t NumDimensions>
axis_aligned_bounding_box(vec<Real0, NumDimensions> const&,
                          vec<Real1, NumDimensions> const&)
    -> axis_aligned_bounding_box<common_type<Real0, Real1>, NumDimensions>;
//------------------------------------------------------------------------------
template <typename Real0, typename Real1, std::size_t NumDimensions>
axis_aligned_bounding_box(vec<Real0, NumDimensions>&&,
                          vec<Real1, NumDimensions>&&)
    -> axis_aligned_bounding_box<common_type<Real0, Real1>, NumDimensions>;
//------------------------------------------------------------------------------
template <typename Tensor0, typename Tensor1, typename Real0, typename Real1,
          std::size_t NumDimensions>
axis_aligned_bounding_box(base_tensor<Tensor0, Real0, NumDimensions>&&,
                          base_tensor<Tensor1, Real1, NumDimensions>&&)
    -> axis_aligned_bounding_box<common_type<Real0, Real1>, NumDimensions>;

//==============================================================================
// ostream output
//==============================================================================
template <typename Real, std::size_t NumDimensions>
auto operator<<(std::ostream&                                         out,
                axis_aligned_bounding_box<Real, NumDimensions> const& bb)
    -> std::ostream& {
  out << std::scientific;
  for (std::size_t i = 0; i < NumDimensions; ++i) {
    out << "[ ";
    if (bb.min(i) >= 0) {
      out << ' ';
    }
    out << bb.min(i) << " .. ";
    if (bb.max(i) >= 0) {
      out << ' ';
    }
    out << bb.max(i) << " ]\n";
  }
  out << std::defaultfloat;
  return out;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
