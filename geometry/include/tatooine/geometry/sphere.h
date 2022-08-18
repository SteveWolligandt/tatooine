#ifndef TATOOINE_GEOMETRY_SPHERE_H
#define TATOOINE_GEOMETRY_SPHERE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/line.h>
#include <tatooine/real.h>
#include <tatooine/tensor.h>
#include <tatooine/unstructured_triangular_grid.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <floating_point Real, size_t N>
struct sphere : ray_intersectable<Real, N> {
  using this_type   = sphere<Real, N>;
  using parent_type = ray_intersectable<Real, N>;
  using vec_t    = vec<Real, N>;
  using typename parent_type::intersection_type;
  using typename parent_type::optional_intersection_type;
  //============================================================================
 private:
  vec_t m_center = vec_t::zeros();
  Real  m_radius = 1;
  //============================================================================
 public:
  sphere() = default;
  explicit sphere(Real const radius) : m_radius{radius} {}
  sphere(Real const radius, vec_t const& center)
      : m_center{center}, m_radius{radius} {}
  sphere(vec_t const& center, Real const radius)
      : m_center{center}, m_radius{radius} {}
  //----------------------------------------------------------------------------
  sphere(sphere const&) = default;
  sphere(sphere&&)      = default;
  sphere& operator=(sphere const&) = default;
  sphere& operator=(sphere&&) = default;
  //============================================================================
  auto check_intersection(ray<Real, N> const& r, Real const min_t = 0) const
      -> optional_intersection_type override {
    if constexpr (N == 3) {
      auto const m = r.origin();
      auto const b = dot(m, r.direction());
      auto const c = dot(m, m) - radius() * radius();

      // Exit if r’s origin outside s (c > 0) and r pointing away from s (b > 0)
      if (c > 0 && b > 0) {
        return {};
      }
      auto const discr = b * b - c;

      // A negative discriminant corresponds to ray missing sphere
      if (discr < 0) {
        return {};
      }

      // Ray now found to intersect sphere, compute smallest t value of
      // intersection
      auto t = -b - std::sqrt(discr);

      // If t is negative, ray started inside sphere so clamp t to zero
      if (t < min_t) {
        return {};
      }

      auto const hit_pos = r(t);
      auto const nor     = normalize(hit_pos);
      // vec        uv{std::atan2(nor(0), nor(2)) / (2 * M_PI) + M_PI / 2,
      //       std::acos(-nor(1)) / M_PI};
      return intersection_type{this, r, t, hit_pos, nor};
    } else {
      throw std::runtime_error{"sphere ray intersection not implemented for " +
                               std::to_string(N) + " dimensions."};
      return {};
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto center() const -> auto const& { return m_center; }
  constexpr auto center() -> auto& { return m_center; }
  //----------------------------------------------------------------------------
  constexpr auto radius() const { return m_radius; }
  constexpr auto radius() -> auto& { return m_radius; }
  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64>
  auto random_point(RandomEngine&& eng = RandomEngine{
                        std::random_device{}()}) const {
    auto       rand = random::uniform<Real, std::decay_t<RandomEngine>>{eng};
    auto const u         = rand();
    auto const v         = rand();
    auto const theta     = u * 2 * M_PI;
    auto const phi       = std::acos(2 * v - 1);
    auto const r         = std::cbrt(rand()) * m_radius;
    auto const sin_theta = std::sin(theta);
    auto const cos_theta = std::cos(theta);
    auto const sin_phi   = std::sin(phi);
    auto const cos_phi   = std::cos(phi);
    auto const x         = r * sin_phi * cos_theta;
    auto const y         = r * sin_phi * sin_theta;
    auto const z         = r * cos_phi;
    return vec{x, y, z};
  }
  //----------------------------------------------------------------------------
  template <typename RandReal, typename RandEngine>
  auto random_points(size_t const                           n,
                     random::uniform<RandReal, RandEngine>& rand) const {
    auto ps = std::vector<vec<Real, N>>{};
    for (size_t i = 0; i < n; ++i) {
      auto const u         = rand();
      auto const v         = rand();
      auto const theta     = u * 2 * M_PI;
      auto const phi       = std::acos(2 * v - 1);
      auto const r         = std::cbrt(rand()) * m_radius / 2;
      auto const sin_theta = std::sin(theta);
      auto const cos_theta = std::cos(theta);
      auto const sin_phi   = std::sin(phi);
      auto const cos_phi   = std::cos(phi);
      auto const x         = r * sin_phi * cos_theta;
      auto const y         = r * sin_phi * sin_theta;
      auto const z         = r * cos_phi;
      ps.emplace_back(x, y, z);
    }
    return ps;
  }
  //----------------------------------------------------------------------------
  template <typename RandEngine = std::mt19937_64>
  auto random_points(size_t const n) const {
    auto rand = random::uniform<Real, RandEngine>{};
    return random_points(n, rand);
  }
};
//------------------------------------------------------------------------------
template <floating_point Real0, floating_point Real1, size_t N>
sphere(Real0 radius, vec<Real1, N>&&)
    -> sphere<std::common_type_t<Real0, Real1>, N>;
template <floating_point Real0, floating_point Real1, size_t N>
sphere(Real0 radius, vec<Real1, N> const&)
    -> sphere<std::common_type_t<Real0, Real1>, N>;
//------------------------------------------------------------------------------
template <floating_point Real>
auto discretize(sphere<Real, 2> const& s, size_t const num_vertices) {
  using namespace std::ranges;
  auto radial = linspace<Real>{0.0, M_PI * 2, num_vertices + 1};
  radial.pop_back();

  auto ellipse             = line<Real, 2>{};
  auto radian_to_cartesian = [&s](auto const t) {
    return vec{std::cos(t) * s.radius(), std::sin(t) * s.radius()};
  };
  auto out_it = std::back_inserter(ellipse);
  copy(radial | views::transform(radian_to_cartesian), out_it);
  ellipse.set_closed(true);
  return ellipse;
}
//------------------------------------------------------------------------------
template <floating_point Real>
auto discretize(sphere<Real, 3> const& s, size_t num_subdivisions = 0) {
  using mesh_type     = unstructured_triangular_grid<Real, 3>;
  using vertex_handle = typename mesh_type::vertex_handle;
  // Real const  X = 0.525731112119133606;
  // Real const  Z = 0.850650808352039932;
  Real const X = 0.525731112119133606;
  Real const Z = 0.850650808352039932;
  auto       vertices =
      std::vector{vec{-X, 0, Z}, vec{X, 0, Z},  vec{-X, 0, -Z}, vec{X, 0, -Z},
                  vec{0, Z, X},  vec{0, Z, -X}, vec{0, -Z, X},  vec{0, -Z, -X},
                  vec{Z, X, 0},  vec{-Z, X, 0}, vec{Z, -X, 0},  vec{-Z, -X, 0}};
  auto faces = std::vector<std::array<vertex_handle, 3>>{
      {vertex_handle{0}, vertex_handle{4}, vertex_handle{1}},
      {vertex_handle{0}, vertex_handle{9}, vertex_handle{4}},
      {vertex_handle{9}, vertex_handle{5}, vertex_handle{4}},
      {vertex_handle{4}, vertex_handle{5}, vertex_handle{8}},
      {vertex_handle{4}, vertex_handle{8}, vertex_handle{1}},
      {vertex_handle{8}, vertex_handle{10}, vertex_handle{1}},
      {vertex_handle{8}, vertex_handle{3}, vertex_handle{10}},
      {vertex_handle{5}, vertex_handle{3}, vertex_handle{8}},
      {vertex_handle{5}, vertex_handle{2}, vertex_handle{3}},
      {vertex_handle{2}, vertex_handle{7}, vertex_handle{3}},
      {vertex_handle{7}, vertex_handle{10}, vertex_handle{3}},
      {vertex_handle{7}, vertex_handle{6}, vertex_handle{10}},
      {vertex_handle{7}, vertex_handle{11}, vertex_handle{6}},
      {vertex_handle{11}, vertex_handle{0}, vertex_handle{6}},
      {vertex_handle{0}, vertex_handle{1}, vertex_handle{6}},
      {vertex_handle{6}, vertex_handle{1}, vertex_handle{10}},
      {vertex_handle{9}, vertex_handle{0}, vertex_handle{11}},
      {vertex_handle{9}, vertex_handle{11}, vertex_handle{2}},
      {vertex_handle{9}, vertex_handle{2}, vertex_handle{5}},
      {vertex_handle{7}, vertex_handle{2}, vertex_handle{11}}};
  for (size_t i = 0; i < num_subdivisions; ++i) {
    std::vector<std::array<vertex_handle, 3>> subdivided_faces;
    using edge_t = std::pair<vertex_handle, vertex_handle>;
    auto subdivided =
        std::map<edge_t, size_t>{};  // vertex_handle index on edge
    for (auto& [v0, v1, v2] : faces) {
      auto edges = std::array{edge_t{v0, v1}, edge_t{v0, v2}, edge_t{v1, v2}};
      auto nvs =
          std::array{vertex_handle{0}, vertex_handle{0}, vertex_handle{0}};
      size_t i = 0;
      for (auto& edge : edges) {
        if (edge.first < edge.second) {
          std::swap(edge.first, edge.second);
        }
        if (subdivided.find(edge) == end(subdivided)) {
          subdivided[edge] = size(vertices);
          nvs[i++]         = size(vertices);
          vertices.push_back(normalize(
              (vertices[edge.first.index()] + vertices[edge.second.index()]) *
              0.5));
        } else {
          nvs[i++] = subdivided[edge];
        }
      }
      subdivided_faces.emplace_back(std::array{v0, nvs[1], nvs[0]});
      subdivided_faces.emplace_back(std::array{nvs[0], nvs[2], v1});
      subdivided_faces.emplace_back(std::array{nvs[1], v2, nvs[2]});
      subdivided_faces.emplace_back(std::array{nvs[0], nvs[1], nvs[2]});
    }
    faces = subdivided_faces;
  }
  for (auto& v : vertices) {
    v *= s.radius();
  }
  auto m = mesh_type{};
  for (auto& vertex_handle : vertices) {
    m.insert_vertex(std::move(vertex_handle) + s.center());
  }
  for (auto& f : faces) {
    m.insert_simplex(f[0], f[1], f[2]);
  }
  return m;
}
//==============================================================================
using sphere2 = sphere<real_number, 2>;
using circle = sphere<real_number, 2>;
using sphere3 = sphere<real_number, 3>;
using sphere4 = sphere<real_number, 4>;
//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================
#endif

