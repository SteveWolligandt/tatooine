#ifndef TATOOINE_GEOMETRY_SPHERE_H
#define TATOOINE_GEOMETRY_SPHERE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/line.h>
#include <tatooine/tensor.h>
#include <tatooine/triangular_mesh.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>

#include "primitive.h"
#include "sphere_ray_intersection.h"
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <floating_point Real, size_t N>
struct sphere : primitive<Real, N> {
  using this_t   = sphere<Real, N>;
  using parent_t = primitive<Real, N>;
  using typename parent_t::pos_t;
  //============================================================================
 private:
  Real  m_radius;
  pos_t m_center;
  //============================================================================
 public:
  sphere() : m_radius{Real(0.5)}, m_center{pos_t::zeros()} {}
  explicit sphere(Real radius) : m_radius{radius}, m_center{pos_t::zeros()} {}
  sphere(Real radius, pos_t&& center)
      : m_radius{radius}, m_center{std::move(center)} {}
  sphere(Real radius, pos_t const& center)
      : m_radius{radius}, m_center{center} {}
  //----------------------------------------------------------------------------
  sphere(sphere const&) = default;
  sphere(sphere&&)      = default;
  sphere& operator=(sphere const&) = default;
  sphere& operator=(sphere&&) = default;
  //============================================================================
  auto check_intersection(ray<Real, N> const& r, Real const min_t = 0) const
      -> std::optional<intersection<Real, N>> override {
    return tatooine::geometry::check_intersection(*this, r, min_t);
  }
  //----------------------------------------------------------------------------
  constexpr auto radius() const { return m_radius; }
  constexpr auto radius() -> auto& { return m_radius; }
  //----------------------------------------------------------------------------
  constexpr auto center() const -> auto const& { return m_center; }
  constexpr auto center() -> auto& { return m_center; }
  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64>
  auto random_point(RandomEngine&& eng = RandomEngine{
                        std::random_device{}()}) const {
    auto       rand = random_uniform<Real, std::decay_t<RandomEngine>>{eng};
    auto const u         = rand();
    auto const v         = rand();
    auto const theta     = u * 2 * M_PI;
    auto const phi       = std::acos(2 * v - 1);
    auto const r         = std::cbrt(rand()) * m_radius;
    auto const sin_theta = std::sin(theta);
    auto const cos_theta = std::cos(theta);
    auto const sin_phi   = std::sin(phi);
    auto const cos_phi   = std::cos(phi);
    auto const x         = r * sin_phi * cos_theta + m_center(0);
    auto const y         = r * sin_phi * sin_theta + m_center(1);
    auto const z         = r * cos_phi + m_center(2);
    return vec{x, y, z};
  }
  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64>
  auto random_points(size_t const n, RandomEngine&& eng = RandomEngine{
                                         std::random_device{}()}) const {
    std::vector<vec<Real, N>> ps;
    auto rand = random_uniform<Real, std::decay_t<RandomEngine>>{
        std::forward<RandomEngine>(eng)};
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
      ps.emplace_back(x + m_center(0), y + m_center(1), z + m_center(2));
    }
    return ps;
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
  using namespace boost;
  using namespace adaptors;
  linspace<Real> radial{0.0, M_PI * 2, num_vertices};
  radial.pop_back();

  line<Real, 2> ellipse;
  auto          radian_to_cartesian = [&s](auto const t) {
    return vec{std::cos(t) * s.radius(), std::sin(t) * s.radius()} + s.center();
  };
  auto out_it = std::back_inserter(ellipse);
  copy(radial | transformed(radian_to_cartesian), out_it);
  ellipse.set_closed(true);
  return ellipse;
}
//------------------------------------------------------------------------------
template <floating_point Real>
auto discretize(sphere<Real, 3> const& s, size_t num_subdivisions = 0) {
  using mesh_t        = triangular_mesh<Real, 3>;
  using vertex_handle = typename mesh_t::vertex_handle;
  // Real const  X = 0.525731112119133606;
  // Real const  Z = 0.850650808352039932;
  Real const  X = 0.525731112119133606;
  Real const  Z = 0.850650808352039932;
  std::vector vertices{vec{-X, 0, Z}, vec{X, 0, Z},   vec{-X, 0, -Z},
                       vec{X, 0, -Z}, vec{0, Z, X},   vec{0, Z, -X},
                       vec{0, -Z, X}, vec{0, -Z, -X}, vec{Z, X, 0},
                       vec{-Z, X, 0}, vec{Z, -X, 0},  vec{-Z, -X, 0}};
  std::vector<std::array<vertex_handle, 3>> faces{
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
    std::map<edge_t, size_t> subdivided;  // vertex_handle index on edge
    for (auto& [v0, v1, v2] : faces) {
      std::array edges{edge_t{v0, v1}, edge_t{v0, v2}, edge_t{v1, v2}};
      std::array nvs{vertex_handle{0}, vertex_handle{0}, vertex_handle{0}};
      size_t     i = 0;
      for (auto& edge : edges) {
        if (edge.first < edge.second) {
          std::swap(edge.first, edge.second);
        }
        if (subdivided.find(edge) == end(subdivided)) {
          subdivided[edge] = size(vertices);
          nvs[i++]         = size(vertices);
          vertices.push_back(normalize(
              (vertices[edge.first.i] + vertices[edge.second.i]) * 0.5));
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
    v += s.center();
  }
  mesh_t m;
  for (auto& vertex_handle : vertices) {
    m.insert_vertex(std::move(vertex_handle));
  }
  for (auto& f : faces) {
    m.insert_face(f[0], f[1], f[2]);
  }
  return m;
}
using sphere2 = sphere<double, 2>;
using sphere3 = sphere<double, 3>;
using sphere4 = sphere<double, 4>;
//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================
#endif

