#ifndef TATOOINE_GEOMETRY_SPHERE_H
#define TATOOINE_GEOMETRY_SPHERE_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/simple_tri_mesh.h>
#include "primitive.h"
#include "sphere_ray_intersection.h"
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <typename Real, size_t N>
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
  sphere() : m_radius{1}, m_center{pos_t::zeros()} {}
  explicit sphere(Real radius) : m_radius{radius}, m_center{pos_t::zeros()} {}
  sphere(Real radius, pos_t&& center)
      : m_radius{radius}, m_center{std::move(center)} {}
  sphere(Real radius, const pos_t& center)
      : m_radius{radius}, m_center{center} {}
  //----------------------------------------------------------------------------
  sphere(const sphere&) = default;
  sphere(sphere&&) = default;
  sphere& operator=(const sphere&) = default;
  sphere& operator=(sphere&&) = default;
  //============================================================================
  std::optional<intersection<Real, N>> check_intersection(
      const ray<Real, N>& r, const Real min_t = 0) const override {
    return tatooine::geometry::check_intersection(*this, r, min_t);
  }
  //----------------------------------------------------------------------------
  constexpr auto  radius() const { return m_radius; }
  constexpr auto& radius() { return m_radius; }
  //----------------------------------------------------------------------------
  constexpr const auto& center() const { return m_center; }
  constexpr auto&       center() { return m_center; }
};
//------------------------------------------------------------------------------
template <typename Real>
auto discretize(const sphere<Real, 3>& s, size_t num_subdivisions = 0) {
  using mesh_t = simple_tri_mesh<Real, 3>;
  using vertex  = typename mesh_t::vertex;
  const Real  X = 0.525731112119133606;
  const Real  Z = 0.850650808352039932;
  std::vector vertices{vec{-X, 0, Z}, vec{X, 0, Z},   vec{-X, 0, -Z},
                       vec{X, 0, -Z}, vec{0, Z, X},   vec{0, Z, -X},
                       vec{0, -Z, X}, vec{0, -Z, -X}, vec{Z, X, 0},
                       vec{-Z, X, 0}, vec{Z, -X, 0},  vec{-Z, -X, 0}};
  std::vector<std::array<vertex, 3>> triangles = {
      {0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1},
      {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3},  {2, 7, 3},
      {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
      {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5},  {7, 2, 11}};

  for (size_t i = 0; i < num_subdivisions; ++i) {
    std::vector<std::array<vertex, 3>> subdivided_triangles;
    using edge_t = std::pair<vertex, vertex>;
    std::map<edge_t, size_t> subdivided; //vertex index on edge
    for (auto &[v0, v1, v2] : triangles) {
      std::array edges{edge_t{v0, v1}, edge_t{v0, v2}, edge_t{v1, v2}};
      std::array<vertex, 3> nvs{0, 0, 0};
      size_t                i = 0;
      for (auto& edge : edges) {
        if (edge.first < edge.second) { std::swap(edge.first, edge.second); }
        if (subdivided.find(edge) == end(subdivided)) {
          subdivided[edge]  = size(vertices);
          nvs[i++] = size(vertices);
          vertices.push_back(normalize(
              (vertices[edge.first.i] + vertices[edge.second.i]) * 0.5));
        } else {
          nvs[i++] = subdivided[edge];
        }
      }
      subdivided_triangles.emplace_back(std::array<vertex,3>{v0, nvs[0], nvs[1]});
      subdivided_triangles.emplace_back(std::array<vertex,3>{nvs[0], v1, nvs[2]});
      subdivided_triangles.emplace_back(std::array<vertex,3>{nvs[1], nvs[2], v2});
      subdivided_triangles.emplace_back(std::array<vertex,3>{nvs[0], nvs[1], nvs[2]});
    }
    triangles = subdivided_triangles;
  }
  for (auto& v : vertices) {
    v *= s.radius();
    v += s.center();
  }
  mesh_t m;
  for (auto& vertex : vertices) { m.insert_vertex(std::move(vertex)); }
  for (auto& tri : triangles) { m.insert_face(std::move(tri)); }
  return m;
}
//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================
#endif

