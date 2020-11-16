#ifndef TATOOINE_GEOMETRY_SPHERE_H
#define TATOOINE_GEOMETRY_SPHERE_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/line.h>
#include <tatooine/triangular_mesh.h>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>

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
  sphere() : m_radius{Real(0.5)}, m_center{pos_t::zeros()} {}
  explicit sphere(Real radius) : m_radius{radius}, m_center{pos_t::zeros()} {}
  sphere(Real radius, pos_t&& center)
      : m_radius{radius}, m_center{std::move(center)} {}
  sphere(Real radius, const pos_t& center)
      : m_radius{radius}, m_center{center} {}
  //----------------------------------------------------------------------------
  sphere(const sphere&) = default;
  sphere(sphere&&)      = default;
  sphere& operator=(const sphere&) = default;
  sphere& operator=(sphere&&) = default;
  //============================================================================
  std::optional<intersection<Real, N>> check_intersection(
      const ray<Real, N>& r, const Real min_t = 0) const override {
    return tatooine::geometry::check_intersection(*this, r, min_t);
  }
  //----------------------------------------------------------------------------
  constexpr auto radius() const {
    return m_radius;
  }
  constexpr auto& radius() {
    return m_radius;
  }
  //----------------------------------------------------------------------------
  constexpr const auto& center() const {
    return m_center;
  }
  constexpr auto& center() {
    return m_center;
  }
};
//------------------------------------------------------------------------------
template <typename Real>
auto discretize(const sphere<Real, 2>& s, size_t const num_vertices) {
  using namespace boost;
  using namespace adaptors;
  linspace<Real>      radial{0.0, M_PI * 2, num_vertices};
  radial.pop_back();

  line<Real, 2> ellipse;
  auto          radian_to_cartesian = [&s](auto const t) {
    return vec{std::cos(t) * s.radius(), std::sin(t) * s.radius()} + s.center();
  };
  auto          out_it = std::back_inserter(ellipse);
  copy(radial | transformed(radian_to_cartesian), out_it);
  ellipse.set_closed(true);
  return ellipse;
}
//------------------------------------------------------------------------------
template <typename Real>
auto discretize(const sphere<Real, 3>& s, size_t num_subdivisions = 0) {
  using mesh_t       = triangular_mesh<Real, 3>;
  using vertex_index = typename mesh_t::vertex_index;
  // const Real  X = 0.525731112119133606;
  // const Real  Z = 0.850650808352039932;
  const Real  X = 0.525731112119133606;
  const Real  Z = 0.850650808352039932;
  std::vector vertices{vec{-X, 0, Z}, vec{X, 0, Z},   vec{-X, 0, -Z},
                       vec{X, 0, -Z}, vec{0, Z, X},   vec{0, Z, -X},
                       vec{0, -Z, X}, vec{0, -Z, -X}, vec{Z, X, 0},
                       vec{-Z, X, 0}, vec{Z, -X, 0},  vec{-Z, -X, 0}};
  std::vector<std::array<vertex_index, 3>> triangles{
      {vertex_index{0}, vertex_index{4}, vertex_index{1}},
      {vertex_index{0}, vertex_index{9}, vertex_index{4}},
      {vertex_index{9}, vertex_index{5}, vertex_index{4}},
      {vertex_index{4}, vertex_index{5}, vertex_index{8}},
      {vertex_index{4}, vertex_index{8}, vertex_index{1}},
      {vertex_index{8}, vertex_index{10}, vertex_index{1}},
      {vertex_index{8}, vertex_index{3}, vertex_index{10}},
      {vertex_index{5}, vertex_index{3}, vertex_index{8}},
      {vertex_index{5}, vertex_index{2}, vertex_index{3}},
      {vertex_index{2}, vertex_index{7}, vertex_index{3}},
      {vertex_index{7}, vertex_index{10}, vertex_index{3}},
      {vertex_index{7}, vertex_index{6}, vertex_index{10}},
      {vertex_index{7}, vertex_index{11}, vertex_index{6}},
      {vertex_index{11}, vertex_index{0}, vertex_index{6}},
      {vertex_index{0}, vertex_index{1}, vertex_index{6}},
      {vertex_index{6}, vertex_index{1}, vertex_index{10}},
      {vertex_index{9}, vertex_index{0}, vertex_index{11}},
      {vertex_index{9}, vertex_index{11}, vertex_index{2}},
      {vertex_index{9}, vertex_index{2}, vertex_index{5}},
      {vertex_index{7}, vertex_index{2}, vertex_index{11}}};
  for (size_t i = 0; i < num_subdivisions; ++i) {
    std::vector<std::array<vertex_index, 3>> subdivided_triangles;
    using edge_t = std::pair<vertex_index, vertex_index>;
    std::map<edge_t, size_t> subdivided;  // vertex_index index on edge
    for (auto& [v0, v1, v2] : triangles) {
      std::array edges{edge_t{v0, v1}, edge_t{v0, v2}, edge_t{v1, v2}};
      std::array nvs{vertex_index{0}, vertex_index{0}, vertex_index{0}};
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
      subdivided_triangles.emplace_back(std::array{v0, nvs[1], nvs[0]});
      subdivided_triangles.emplace_back(std::array{nvs[0], nvs[2], v1});
      subdivided_triangles.emplace_back(std::array{nvs[1], v2, nvs[2]});
      subdivided_triangles.emplace_back(std::array{nvs[0], nvs[1], nvs[2]});
    }
    triangles = subdivided_triangles;
  }
  for (auto& v : vertices) {
    v *= s.radius();
    v += s.center();
  }
  mesh_t m;
  for (auto& vertex_index : vertices) {
    m.insert_vertex(std::move(vertex_index));
  }
  for (auto& tri : triangles) {
    m.insert_triangle(std::move(tri));
  }
  return m;
}
//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================
#endif

