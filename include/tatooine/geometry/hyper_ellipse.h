#ifndef TATOOINE_GEOMETRY_HYPER_ELLIPSE_H
#define TATOOINE_GEOMETRY_HYPER_ELLIPSE_H
//==============================================================================
#include <tatooine/reflection.h>
#include <tatooine/unstructured_triangular_grid.h>
#include <tatooine/tensor.h>
#include <tatooine/transposed_tensor.h>
//==============================================================================
namespace tatooine {
namespace geometry {
//==============================================================================
template <floating_point Real, size_t N>
struct hyper_ellipse {
  using this_t = hyper_ellipse<Real, N>;
  using vec_t  = vec<Real, N>;
  using pos_t  = vec_t;
  using mat_t  = mat<Real, N, N>;

 private:
  vec_t m_center = vec_t::zeros();
  mat_t m_S      = mat_t::eye();

 public:
  //----------------------------------------------------------------------------
  /// defaults to unit hypersphere
  constexpr hyper_ellipse() = default;
  //----------------------------------------------------------------------------
  constexpr hyper_ellipse(hyper_ellipse const&)     = default;
  constexpr hyper_ellipse(hyper_ellipse&&) noexcept = default;
  //----------------------------------------------------------------------------
  constexpr auto operator=(hyper_ellipse const&) -> hyper_ellipse& = default;
  constexpr auto operator=(hyper_ellipse&&) noexcept
      -> hyper_ellipse&  = default;
  //----------------------------------------------------------------------------
  ~hyper_ellipse() = default;
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radius.
  constexpr hyper_ellipse(Real const radius) : m_S{mat_t::eye() * radius} {}
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radius and origin point.
  constexpr hyper_ellipse(Real const radius, vec_t const& center)
      : m_center{center}, m_S{mat_t::eye() * radius} {}
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radius and origin point.
  constexpr hyper_ellipse(vec_t const& center, Real const radius)
      : m_center{center}, m_S{mat_t::eye() * radius} {}
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radius and origin point.
  constexpr hyper_ellipse(vec_t const& center, mat_t const& S)
      : m_center{center}, m_S{S} {}
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radii.
  template <typename... Radii, enable_if_arithmetic<Radii...> = true>
  constexpr hyper_ellipse(vec_t const& center, Radii const... radii)
      : m_center{center}, m_S{diag(vec{static_cast<Real>(radii)...})} {
    static_assert(sizeof...(Radii) == N,
                  "Number of radii does not match number of dimensions.");
  }
  //----------------------------------------------------------------------------
  /// Fits an ellipse through specified points.
  template <typename... Points, enable_if_vec<Points...> = true>
  constexpr hyper_ellipse(Points const&... points) {
    static_assert(sizeof...(Points) == N,
                  "Number of points does not match number of dimensions.");
    fit(points...);
  }
  //----------------------------------------------------------------------------
  /// Fits an ellipse through specified points
  template <typename... Points, enable_if_vec<Points...> = true>
  constexpr hyper_ellipse(mat_t const& H) {
    fit(H);
  }
  //============================================================================
  auto S() const -> auto const& { return m_S; }
  auto S() -> auto& { return m_S; }
  //----------------------------------------------------------------------------
  auto center() const -> auto const& { return m_center; }
  auto center() -> auto& { return m_center; }
  //----------------------------------------------------------------------------
  auto local_coordinate(pos_t const& x) const {
    return solve(S(), (x - center()));
  }
  //----------------------------------------------------------------------------
  auto squared_euclidean_distance_to_center(pos_t const& x) const {
    return squared_euclidean_distance(x, center());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto euclidean_distance_to_center(pos_t const& x) const {
    return distance(x, center());
  }
  //----------------------------------------------------------------------------
  auto squared_local_euclidean_distance_to_center(pos_t const& x) const {
    return squared_euclidean_length(local_coordinate(x));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto local_distance_to_center(pos_t const& x) const {
    return euclidean_length(local_coordinate(x));
  }
  //----------------------------------------------------------------------------
  /// Computes euclidean distance to nearest boundary point
  constexpr auto distance_to_boundary(pos_t const& x) const {
    auto const x_local                  = local_coordinate();
    auto const local_distance_to_point  = euclidian_length(x_local);
    auto const local_point_on_boundary  = x_local / local_distance_to_point;
    auto const local_offset_to_boundary = x_local - local_point_on_boundary;
    return euclidian_length(m_S * local_offset_to_boundary);
  }
  //----------------------------------------------------------------------------
  auto local_nearest_point_boundary(pos_t const& x) const {
    auto const local_point_on_boundary = normalize(local_coordinate());
    return local_point_on_boundary;
  }
  //----------------------------------------------------------------------------
  auto nearest_point_boundary(pos_t const& x) const {
    return S() * local_nearest_point_boundary(x) + center();
  }
  //============================================================================
 private:
  /// Fits an ellipse through specified points
  template <size_t... Is, typename... Points, enable_if_vec<Points...> = true>
  constexpr auto fit(std::index_sequence<Is...> /*seq*/,
                     Points const&... points) {
    auto H = mat_t{};
    ([&] { H.col(Is) = points; }(), ...);
    fit(H);
  }
  //----------------------------------------------------------------------------
 public:
  /// Fits an ellipse through specified points
  template <typename... Points, enable_if_vec<Points...> = true>
  constexpr auto fit(Points const&... points) {
    static_assert(sizeof...(Points) == N,
                  "Number of points does not match number of dimensions.");
    fit(std::make_index_sequence<N>{}, points...);
  }
  //----------------------------------------------------------------------------
  /// Fits an ellipse through columns of H
  /// \returns main axes
  constexpr auto fit(mat_t const& H) {
    auto const HHt      = H * transposed(H);
    auto const [Q, Sig] = eigenvectors_sym(HHt);
    m_S                 = Q * sqrt(diag(Sig)) * transposed(Q);
  }
  //============================================================================
  /// Computes the main axes of the ellipse.
  /// \returns main axes
  constexpr auto main_axes() const {
    auto const [Q, lambdas] = eigenvectors_sym(m_S);
    return Q * diag(lambdas);
  }
  //----------------------------------------------------------------------------
  /// Computes the main axes of the ellipse.
  /// \returns main axes
  template <typename V, typename VReal>
  constexpr auto nearest_point_on_boundary(
      base_tensor<V, VReal, N> const& x) const {
    return m_S * normalize(solve(m_S, x - m_center)) + m_center;
  }
  //----------------------------------------------------------------------------
  /// Checks if a point x is inside the ellipse.
  /// \param x point to check
  /// \returns true if x is inside ellipse.
  constexpr auto is_inside(pos_t const& x) const {
    return squared_euclidean_length(solve(m_S, x - m_center)) <= 1;
  }
};
//------------------------------------------------------------------------------
template <typename Real>
auto discretize(hyper_ellipse<Real, 3> const& s, size_t num_subdivisions = 0) {
  using mesh_t        = tatooine::unstructured_triangular_grid<Real, 3>;
  using vh = typename mesh_t::vertex_handle;
  Real const  X = 0.525731112119133606;
  Real const  Z = 0.850650808352039932;
  std::vector vertices{vec{-X, 0, Z}, vec{X, 0, Z},   vec{-X, 0, -Z},
                       vec{X, 0, -Z}, vec{0, Z, X},   vec{0, Z, -X},
                       vec{0, -Z, X}, vec{0, -Z, -X}, vec{Z, X, 0},
                       vec{-Z, X, 0}, vec{Z, -X, 0},  vec{-Z, -X, 0}};
  auto        faces = std::vector<std::array<vh, 3>>{
      {vh{0}, vh{4}, vh{1}},
      {vh{0}, vh{9}, vh{4}},
      {vh{9}, vh{5}, vh{4}},
      {vh{4}, vh{5}, vh{8}},
      {vh{4}, vh{8}, vh{1}},
      {vh{8}, vh{10}, vh{1}},
      {vh{8}, vh{3}, vh{10}},
      {vh{5}, vh{3}, vh{8}},
      {vh{5}, vh{2}, vh{3}},
      {vh{2}, vh{7}, vh{3}},
      {vh{7}, vh{10}, vh{3}},
      {vh{7}, vh{6}, vh{10}},
      {vh{7}, vh{11}, vh{6}},
      {vh{11}, vh{0}, vh{6}},
      {vh{0}, vh{1}, vh{6}},
      {vh{6}, vh{1}, vh{10}},
      {vh{9}, vh{0}, vh{11}},
      {vh{9}, vh{11}, vh{2}},
      {vh{9}, vh{2}, vh{5}},
      {vh{7}, vh{2}, vh{11}}};
  for (size_t i = 0; i < num_subdivisions; ++i) {
    std::vector<std::array<vh, 3>> subdivided_faces;
    using edge_t = std::pair<vh, vh>;
    std::map<edge_t, size_t> subdivided;  // vh index on edge
    for (auto& [v0, v1, v2] : faces) {
      std::array edges{edge_t{v0, v1}, edge_t{v0, v2}, edge_t{v1, v2}};
      std::array nvs{vh{0}, vh{0}, vh{0}};
      size_t     i = 0;
      for (auto& edge : edges) {
        if (edge.first < edge.second) {
          std::swap(edge.first, edge.second);
        }
        if (subdivided.find(edge) == end(subdivided)) {
          subdivided[edge] = size(vertices);
          nvs[i++]         = vh{};
          vertices.push_back(normalize(
              (vertices[edge.first.index()] + vertices[edge.second.index()]) * 0.5));
        } else {
          nvs[i++] = vh{subdivided[edge]};
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
    v = s.S() * v + s.center();
  }
  mesh_t m;
  for (auto& v : vertices) {
    m.insert_vertex(std::move(v));
  }
  for (auto& f : faces) {
    m.insert_cell(f[0], f[1], f[2]);
  }
  return m;
}
//==============================================================================
}  // namespace geometry
//==============================================================================
namespace reflection {
template <typename Real, size_t N>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (geometry::hyper_ellipse<Real, N>),
    TATOOINE_REFLECTION_INSERT_METHOD(center, center()),
    TATOOINE_REFLECTION_INSERT_METHOD(S, S()))
}  // namespace reflection
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/geometry/ellipse.h>
#include <tatooine/geometry/ellipsoid.h>
//==============================================================================
#endif
