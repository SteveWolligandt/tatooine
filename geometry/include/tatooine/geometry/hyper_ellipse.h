#ifndef TATOOINE_GEOMETRY_HYPER_ELLIPSE_H
#define TATOOINE_GEOMETRY_HYPER_ELLIPSE_H
//==============================================================================
#include <tatooine/line.h>
#include <tatooine/reflection.h>
#include <tatooine/tensor.h>
#include <tatooine/transposed_tensor.h>
#include <tatooine/unstructured_triangular_grid.h>
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions> struct hyper_ellipse {
  static_assert(NumDimensions > 1);
  using this_type = hyper_ellipse<Real, NumDimensions>;
  using vec_type = vec<Real, NumDimensions>;
  using pos_type = vec_type;
  using mat_type = mat<Real, NumDimensions, NumDimensions>;
  using real_type = Real;
  static auto constexpr num_dimensions() { return NumDimensions; }
  //----------------------------------------------------------------------------
private:
  //----------------------------------------------------------------------------
  vec_type m_center = vec_type::zeros();
  mat_type m_S = mat_type::eye();
  //----------------------------------------------------------------------------
public:
  //----------------------------------------------------------------------------
  /// defaults to unit hypersphere
  constexpr hyper_ellipse()
      : m_center{vec_type::zeros()}, m_S{mat_type::eye()} {}
  //----------------------------------------------------------------------------
  constexpr hyper_ellipse(hyper_ellipse const &) = default;
  constexpr hyper_ellipse(hyper_ellipse &&) noexcept = default;
  //----------------------------------------------------------------------------
  constexpr auto operator=(hyper_ellipse const &) -> hyper_ellipse & = default;
  constexpr auto operator=(hyper_ellipse &&) noexcept
      -> hyper_ellipse & = default;
  //----------------------------------------------------------------------------
  ~hyper_ellipse() = default;
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radius.
  constexpr hyper_ellipse(Real const radius) : m_S{mat_type::eye() * radius} {}
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radius and origin point.
  constexpr hyper_ellipse(Real const radius, vec_type const &center)
      : m_center{center}, m_S{mat_type::eye() * radius} {}
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radius and origin point.
  constexpr hyper_ellipse(vec_type const &center, Real const radius)
      : m_center{center}, m_S{mat_type::eye() * radius} {}
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radius and origin point.
  constexpr hyper_ellipse(fixed_size_vec<NumDimensions> auto const &center,
                          fixed_size_quadratic_mat<NumDimensions> auto const &S)
      : m_center{center}, m_S{S} {}
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radii.
  constexpr hyper_ellipse(vec_type const &center,
                          arithmetic auto const... radii)
    requires(sizeof...(radii) > 0)
      : m_center{center}, m_S{diag(vec{static_cast<Real>(radii)...})} {
    static_assert(sizeof...(radii) == NumDimensions,
                  "Number of radii does not match number of dimensions.");
  }
  //----------------------------------------------------------------------------
  /// Sets up a sphere with specified radii.
  constexpr hyper_ellipse(arithmetic auto const... radii)
    requires(sizeof...(radii) > 1)
      : m_center{pos_type::zeros()},
        m_S(diag(vec{static_cast<Real>(radii)...})) {
    static_assert(sizeof...(radii) == NumDimensions,
                  "Number of radii does not match number of dimensions.");
  }
  //----------------------------------------------------------------------------
  /// Fits an ellipse through specified points.
  constexpr hyper_ellipse(fixed_size_vec<NumDimensions> auto const &...points)
    requires(sizeof...(points) == NumDimensions)
  {
    fit(points...);
  }
  //----------------------------------------------------------------------------
  /// Fits an ellipse through specified points
  constexpr hyper_ellipse(
      fixed_size_quadratic_mat<NumDimensions> auto const &H) {
    fit(H);
  }
  //============================================================================
  auto S() const -> auto const & { return m_S; }
  auto S() -> auto & { return m_S; }
  //----------------------------------------------------------------------------
  auto center() const -> auto const & { return m_center; }
  auto center() -> auto & { return m_center; }
  //----------------------------------------------------------------------------
  auto center(std::size_t const i) const { return m_center(i); }
  auto center(std::size_t const i) -> auto & { return m_center(i); }
  //----------------------------------------------------------------------------
  auto local_coordinate(pos_type const &x) const {
    return *solve(S(), (x - center()));
  }
  //----------------------------------------------------------------------------
  auto squared_euclidean_distance_to_center(pos_type const &x) const {
    return squared_euclidean_distance(x, center());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto euclidean_distance_to_center(pos_type const &x) const {
    return distance(x, center());
  }
  //----------------------------------------------------------------------------
  auto squared_local_euclidean_distance_to_center(pos_type const &x) const {
    return squared_euclidean_length(local_coordinate(x));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto local_distance_to_center(pos_type const &x) const {
    return euclidean_length(local_coordinate(x));
  }
  //----------------------------------------------------------------------------
  /// Computes euclidean distance to nearest boundary point
  constexpr auto distance_to_boundary(pos_type const &x) const {
    auto const x_local = local_coordinate(x);
    auto const local_distance_to_point = euclidian_length(x_local);
    auto const local_point_on_boundary = x_local / local_distance_to_point;
    auto const local_offset_to_boundary = x_local - local_point_on_boundary;
    return euclidian_length(m_S * local_offset_to_boundary);
  }
  //----------------------------------------------------------------------------
  auto local_nearest_point_boundary(pos_type const &x) const {
    auto const local_point_on_boundary = normalize(local_coordinate(x));
    return local_point_on_boundary;
  }
  //----------------------------------------------------------------------------
  auto nearest_point_boundary(pos_type const &x) const {
    return S() * local_nearest_point_boundary(x) + center();
  }
  //============================================================================
private:
  //----------------------------------------------------------------------------
  /// Fits an ellipse through specified points
  template <std::size_t... Is>
  constexpr auto fit(std::index_sequence<Is...> /*seq*/,
                     fixed_size_vec<NumDimensions> auto const &...points) {
    auto H = mat_type{};
    ([&] { H.col(Is) = points; }(), ...);
    fit(H);
  }
  //----------------------------------------------------------------------------
public:
  //----------------------------------------------------------------------------
  /// Fits an ellipse through specified points
  constexpr auto fit(fixed_size_vec<NumDimensions> auto const &...points) {
    static_assert(sizeof...(points) == NumDimensions,
                  "Number of points does not match number of dimensions.");
    fit(std::make_index_sequence<NumDimensions>{}, points...);
  }
  //----------------------------------------------------------------------------
  /// Fits an ellipse through columns of H
  /// \returns main axes
  constexpr auto fit(fixed_size_quadratic_mat<NumDimensions> auto const &H) {
    auto const HHt = H * transposed(H);
    auto const [Q, Sig] = eigenvectors_sym(HHt);
    m_S = Q * sqrt(diag(Sig)) * transposed(Q);
  }
  //============================================================================
  /// Computes the main axes of the ellipse.
  /// \returns main axes
  template <typename V, typename VReal>
  constexpr auto nearest_point_on_boundary(
      base_tensor<V, VReal, NumDimensions> const &x) const {
    return m_S * normalize(*solve(m_S, x - m_center)) + m_center;
  }
  //----------------------------------------------------------------------------
  /// Checks if a point x is inside the ellipse.
  /// \param x point to check
  /// \returns true if x is inside ellipse.
  constexpr auto is_inside(pos_type const &x) const {
    return squared_euclidean_length(*solve(m_S, x - m_center)) <= 1;
  }
  //----------------------------------------------------------------------------
  auto discretize(std::size_t const num_vertices = 32) const
    requires(NumDimensions == 2)
  {
    using namespace std::ranges;
    auto radial = linspace<Real>{0.0, M_PI * 2, num_vertices + 1};
    radial.pop_back();

    auto discretization = line<Real, 2>{};
    auto radian_to_cartesian = [](auto const t) {
      return vec{std::cos(t), std::sin(t)};
    };
    auto out_it = std::back_inserter(discretization);
    copy(radial | views::transform(radian_to_cartesian), out_it);
    discretization.set_closed(true);
    for (auto const v : discretization.vertices()) {
      discretization[v] = S() * discretization[v] + center();
    }
    return discretization;
  }
  //----------------------------------------------------------------------------
  auto discretize(std::size_t const num_subdivisions = 2) const
    requires(NumDimensions == 3)
  {
    using grid = tatooine::unstructured_triangular_grid<Real, 3>;
    using vh = typename grid::vertex_handle;
    using edge = std::pair<vh, vh>;
    using cell = std::array<vh, 3>;
    using cell_list = std::vector<cell>;
    static constexpr auto X = Real(0.525731112119133606);
    static constexpr auto Z = Real(0.850650808352039932);
    auto g = grid{vec{-X, 0, Z}, vec{X, 0, Z},  vec{-X, 0, -Z}, vec{X, 0, -Z},
                  vec{0, Z, X},  vec{0, Z, -X}, vec{0, -Z, X},  vec{0, -Z, -X},
                  vec{Z, X, 0},  vec{-Z, X, 0}, vec{Z, -X, 0},  vec{-Z, -X, 0}};
    auto triangles = cell_list{
        {vh{0}, vh{4}, vh{1}},  {vh{0}, vh{9}, vh{4}},  {vh{9}, vh{5}, vh{4}},
        {vh{4}, vh{5}, vh{8}},  {vh{4}, vh{8}, vh{1}},  {vh{8}, vh{10}, vh{1}},
        {vh{8}, vh{3}, vh{10}}, {vh{5}, vh{3}, vh{8}},  {vh{5}, vh{2}, vh{3}},
        {vh{2}, vh{7}, vh{3}},  {vh{7}, vh{10}, vh{3}}, {vh{7}, vh{6}, vh{10}},
        {vh{7}, vh{11}, vh{6}}, {vh{11}, vh{0}, vh{6}}, {vh{0}, vh{1}, vh{6}},
        {vh{6}, vh{1}, vh{10}}, {vh{9}, vh{0}, vh{11}}, {vh{9}, vh{11}, vh{2}},
        {vh{9}, vh{2}, vh{5}},  {vh{7}, vh{2}, vh{11}}};

    for (std::size_t i = 0; i < num_subdivisions; ++i) {
      auto subdivided_cells = cell_list{};
      auto subdivided = std::map<edge, std::size_t>{}; // vh index on edge
      for (auto &[v0, v1, v2] : triangles) {
        auto edges = std::array{edge{v0, v1}, edge{v0, v2}, edge{v1, v2}};
        auto nvs = cell{vh{0}, vh{0}, vh{0}};
        auto i = std::size_t{};
        for (auto &edge : edges) {
          auto &[v0, v1] = edge;
          if (v0 < v1) {
            std::swap(v0, v1);
          }
          if (subdivided.find(edge) == end(subdivided)) {
            subdivided[edge] = vertices(g).size();
            nvs[i++] = g.insert_vertex(normalize((g[v0] + g[v1]) * 0.5));
          } else {
            nvs[i++] = vh{subdivided[edge]};
          }
        }
        subdivided_cells.emplace_back(cell{v0, nvs[1], nvs[0]});
        subdivided_cells.emplace_back(cell{nvs[0], nvs[2], v1});
        subdivided_cells.emplace_back(cell{nvs[1], v2, nvs[2]});
        subdivided_cells.emplace_back(cell{nvs[0], nvs[1], nvs[2]});
      }
      triangles = subdivided_cells;
    }
    for (auto v : g.vertices()) {
      g[v] = S() * g[v] + center();
    }
    for (auto const &c : triangles) {
      g.insert_triangle(c[0], c[1], c[2]);
    }
    return g;
  }
  //----------------------------------------------------------------------------
public:
  //----------------------------------------------------------------------------
  /// Returns a the main axes of the hyper ellipse as a matrix and the
  /// lengths of the axes as a vector.
  auto main_axes() const { return eigenvectors_sym(S()); }
  //----------------------------------------------------------------------------
  /// Returns a the radii of the hyper ellipse as a vector.
  auto radii() const { return eigenvalues_sym(S()); }
  //----------------------------------------------------------------------------
  /// Returns a the radii of the hyper ellipse as a vector.
  auto base_coordinate_system() const {
    auto const [axes, lengths] = main_axes();
    return axes * diag(lengths);
  }
};
//------------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions>
  requires(NumDimensions == 2 || NumDimensions == 3)
auto discretize(hyper_ellipse<Real, NumDimensions> const &s,
                std::size_t const n = 32) {
  return s.discretize(n);
}
template <std::size_t NumDimensions>
using HyperEllipse = hyper_ellipse<real_number, NumDimensions>;
//==============================================================================
} // namespace tatooine::geometry
//==============================================================================
namespace tatooine {
namespace detail::geometry::hyper_ellipse {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
using he_t = tatooine::geometry::hyper_ellipse<Real, NumDimensions>;
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
auto ptr_convertible(he_t<Real, NumDimensions> const volatile *)
    -> std::true_type;
auto ptr_convertible(void const volatile *) -> std::false_type;
//------------------------------------------------------------------------------
template <typename> auto is_derived(...) -> std::true_type;
//------------------------------------------------------------------------------
template <typename D>
auto is_derived(int) -> decltype(ptr_convertible(static_cast<D *>(nullptr)));
//------------------------------------------------------------------------------
template <typename T>
struct is_derived_impl
    : std::integral_constant<bool, std::is_class_v<T> &&decltype(is_derived<T>(
                                       0))::value> {};
//==============================================================================
} // namespace detail::geometry::hyper_ellipse
//==============================================================================
template <typename T>
static auto constexpr is_derived_from_hyper_ellipse =
    detail::geometry::hyper_ellipse::is_derived_impl<T>::value;
static_assert(is_derived_from_hyper_ellipse<geometry::HyperEllipse<2>>);
//==============================================================================
} // namespace tatooine
//==============================================================================
namespace tatooine::reflection {
//==============================================================================
// template <typename Real, std::size_t NumDimensions>
// TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
//    (geometry::hyper_ellipse<Real, NumDimensions>),
//    TATOOINE_REFLECTION_INSERT_METHOD(center, center()),
//    TATOOINE_REFLECTION_INSERT_METHOD(S, S()))
//==============================================================================
} // namespace tatooine::reflection
//==============================================================================
#include <tatooine/geometry/ellipse.h>
#include <tatooine/geometry/ellipsoid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <range Ellipsoids>
  requires(is_derived_from_hyper_ellipse<
              std::ranges::range_value_t<Ellipsoids>>) &&
          (std::ranges::range_value_t<Ellipsoids>::num_dimensions() == 2)
auto write_vtp(Ellipsoids const &ellipsoids, filesystem::path const &path) {
  using namespace std::ranges;
  using ellipsoid_t = std::ranges::range_value_t<Ellipsoids>;
  using real_t = typename ellipsoid_t::real_type;
  auto file = std::ofstream{path, std::ios::binary};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not write " + path.string()};
  }
  auto appended_data_offset = std::size_t{};
  using header_type = std::uint64_t;
  using connectivity_int = std::int32_t;
  using offset_int = connectivity_int;
  file << "<VTKFile"
       << " type=\"PolyData\""
       << " version=\"1.0\" "
          "byte_order=\"LittleEndian\""
       << " header_type=\"" << vtk::xml::to_data_type<header_type>() << "\">";
  file << "<PolyData>\n";
  auto discretized_unit_circle =
      discretize(geometry::hyper_ellipse<real_t, 2>{}, 512);
  discretized_unit_circle.set_closed(true);
  auto transformed_circle = discretized_unit_circle;
  auto const num_points = discretized_unit_circle.vertices().size();
  auto const num_lines = discretized_unit_circle.num_line_segments();

  auto offsets = std::vector<offset_int>(num_lines, 2);
  offsets.front() = 0;
  for (std::size_t i = 2; i < size(offsets); ++i) {
    offsets[i] += offsets[i - 1];
  }

  auto connectivity_data = std::vector<connectivity_int>{};
  connectivity_data.reserve(num_lines * 2);
  for (connectivity_int i = 0; i < static_cast<connectivity_int>(
                                       discretized_unit_circle.num_vertices()) -
                                       1;
       ++i) {
    connectivity_data.push_back(i);
    connectivity_data.push_back(i + 1);
  }
  if (discretized_unit_circle.is_closed()) {
    connectivity_data.push_back(static_cast<connectivity_int>(
        discretized_unit_circle.num_vertices() - 1));
    connectivity_data.push_back(0);
  }

  auto const num_bytes_points = header_type(3 * num_points * sizeof(real_t));
  auto const num_bytes_connectivity =
      header_type(num_lines * 2 * sizeof(connectivity_int));
  auto const num_bytes_offsets =
      header_type(offsets.size() * sizeof(offset_int));
  for (std::size_t i = 0; i < size(ellipsoids); ++i) {
    file << "<Piece"
         << " NumberOfPoints=\"" << num_points << "\""
         << " NumberOfLines=\"" << num_lines << "\""
         << ">\n";

    auto const points_base =
        num_bytes_offsets + num_bytes_connectivity + sizeof(header_type) * 2;
    // Points
    file << "<Points>\n";
    file << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << points_base + appended_data_offset << "\""
         << " type=\"" << vtk::xml::to_data_type<real_t>()
         << "\" NumberOfComponents=\"" << 3 << "\"/>\n";
    appended_data_offset += num_bytes_points + sizeof(header_type);
    file << "</Points>\n";

    // Lines
    file << "<Lines>\n";
    // Lines - connectivity
    file << "<DataArray format=\"appended\" offset=\"0\" type =\""
         << vtk::xml::to_data_type<connectivity_int>()
         << "\" Name=\"connectivity\"/>\n";

    // Lines - offsets
    file << "<DataArray format=\"appended\" offset=\""
         << num_bytes_connectivity + sizeof(header_type) << "\" type =\""
         << vtk::xml::to_data_type<offset_int>() << "\" Name =\"offsets\"/>\n";
    file << "</Lines>\n";
    file << "</Piece>\n";
  }
  file << "</PolyData>\n";
  file << "<AppendedData encoding=\"raw\">_";

  // Writing lines connectivity data to appended data section
  file.write(reinterpret_cast<char const *>(&num_bytes_connectivity),
             sizeof(header_type));
  file.write(reinterpret_cast<char const *>(connectivity_data.data()),
             num_bytes_connectivity);

  // Writing lines offsets to appended data section
  file.write(reinterpret_cast<char const *>(&num_bytes_offsets),
             sizeof(header_type));
  file.write(reinterpret_cast<char const *>(offsets.data()), num_bytes_offsets);

  // Writing vertex data to appended data section
  for (auto const &ellipsoid : ellipsoids) {
    for (auto v : discretized_unit_circle.vertices()) {
      transformed_circle[v] =
          ellipsoid.center() + ellipsoid.S() * discretized_unit_circle[v];
    }

    // Writing points
    file.write(reinterpret_cast<char const *>(&num_bytes_points),
               sizeof(header_type));
    auto zero = real_t{};
    for (auto const v : transformed_circle.vertices()) {
      file.write(
          reinterpret_cast<char const *>(transformed_circle.at(v).data()),
          sizeof(real_t) * 2);
      file.write(reinterpret_cast<char const *>(&zero), sizeof(real_t));
    }
  }
  file << "</AppendedData>";
  file << "</VTKFile>";
}
//==============================================================================
template <range Ellipsoids>
  requires(is_derived_from_hyper_ellipse<
              std::ranges::range_value_t<Ellipsoids>>) &&
          (std::ranges::range_value_t<Ellipsoids>::num_dimensions() == 3)
auto write_vtp(Ellipsoids const &ellipsoids, filesystem::path const &path) {
  using namespace std::ranges;
  using ellipsoid_t = std::ranges::range_value_t<Ellipsoids>;
  using real_t = typename ellipsoid_t::real_type;
  auto file = std::ofstream{path, std::ios::binary};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not write " + path.string()};
  }
  auto offset = std::size_t{};
  using header_type = std::uint64_t;
  using connectivity_int = std::int32_t;
  using offset_int = connectivity_int;
  file << "<VTKFile"
       << " type=\"PolyData\""
       << " version=\"1.0\" "
          "byte_order=\"LittleEndian\""
       << " header_type=\"" << vtk::xml::to_data_type<header_type>() << "\">";
  file << "<PolyData>\n";
  auto const discretized_unit_sphere =
      discretize(geometry::hyper_ellipse<real_t, 3>{}, 2);
  auto transformed_ellipse = discretized_unit_sphere;
  auto const num_points = discretized_unit_sphere.vertices().size();
  auto const num_polys = discretized_unit_sphere.triangles().size();

  auto offsets = std::vector<offset_int>(num_polys, 3);
  offsets.front() = 0;
  for (std::size_t i = 1; i < size(offsets); ++i) {
    offsets[i] += offsets[i - 1];
  };

  auto index = [](auto const handle) -> connectivity_int {
    return handle.index();
  };
  auto connectivity_data = std::vector<connectivity_int>(num_polys * 3);
  copy(discretized_unit_sphere.simplices().data_container() |
           views::transform(index),
       begin(connectivity_data));

  auto const num_bytes_connectivity = num_polys * 3 * sizeof(connectivity_int);
  auto const num_bytes_offsets = sizeof(offset_int) * offsets.size();
  for (std::size_t i = 0; i < size(ellipsoids); ++i) {
    file << "<Piece"
         << " NumberOfPoints=\"" << num_points << "\""
         << " NumberOfPolys=\"" << num_polys << "\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\"0\""
         << " NumberOfStrips=\"0\""
         << ">\n";

    // Points
    file << "<Points>";
    file << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\"" << vtk::xml::to_data_type<real_t>()
         << "\" NumberOfComponents=\"" << 3 << "\"/>";
    auto const num_bytes_points = header_type(sizeof(real_t) * 3 * num_points);
    offset += num_bytes_points + sizeof(header_type);
    file << "</Points>\n";

    // Polys
    file << "<Polys>\n";
    // Polys - connectivity
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::to_data_type<connectivity_int>()
         << "\" Name=\"connectivity\"/>\n";
    offset += num_bytes_connectivity + sizeof(header_type);
    // Polys - offsets
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::to_data_type<offset_int>() << "\" Name=\"offsets\"/>\n";
    offset += num_bytes_offsets + sizeof(header_type);
    file << "</Polys>\n";
    file << "</Piece>\n";
  }
  file << "</PolyData>\n";
  file << "<AppendedData encoding=\"raw\">_";
  // Writing vertex data to appended data section
  for (auto const &ellipsoid : ellipsoids) {
    for (auto v : discretized_unit_sphere.vertices()) {
      transformed_ellipse[v] =
          ellipsoid.center() + ellipsoid.S() * discretized_unit_sphere[v];
    }
    auto const num_bytes_points = header_type(sizeof(real_t) * 3 * num_points);

    // Writing points
    file.write(reinterpret_cast<char const *>(&num_bytes_points),
               sizeof(header_type));
    for (auto const v : transformed_ellipse.vertices()) {
      file.write(
          reinterpret_cast<char const *>(transformed_ellipse.at(v).data()),
          sizeof(real_t) * 3);
    }

    // Writing polys connectivity data to appended data section
    file.write(reinterpret_cast<char const *>(&num_bytes_connectivity),
               sizeof(header_type));
    file.write(reinterpret_cast<char const *>(connectivity_data.data()),
               num_bytes_connectivity);

    // Writing polys offsets to appended data section
    file.write(reinterpret_cast<char const *>(&num_bytes_offsets),
               sizeof(header_type));
    file.write(reinterpret_cast<char const *>(offsets.data()),
               num_bytes_offsets);
  }
  file << "</AppendedData>";
  file << "</VTKFile>";
}
//==============================================================================
} // namespace tatooine
//==============================================================================
#endif
