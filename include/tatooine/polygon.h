#ifndef TATOOINE_POLYGON_H
#define TATOOINE_POLYGON_H
//==============================================================================
#include <tatooine/vec.h>
#include <tatooine/vtk_legacy.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/generate.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct polygon {
  using this_t = polygon<Real, N>;
  using pos_t  = vec<Real, N>;
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return N; }
  //----------------------------------------------------------------------------
 private:
  std::vector<pos_t> m_vertices{};
  //----------------------------------------------------------------------------
 public:
  template <size_t NumVertices>
  explicit polygon(std::array<pos_t, NumVertices> const& vertices)
      : m_vertices(begin(vertices), end(vertices)) {}
  explicit polygon(std::vector<pos_t> vertices)
      : m_vertices(std::move(vertices)) {}
  polygon(std::initializer_list<pos_t>&& vertices)
      : m_vertices(begin(vertices), end(vertices)) {}
  polygon(polygon const&)     = default;
  polygon(polygon&&) noexcept = default;
  auto operator=(polygon const&) -> polygon& = default;
  auto operator=(polygon&&) noexcept -> polygon& = default;
  //----------------------------------------------------------------------------
  auto num_vertices() const { return size(m_vertices); }
  auto vertices() const -> auto const& { return m_vertices; }
  auto vertex(size_t const i) const -> auto const& { return m_vertices[i]; }
  //----------------------------------------------------------------------------
  auto previous_index(auto const i) const -> size_t {
    if (i == 0) {
      return num_vertices() - 1;
    }
    return i - 1;
  };
  //----------------------------------------------------------------------------
  auto next_index(auto const i) const -> size_t {
    if (i == num_vertices() - 1) {
      return 0;
    }
    return i + 1;
  };
  //----------------------------------------------------------------------------
  /// Creates a meta functor that iterates and calls its underlying function f.
  template <typename F>
  auto indexed_with_neighbors(F&& f) const -> decltype(auto) {
    return [this, &f, i = size_t(0)]() mutable -> decltype(auto) {
      decltype(auto) return_value = f(previous_index(i), i, next_index(i));
      ++i;
      return return_value;
    };
  };
  //----------------------------------------------------------------------------
  /// Creates a meta functor that iterates and calls its underlying function f.
  template <typename F>
  auto indexed(F&& f) const -> decltype(auto) {
    return [&f, i = size_t(0)]() mutable -> decltype(auto) {
      decltype(auto) return_value = f(i);
      ++i;
      return return_value;
    };
  };
  //----------------------------------------------------------------------------
  /// Implementation of \cite Hormann2005BarycentricCF.
  auto barycentric_coordinates(pos_t const& query_point) const
      -> std::vector<real_t> {
    using namespace boost;
    using namespace boost::adaptors;
    using scalar_list           = std::vector<real_t>;
    using pos_list              = std::vector<pos_t>;
    static real_t constexpr eps = 1e-8;

    auto const nv = num_vertices();
    // create data fields
    auto weights        = scalar_list(nv, 0);  // weighting of each vertex
    auto triangle_areas = scalar_list(nv, 0);  // area of v, v_i, v_(i+1)
    auto dot_products   = scalar_list(nv, 0);  //
    auto distances_to_vertices = scalar_list(nv, 0);
    auto direction_to_vertices = pos_list(nv, {0, 0});

    // formula to compute weight of one vertex
    auto weight_of_vertex = [&distances_to_vertices, &triangle_areas,
                             &dot_products](auto const prev_idx,
                                            auto const cur_idx,
                                            auto const next_idx) -> real_t {
      auto weight = real_t(0);
      if (std::abs(triangle_areas[prev_idx]) > eps) {
        weight += (distances_to_vertices[prev_idx] -
                   dot_products[prev_idx] / distances_to_vertices[cur_idx]) /
                  triangle_areas[prev_idx];
      }
      if (std::abs(triangle_areas[cur_idx]) > eps) {
        weight += (distances_to_vertices[next_idx] -
                   dot_products[cur_idx] / distances_to_vertices[cur_idx]) /
                  triangle_areas[cur_idx];
      }
      return weight;
    };

    auto accumulated_weights = real_t(0);
    auto calculate_and_accumulate_weights =
        [&accumulated_weights, &weight_of_vertex](
            auto const prev_idx, auto const cur_idx, auto const next_idx) {
          auto const weight = weight_of_vertex(prev_idx, cur_idx, next_idx);
          accumulated_weights += weight;
          return weight;
        };
    auto calculate_direction_to_vertex = indexed(
        [this, &query_point](auto const i) { return vertex(i) - query_point; });

    // compute all directions to vertices
    generate(direction_to_vertices, calculate_direction_to_vertex);

    // compute distance to vertex and check if query_point is on a vertex or on
    // an edge
    for (size_t i = 0; i < nv; ++i) {
      auto const next_idx = next_index(i);
      distances_to_vertices[i] = length(direction_to_vertices[i]);
      if (std::abs(distances_to_vertices[i]) <= eps) {
        weights[i] = 1;
        return weights;
      }
      triangle_areas[i] =
          direction_to_vertices[i][0] * direction_to_vertices[next_idx][1] -
          direction_to_vertices[i][1] * direction_to_vertices[next_idx][0] / 2;
      dot_products[i] =
          dot(direction_to_vertices[i], direction_to_vertices[next_idx]);

      if (std::abs(triangle_areas[i]) <= eps && dot_products[i] <= eps) {
        distances_to_vertices[next_idx] =
            length(direction_to_vertices[next_idx]);
        real_t const norm =
            1 / (distances_to_vertices[i] + distances_to_vertices[next_idx]);
        weights[i]        = distances_to_vertices[next_idx] * norm;
        weights[next_idx] = distances_to_vertices[i] * norm;
        return weights;
      }
    }

    // compute actual weights of vertices
    generate(weights, indexed_with_neighbors(calculate_and_accumulate_weights));

    // normalize weights to make the sum of the weights = 1
    auto normalize_w = [inverse_accumulated_weights = 1 / accumulated_weights](
                           auto& w) { return w * inverse_accumulated_weights; };
    copy(weights | transformed(normalize_w), begin(weights));
    return weights;
  }
  //------------------------------------------------------------------------------
  auto write_vtk(filesystem::path const& path) const {
    using boost::copy;
    using boost::adaptors::transformed;
    auto writer = vtk::legacy_file_writer{path, vtk::dataset_type::polydata};
    if (writer.is_open()) {
      writer.set_title("");
      writer.write_header();
      if constexpr (num_dimensions() == 2) {
        auto three_dims = [](vec<Real, 2> const& v2) {
          return vec<Real, 3>{v2(0), v2(1), 0};
        };
        std::vector<vec<Real, 3>> v3s(num_vertices());
        auto                      three_dimensional = transformed(three_dims);
        copy(m_vertices | three_dimensional, begin(v3s));
        writer.write_points(v3s);

      } else if constexpr (num_dimensions() == 3) {
        writer.write_points(m_vertices);
      }

      std::vector<std::vector<size_t>> lines;
      auto& line = lines.emplace_back();
      line.reserve(m_vertices.size() * 2);
      for (size_t i = 0; i < num_vertices(); ++i) {
        line.push_back(i);
        line.push_back(next_index(i));
      }
      writer.write_lines(lines);

      writer.close();
      return true;
    }
    return false;
  }
};
//==============================================================================
template <size_t N>
using Polygon  = polygon<real_t, N>;
using polygon2 = Polygon<2>;
using polygon3 = Polygon<3>;
using polygon4 = Polygon<4>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
