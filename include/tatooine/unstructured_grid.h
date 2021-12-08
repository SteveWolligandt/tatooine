#ifndef TATOOINE_UNSTRUCTURED_GRID_H
#define TATOOINE_UNSTRUCTURED_GRID_H
//==============================================================================
#include <tatooine/pointset.h>

#include <boost/range.hpp>
#include <boost/range/adaptor/transformed.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t NumDimensions>
struct unstructured_grid : pointset<Real, NumDimensions> {
 public:
  using this_t   = unstructured_grid;
  using parent_t = pointset<Real, NumDimensions>;
  using real_t   = Real;
  using typename parent_t::pos_t;
  using typename parent_t::vertex_handle;
  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  static constexpr auto num_dimensions() { return NumDimensions; }

  struct cell_vertex_iterator
      : boost::iterator_facade<cell_vertex_iterator, vertex_handle,
                               boost::bidirectional_traversal_tag,
                               vertex_handle> {
    using this_t = cell_vertex_iterator;
    using grid_t = unstructured_grid;
    cell_vertex_iterator(std::vector<size_t>::const_iterator it) : m_it{it} {}
    cell_vertex_iterator(cell_vertex_iterator const& other)
        : m_it{other.m_it} {}

   private:
    std::vector<size_t>::const_iterator m_it;

    friend class boost::iterator_core_access;

    auto increment() { ++m_it; }
    auto decrement() { --m_it; }

    auto equal(cell_vertex_iterator const& other) const {
      return m_it == other.m_it;
    }
    auto dereference() const { return vertex_handle{*m_it}; }
  };
  struct cell_iterator;
  struct cell {
   public:
    friend struct cell_iterator;
    using this_t         = cell;
    using grid_t         = unstructured_grid;
    using iterator       = cell_vertex_iterator;
    using const_iterator = iterator;

   private:
    // [number of vertices of cell 0, [vertex indices of cell 0],
    // [number of vertices of cell 1, [vertex indices of cell 1],
    // ...,
    // [number of vertices of cell n-1, [vertex indices of cell n-1],
    // number of vertices of cell n-1 (this is for getting the last cell
    // index)
    grid_t const*                       m_grid;
    std::vector<size_t>::const_iterator m_cell_begin;

   public:
    cell(grid_t const* grid, std::vector<size_t>::const_iterator cell_begin)
        : m_grid{grid}, m_cell_begin{cell_begin} {}
    auto size() const { return *m_cell_begin; }
    auto at(size_t const i) const {
      return vertex_handle{*next(m_cell_begin, i + 1)};
    }
    auto operator[](size_t const i) const { return at(i); }
    auto begin() const { return const_iterator{next(m_cell_begin)}; }
    auto end() const { return const_iterator{next(m_cell_begin, size() + 1)}; }
    //----------------------------------------------------------------------------
    /// Checks if a point x is inside the cell.
    /// From here: https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    auto is_inside(pos_t const& x) const requires(NumDimensions == 2) {
      bool c = false;
      for (size_t i = 0, j = size() - 1; i < size(); j = i++) {
        auto const& xi = m_grid->at(at(i));
        auto const& xj = m_grid->at(at(j));
        if (((xi(1) > x(1)) != (xj(1) > x(1))) &&
            (x(0) < (xj(0) - xi(0)) * (x(1) - xi(1)) / (xj(1) - xi(1)) + xi(0)))
          c = !c;
      }
      return c;
    }
    //----------------------------------------------------------------------------
    /// \brief Computes generalized barycentric coordinates for arbitrary
    /// polygons.
    ///
    /// This is an implementation of \"\a Barycentric \a coordinates \a for \a
    /// arbitrary  \a polygons \a in \a the \a plane\" by Hormann
    /// \cite Hormann2005BarycentricCF.
    auto barycentric_coordinates(pos_t const& query_point) const
        requires(NumDimensions == 2) {
      // some typedefs and namespaces
      using namespace boost;
      using namespace boost::adaptors;
      using scalar_list = std::vector<real_t>;
      using pos_list    = std::vector<pos_t>;

      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      // create data fields
      static real_t constexpr eps           = 1e-8;
      auto const num_vertices               = size();
      auto       vertex_weights             = scalar_list(num_vertices, 0);
      auto       accumulated_vertex_weights = real_t(0);
      auto       triangle_areas             = scalar_list(num_vertices, 0);
      auto       dot_products               = scalar_list(num_vertices, 0);
      auto       distances_to_vertices      = scalar_list(num_vertices, 0);
      auto       direction_to_vertices      = pos_list(num_vertices, {0, 0});

      //------------------------------------------------------------------------
      auto previous_index = [num_vertices](auto const i) -> size_t {
        return i == 0 ? num_vertices - 1 : i - 1;
      };
      //------------------------------------------------------------------------
      auto next_index = [num_vertices](auto const i) -> size_t {
        return i == num_vertices - 1 ? 0 : i + 1;
      };
      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      // create functors

      // Creates a meta functor that iterates and calls its underlying function
      // f.
      auto indexed_with_neighbors = [&](auto&& f) -> decltype(auto) {
        return [&f, i = size_t(0), &previous_index,
                &next_index]() mutable -> decltype(auto) {
          decltype(auto) ret_val = f(previous_index(i), i, next_index(i));
          ++i;
          return ret_val;
        };
      };

      // Creates a meta functor that iterates and calls its underlying function
      // f.
      auto indexed = [&](auto&& f) -> decltype(auto) {
        return [&f, i = size_t(0)]() mutable -> decltype(auto) {
          decltype(auto) return_value = f(i);
          ++i;
          return return_value;
        };
      };

      auto calculate_weight_of_vertex = [&distances_to_vertices,
                                         &triangle_areas,
                                         &dot_products](auto const prev_idx,
                                                        auto const cur_idx,
                                                        auto const next_idx) {
        auto weight = real_t(0);
        if (std::abs(triangle_areas[prev_idx]) > eps) /* A != 0 */ {
          weight += (distances_to_vertices[prev_idx] -
                     dot_products[prev_idx] / distances_to_vertices[cur_idx]) /
                    triangle_areas[prev_idx];
        }
        if (std::abs(triangle_areas[cur_idx]) > eps) /* A != 0 */ {
          weight += (distances_to_vertices[next_idx] -
                     dot_products[cur_idx] / distances_to_vertices[cur_idx]) /
                    triangle_areas[cur_idx];
        }
        return weight;
      };

      auto calculate_and_accumulate_vertex_weight =
          [&accumulated_vertex_weights, &calculate_weight_of_vertex](
              auto const prev_idx, auto const cur_idx, auto const next_idx) {
            auto const weight =
                calculate_weight_of_vertex(prev_idx, cur_idx, next_idx);
            accumulated_vertex_weights += weight;
            return weight;
          };

      auto calculate_direction_to_vertex = [this, &query_point](auto const i) {
        return m_grid->at(at(i)) - query_point;
      };

      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      // fill actual data
      generate(direction_to_vertices, indexed(calculate_direction_to_vertex));

      // compute distance to vertex and check if query_point is on a vertex
      // or on an edge
      for (size_t i = 0; i < num_vertices; ++i) {
        auto const next_idx      = next_index(i);
        distances_to_vertices[i] = length(direction_to_vertices[i]);
        if (std::abs(distances_to_vertices[i]) <= eps) {
          vertex_weights[i] = 1;
          return vertex_weights;
        }
        triangle_areas[i] =
            (direction_to_vertices[i](0) * direction_to_vertices[next_idx](1) -
             direction_to_vertices[i](1) * direction_to_vertices[next_idx](0)) /
            2;
        dot_products[i] =
            dot(direction_to_vertices[i], direction_to_vertices[next_idx]);

        if (std::abs(triangle_areas[i]) <= eps && dot_products[i] <= eps) {
          distances_to_vertices[next_idx] =
              length(direction_to_vertices[next_idx]);
          real_t const norm =
              1 / (distances_to_vertices[i] + distances_to_vertices[next_idx]);
          vertex_weights[i]        = distances_to_vertices[next_idx] * norm;
          vertex_weights[next_idx] = distances_to_vertices[i] * norm;
          return vertex_weights;
        }
      }

      // if query point is not on a vertex or an edge of the polygon:
      generate(vertex_weights,
               indexed_with_neighbors(calculate_and_accumulate_vertex_weight));

      // normalize vertex weights to make the sum of the vertex_weights = 1
      auto normalize_w = [inverse_accumulated_weights =
                              1 / accumulated_vertex_weights](auto& w) {
        return w * inverse_accumulated_weights;
      };
      copy(vertex_weights | transformed(normalize_w), vertex_weights.begin());
      return vertex_weights;
    }
  };
  //----------------------------------------------------------------------------
  struct cell_iterator
      : boost::iterator_facade<cell_iterator, cell,
                               boost::forward_traversal_tag, cell const&> {
    using this_t = cell_iterator;
    using grid_t = unstructured_grid;
    cell_iterator(cell c) : m_cell{std::move(c)} {}
    cell_iterator(cell_iterator const& other) : m_cell{other.m_cell} {}

   private:
    cell m_cell;

    friend class boost::iterator_core_access;

    auto increment() { advance(m_cell.m_cell_begin, *m_cell.m_cell_begin + 1); }

    auto equal(cell_iterator const& other) const {
      return m_cell.m_cell_begin == other.m_cell.m_cell_begin;
    }
    auto dereference() const -> auto const& { return m_cell; }
  };
  //----------------------------------------------------------------------------
  struct cell_container {
    using iterator       = cell_iterator;
    using const_iterator = cell_iterator;
    using grid_t         = unstructured_grid;
    //--------------------------------------------------------------------------
    grid_t const* m_grid;
    //--------------------------------------------------------------------------
    auto begin() const {
      return cell_iterator{cell{m_grid, m_grid->m_cell_indices.begin()}};
    }
    //--------------------------------------------------------------------------
    auto end() const {
      return cell_iterator{cell{m_grid, prev(m_grid->m_cell_indices.end())}};
    }
    //--------------------------------------------------------------------------
    auto size() const { return m_grid->m_num_cells; }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  struct barycentric_coordinates_vertex_property_sampler_t
      : field<barycentric_coordinates_vertex_property_sampler_t<T>, Real,
              parent_t::num_dimensions(), T> {
   private:
    using grid_t = unstructured_grid<Real, NumDimensions>;
    using this_t = barycentric_coordinates_vertex_property_sampler_t<T>;

    grid_t const&               m_grid;
    vertex_property_t<T> const& m_prop;
    //--------------------------------------------------------------------------
   public:
    barycentric_coordinates_vertex_property_sampler_t(
        grid_t const& grid, vertex_property_t<T> const& prop)
        : m_grid{grid}, m_prop{prop} {}
    //--------------------------------------------------------------------------
    auto grid() const -> auto const& { return m_grid; }
    auto property() const -> auto const& { return m_prop; }
    //--------------------------------------------------------------------------
    [[nodiscard]] auto evaluate(pos_t const& x, real_t const /*t*/) const -> T {
      for (auto const& cell : m_grid.cells()) {
        if (cell.is_inside(x)) {
          auto b   = cell.barycentric_coordinates(x);
          T    acc = 0;
          for (size_t i = 0; i < b.size(); ++i) {
            acc += b[i] * property()[cell[i]];
          }
          return acc;
        }
      }
      return T{Real(0) / Real(0)};
    }
  };
  using parent_t::at;
  using parent_t::vertex_data;
  using parent_t::vertex_properties;
  using parent_t::vertices;

 private:
  //------------------------------------------------------------------------------
  std::vector<size_t> m_cell_indices{};
  size_t              m_num_cells = 0;

 public:
  unstructured_grid() { m_cell_indices.push_back(0); }
  //------------------------------------------------------------------------------
  auto cells() const { return cell_container{this}; }
  //------------------------------------------------------------------------------
  template <typename... Handles>
  auto insert_cell(Handles const... handles) {
    m_cell_indices.back() = sizeof...(handles);
    (
        [this](auto const h) {
          using handle_t = std::decay_t<decltype(h)>;
          if constexpr (is_same<vertex_handle, handle_t>) {
            m_cell_indices.push_back(h.i);
          } else if constexpr (is_integral<handle_t>) {
            m_cell_indices.push_back(h);
          }
        }(handles),
        ...);
    m_cell_indices.push_back(sizeof...(handles));
    ++m_num_cells;
    // for (auto& [key, prop] : m_cell_properties) {
    //  prop->push_back();
    //}
    // return cell_handle{cells().size() - 1};
  }
  //------------------------------------------------------------------------------
  auto insert_cell(std::vector<vertex_handle> const& handles) {
    using boost::copy;
    using boost::adaptors::transformed;
    m_cell_indices.back() = size(handles);
    copy(handles | transformed([](auto const handle) { return handle.i; }),
         std::back_inserter(m_cell_indices));
    m_cell_indices.push_back(size(handles));
    ++m_num_cells;
    // for (auto& [key, prop] : m_cell_properties) {
    //  prop->push_back();
    //}
    return cell{this, prev(m_cell_indices.end(), m_cell_indices.back() + 2)};
  }
  //------------------------------------------------------------------------------
  auto write(filesystem::path const& path) const {
    if (path.extension() == ".vtk") {
      write_vtk(path);
      return;
    }
    throw std::runtime_error{
        "[unstructured_grid::write()]\n  unknown file extension: " +
        path.extension().string()};
  }
  //----------------------------------------------------------------------------
  template <typename T, typename... Ts, typename Prop, typename Name,
            typename Writer>
  auto write_vtk_vertex_property(Prop const& prop, Name const& name,
                                 Writer& writer) const {
    if (prop->type() == typeid(T)) {
      auto const& casted_prop = *dynamic_cast<
          typename parent_t::template vertex_property_t<T> const*>(prop.get());
      writer.write_scalars(name, casted_prop.data());
      return;
    }
    if constexpr (sizeof...(Ts) > 0) {
      write_vtk_vertex_property<Ts...>(prop, name, writer);
    }
  }
  //----------------------------------------------------------------------------
  auto write_vtk(filesystem::path const& path) const {
    using boost::copy;
    using boost::make_iterator_range;
    using boost::adaptors::transformed;
    auto writer =
        vtk::legacy_file_writer{path, vtk::dataset_type::unstructured_grid};
    if (writer.is_open()) {
      writer.set_title("");
      writer.write_header();
      if constexpr (num_dimensions() == 2) {
        auto three_dims = [this](auto const handle) {
          auto const& v2 = this->at(handle);
          return vec<Real, 3>{v2(0), v2(1), 0};
        };
        auto v3s = std::vector<vec<Real, 3>>(vertices().size());
        copy(vertices() | transformed(three_dims), begin(v3s));
        writer.write_points(v3s);

      } else if constexpr (num_dimensions() == 3) {
        writer.write_points(vertex_data());
      }

      std::vector<std::vector<size_t>> vertices_per_cell;
      // vertices_per_cell.reserve(cells().size());
      std::vector<vtk::cell_type> cell_types(cells().size(),
                                             vtk::cell_type::polygon);
      for (auto const& cell : cells()) {
        auto const num_vertices_of_cell = cell.size();
        auto&      polygon              = vertices_per_cell.emplace_back();
        polygon.reserve(num_vertices_of_cell);
        copy(cell | transformed([](auto const handle) { return handle.i; }),
             std::back_inserter(polygon));
      }
      writer.write_cells(vertices_per_cell);
      writer.write_cell_types(cell_types);

      // write vertex data
      writer.write_point_data(vertices().size());
      for (auto const& [name, prop] : vertex_properties()) {
        write_vtk_vertex_property<float, vec2f, vec3f, vec4f, double, vec2d,
                                  vec3d, vec4d>(prop, name, writer);
      }

      writer.close();
      return true;
    }
    return false;
  }

  //----------------------------------------------------------------------------
  template <typename T>
  auto sampler(vertex_property_t<T> const& prop) const {
    return barycentric_coordinates_vertex_property_sampler_t<T>{*this, prop};
  }
  //--------------------------------------------------------------------------
  template <typename T>
  auto vertex_property_sampler(std::string const& name) const {
    return sampler<T>(this->template vertex_property<T>(name));
  }
};
//==============================================================================
template <size_t NumDimensions>
using UnstructuredGrid = unstructured_grid<real_t, NumDimensions>;
template <size_t NumDimensions>
using UnstructuredGridF = unstructured_grid<float, NumDimensions>;
template <size_t NumDimensions>
using UnstructuredGridD   = unstructured_grid<double, NumDimensions>;
using unstructured_grid2  = UnstructuredGrid<2>;
using unstructured_grid2f = UnstructuredGridF<2>;
using unstructured_grid2d = UnstructuredGridD<2>;
using unstructured_grid3  = UnstructuredGrid<3>;
using unstructured_grid3f = UnstructuredGridF<3>;
using unstructured_grid3d = UnstructuredGridD<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
