#ifndef TATOOINE_POINTSET_H
#define TATOOINE_POINTSET_H
//==============================================================================
#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/algorithm/find.hpp>
#include <flann/flann.hpp>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <vector>

#include <tatooine/dynamic_tensor.h>
#include <tatooine/handle.h>
#include <tatooine/property.h>
#include <tatooine/polynomial.h>
#include <tatooine/tensor.h>
#include <tatooine/type_traits.h>
#include <tatooine/vtk_legacy.h>

// #include "tetgen_inc.h"
// #include "triangle_inc.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct pointset {
  // static constexpr size_t triangle_dims = 2;
  // static constexpr size_t tetgen_dims = 3;
  static constexpr auto num_dimensions() { return N; }
  using real_t = Real;
  using this_t = pointset<Real, N>;
  using pos_t  = vec<Real, N>;
  using flann_index_t = flann::Index<flann::L2<Real>>;
  //----------------------------------------------------------------------------
  struct vertex_handle : handle {
    using handle::handle;
    using handle::operator=;
    bool operator==(vertex_handle other) const { return this->i == other.i; }
    bool operator!=(vertex_handle other) const { return this->i != other.i; }
    bool operator<(vertex_handle other) const { return this->i < other.i; }
    static constexpr auto invalid() {
      return vertex_handle{handle::invalid_idx};
    }
  };
  //----------------------------------------------------------------------------
  struct vertex_iterator
      : boost::iterator_facade<vertex_iterator, vertex_handle,
                               boost::bidirectional_traversal_tag,
                               vertex_handle> {
    vertex_iterator(vertex_handle _v, pointset const* _ps) : v{_v}, ps{_ps} {}
    vertex_iterator(vertex_iterator const& other) : v{other.v}, ps{other.ps} {}

   private:
    vertex_handle    v;
    pointset const* ps;

    friend class boost::iterator_core_access;

    void increment() {
      do
        ++v;
      while (!ps->is_valid(v));
    }
    void decrement() {
      do
        --v;
      while (!ps->is_valid(v));
    }

    auto equal(vertex_iterator const& other) const { return v == other.v; }
    auto dereference() const { return v; }
  };
  //----------------------------------------------------------------------------
  struct vertex_container {
    using iterator       = vertex_iterator;
    using const_iterator = vertex_iterator;
    //==========================================================================
    pointset const* m_pointset;
    //==========================================================================
    auto begin() const {
      vertex_iterator vi{vertex_handle{0}, m_pointset};
      if (!m_pointset->is_valid(*vi)) ++vi;
      return vi;
    }
    //--------------------------------------------------------------------------
    auto end() const {
      return vertex_iterator{vertex_handle{m_pointset->m_vertices.size()},
                             m_pointset};
    }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  using vertex_property_t = vector_property_impl<vertex_handle, T>;
  using vertex_property_container_t =
      std::map<std::string, std::unique_ptr<vector_property<vertex_handle>>>;
  //============================================================================
 private:
  std::vector<pos_t>                             m_vertices;
  std::vector<vertex_handle>                     m_invalid_vertices;
  vertex_property_container_t                    m_vertex_properties;
  mutable std::unique_ptr<flann_index_t>         m_kd_tree;
  //============================================================================
 public:
  pointset() = default;
  //----------------------------------------------------------------------------
  pointset(std::initializer_list<pos_t>&& vertices)
      : m_vertices(std::move(vertices)) {}
  //----------------------------------------------------------------------------
  // #ifdef USE_TRIANGLE
  //   pointset(const triangle::io& io) {
  //     for (int i = 0; i < io.numberofpoints; ++i)
  //       insert_vertex(io.pointlist[i * 2], io.pointlist[i * 2 + 1]);
  //   }
  // #endif

  // template <typename = void>
  // requires (N == 3)
  // pointset(const tetgenio& io) {
  //   for (int i = 0; i < io.numberofpoints; ++i)
  //     insert_vertex(io.pointlist[i * 3], io.pointlist[i * 3 + 1],
  //                   io.pointlist[i * 3 + 2]);
  // }
  //----------------------------------------------------------------------------
  pointset(pointset const& other)
      : m_vertices(other.m_vertices),
        m_invalid_vertices(other.m_invalid_vertices) {
    m_vertex_properties.clear();
    for (auto const& [name, prop] : other.m_vertex_properties)
      m_vertex_properties.insert(std::pair{name, prop->clone()});
  }
  //----------------------------------------------------------------------------
  pointset(pointset&& other)
      : m_vertices(std::move(other.m_vertices)),
        m_invalid_vertices(std::move(other.m_invalid_vertices)),
        m_vertex_properties(std::move(other.m_vertex_properties)) {}
  //----------------------------------------------------------------------------
  auto operator=(pointset const& other) -> pointset& {
    m_vertex_properties.clear();
    m_vertices         = other.m_vertices;
    m_invalid_vertices = other.m_invalid_vertices;
    for (auto const& [name, prop] : other.m_vertex_properties) {
      m_vertex_properties.emplace(name, prop->clone());
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(pointset&& other) noexcept -> pointset& = default;
  //----------------------------------------------------------------------------
  auto vertex_properties() const -> auto const& { return m_vertex_properties; }
  //----------------------------------------------------------------------------
  auto at(vertex_handle const v) -> auto& { return m_vertices[v.i]; }
  auto at(vertex_handle const v) const -> auto const& { return m_vertices[v.i]; }
  //----------------------------------------------------------------------------
  auto vertex_at(vertex_handle const v) -> auto& { return m_vertices[v.i]; }
  auto vertex_at(vertex_handle const v) const -> auto const& {
    return m_vertices[v.i];
  }
  //----------------------------------------------------------------------------
  auto vertex_at(size_t const i) -> auto& { return m_vertices[i]; }
  auto vertex_at(size_t const i) const -> auto const& { return m_vertices[i]; }
  //----------------------------------------------------------------------------
  auto operator[](vertex_handle const v) -> auto& { return at(v); }
  auto operator[](vertex_handle const v) const -> auto const& { return at(v); }
  //----------------------------------------------------------------------------
  auto vertices() const { return vertex_container{this}; }
  //----------------------------------------------------------------------------
  auto vertex_data() -> auto& { return m_vertices; }
  auto vertex_data() const -> auto const& { return m_vertices; }
  //----------------------------------------------------------------------------
  template <arithmetic... Ts>
  requires(sizeof...(Ts) == N)
  auto insert_vertex(Ts const... ts) {
    m_vertices.push_back(pos_t{static_cast<Real>(ts)...});
    for (auto& [key, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{size(m_vertices) - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t const& v) {
    m_vertices.push_back(v);
    for (auto& [key, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{size(m_vertices) - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& v) {
    m_vertices.emplace_back(std::move(v));
    for (auto& [key, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{size(m_vertices) - 1};
  }
  //----------------------------------------------------------------------------
  /// tidies up invalid vertices
  void tidy_up() {
    for (auto const v : m_invalid_vertices) {
      // decrease deleted vertex indices;
      for (auto& v_to_decrease : m_invalid_vertices)
        if (v_to_decrease.i > v.i) --v_to_decrease.i;

      m_vertices.erase(m_vertices.begin() + v.i);
      for (auto const& [key, prop] : m_vertex_properties) {
        prop->erase(v.i);
      }
    }
    m_invalid_vertices.clear();
  }
  //----------------------------------------------------------------------------
  void remove(vertex_handle v) {
    if (is_valid(v) &&
        boost::find(m_invalid_vertices, v) == m_invalid_vertices.end())
      m_invalid_vertices.push_back(v);
  }

  //----------------------------------------------------------------------------
  constexpr auto is_valid(vertex_handle v) const -> bool {
    return boost::find(m_invalid_vertices, v) == m_invalid_vertices.end();
  }

  //----------------------------------------------------------------------------
  auto clear_vertices() {
    m_vertices.clear();
    m_vertices.shrink_to_fit();
    m_invalid_vertices.clear();
    m_invalid_vertices.shrink_to_fit();
    for (auto& [key, val] : m_vertex_properties)
      val->clear();
  }
  auto clear() { clear_vertices(); }

  //----------------------------------------------------------------------------
  auto num_vertices() const {
    return m_vertices.size() - m_invalid_vertices.size();
  }
  //----------------------------------------------------------------------------
  auto join(this_t const& other) {
    for (auto v : other.vertices()) {
      insert_vertex(other[v]);
    }
  }
  //----------------------------------------------------------------------------
  auto find_duplicates(Real eps = 1e-6) {
    std::vector<std::pair<vertex_handle, vertex_handle>> duplicates;
    for (auto v0 = vertices().begin(); v0 != vertices().end(); ++v0)
      for (auto v1 = next(v0); v1 != vertices().end(); ++v1)
        if (approx_equal(at(v0), at(v1), eps)) duplicates.emplace_back(v0, v1);

    return duplicates;
  }

  //#ifdef USE_TRIANGLE
  //  //----------------------------------------------------------------------------
  //  template <typename = void>
  //  requires (N == triangle_dims>>
  //  auto to_triangle_io() const {
  //    triangle::io in;
  //    size_t       i    = 0;
  //    in.numberofpoints = num_vertices();
  //    in.pointlist      = new triangle::Real[in.numberofpoints *
  //    triangle_dims]; for (auto v : vertices()) {
  //      for (size_t j = 0; j < triangle_dims; ++j) {
  //        in.pointlist[i++] = at(v)(j);
  //      }
  //    }
  //
  //    return in;
  //  }
  //
  //  //----------------------------------------------------------------------------
  //  template <typename vertex_cont_t>
  //  requires (N == triangle_dims)
  //  auto to_triangle_io(vertex_cont_t const& vertices) const {
  //    triangle::io in;
  //    size_t       i    = 0;
  //    in.numberofpoints = num_vertices();
  //    in.pointlist      = new triangle::Real[in.numberofpoints *
  //    triangle_dims]; for (auto v : vertices()) {
  //      for (size_t j = 0; j < triangle_dims; ++j) {
  //        in.pointlist[i++] = at(v)(j);
  //      }
  //    }
  //
  //    return in;
  //  }
  //#endif

  //----------------------------------------------------------------------------
  // template <typename = void>
  // requires (N == tetgen_dims)
  // void to_tetgen_io(tetgenio& in) const {
  //   size_t i           = 0;
  //   in.numberofpoints  = num_vertices();
  //   in.pointlist       = new tetgen::Real[in.numberofpoints * tetgen_dims];
  //   in.pointmarkerlist = new int[in.numberofpoints];
  //   in.numberofpointattributes = 1;
  //   in.pointattributelist =
  //       new tetgen::Real[in.numberofpoints * in.numberofpointattributes];
  //   for (auto v : vertices()) {
  //     for (size_t j = 0; j < tetgen_dims; ++j) {
  //       in.pointlist[i * 3 + j] = at(v)(j);
  //     }
  //     in.pointmarkerlist[i]    = i;
  //     in.pointattributelist[i] = v.i;
  //     ++i;
  //   }
  // }

  //----------------------------------------------------------------------------
  /// using specified vertices of point_set
  // template <typename vertex_cont_t>
  // auto to_tetgen_io(vertex_cont_t const& vertices) const {
  //   tetgenio io;
  //   size_t       i    = 0;
  //   io.numberofpoints = vertices.size();
  //   io.pointlist      = new tetgen_real_t[io.numberofpoints * 3];
  //   for (auto v : vertices) {
  //     auto const& x       = at(v);
  //     io.pointlist[i]     = x(0);
  //     io.pointlist[i + 1] = x(1);
  //     io.pointlist[i + 2] = x(2);
  //     i += 2;
  //   }
  //
  //   return io;
  // }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property(std::string const& name) -> auto& {
    auto prop        = m_vertex_properties.at(name).get();
    auto casted_prop = dynamic_cast<vertex_property_t<T>*>(prop);
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property(std::string const& name) const -> const auto& {
    return *dynamic_cast<vertex_property_t<T>*>(
        m_vertex_properties.at(name).get());
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto add_vertex_property(std::string const& name, T const& value = T{})
      -> auto& {
    auto [it, suc] = m_vertex_properties.insert(
        std::pair{name, std::make_unique<vertex_property_t<T>>(value)});
    auto prop = dynamic_cast<vertex_property_t<T>*>(it->second.get());
    prop->resize(m_vertices.size());
    return *prop;
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires(N == 3 || N == 2)
  auto write_vtk(std::string const& path,
                 std::string const& title = "Tatooine pointset") {
    vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      std::vector<std::array<Real, 3>> points;
      points.reserve(m_vertices.size());
      for (auto const& p : m_vertices) {
        if constexpr (N == 3) {
          points.push_back({p(0), p(1), p(2)});
        } else {
          points.push_back({p(0), p(1), 0});
        }
        writer.write_points(points);
        writer.write_point_data(m_vertices.size());
        for (auto const& [name, prop] : m_vertex_properties) {
          std::vector<std::vector<Real>> data;
          data.reserve(m_vertices.size());

          if (prop->type() == typeid(vec<Real, 4>)) {
            for (auto const& v4 :
                 *dynamic_cast<vertex_property_t<vec<Real, 4>> const*>(
                     prop.get())) {
              data.push_back({v4(0), v4(1), v4(2), v4(3)});
            }
          } else if (prop->type() == typeid(vec<Real, 3>)) {
            for (auto const& v3 :
                 *dynamic_cast<vertex_property_t<vec<Real, 3>> const*>(
                     prop.get())) {
              data.push_back({v3(0), v3(1), v3(2)});
            }
          } else if (prop->type() == typeid(vec<Real, 2>)) {
            for (auto const& v2 :
                 *dynamic_cast<vertex_property_t<vec<Real, 2>> const*>(
                     prop.get())) {
              data.push_back({v2(0), v2(1)});
            }
          } else if (prop->type() == typeid(Real)) {
            for (auto const& scalar :
                 *dynamic_cast<vertex_property_t<Real> const*>(prop.get())) {
              data.push_back({scalar});
            }
          }
          writer.write_scalars(name, data);
        }
        writer.close();
      }
    }
  }
  auto rebuild_kd_tree() {
    m_kd_tree.reset();
    kd_tree();
  }
  //----------------------------------------------------------------------------
 private:
  auto kd_tree() const -> auto& {
    if (m_kd_tree == nullptr) {
      flann::Matrix<double> dataset{
          const_cast<Real*>(m_vertices.front().data_ptr()), num_vertices(),
          num_dimensions()};
      m_kd_tree = std::make_unique<flann_index_t>(
          dataset, flann::KDTreeSingleIndexParams{});
      m_kd_tree->buildIndex();
    }
    return *m_kd_tree;
  }
  //----------------------------------------------------------------------------
 public:
  auto nearest_neighbor(pos_t const& x) const {
    flann::Matrix<Real>            qm{const_cast<Real*>(x.data_ptr()), 1,
                           num_dimensions()};
    std::vector<std::vector<int>>  indices;
    std::vector<std::vector<Real>> distances;
    flann::SearchParams            params;
    kd_tree().knnSearch(qm, indices, distances, 1, params);
    return vertex_handle{static_cast<size_t>(indices.front().front())};
  }
  //----------------------------------------------------------------------------
  /// Takes the raw output indices of flann without converting them into vertex
  /// handles.
  auto nearest_neighbors_raw(pos_t const& x, size_t const num_nearest_neighbors,
                             flann::SearchParams const params = {}) const
      -> std::pair<std::vector<int>, std::vector<Real>> {
    flann::Matrix<Real>            qm{const_cast<Real*>(x.data_ptr()), 1,
                           num_dimensions()};
    std::vector<std::vector<int>>  indices;
    std::vector<std::vector<Real>> distances;
    kd_tree().knnSearch(qm, indices, distances, num_nearest_neighbors, params);
    return {std::move(indices.front()), std::move(distances.front())};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto nearest_neighbors(pos_t const& x,
                         size_t const num_nearest_neighbors) const {
    auto const [indices, distances] =
        nearest_neighbors_raw(x, num_nearest_neighbors);
    std::vector<vertex_handle> handles;
    handles.reserve(size(indices));
    for (auto const i : indices) {
      handles.emplace_back(static_cast<size_t>(i));
    }
    return handles;
  }
  //----------------------------------------------------------------------------
  /// Takes the raw output indices of flann without converting them into vertex
  /// handles.
  auto nearest_neighbors_radius_raw(pos_t const& x, Real const radius,
                                    flann::SearchParams const params = {}) const
      -> std::pair<std::vector<int>, std::vector<Real>> {
    flann::Matrix<Real>            qm{const_cast<Real*>(x.data_ptr()), 1,
                           num_dimensions()};
    std::vector<std::vector<int>>  indices;
    std::vector<std::vector<Real>> distances;
    kd_tree().radiusSearch(qm, indices, distances, radius, params);
    return {std::move(indices.front()), std::move(distances.front())};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto nearest_neighbors_radius(pos_t const& x, Real const radius) const {
    auto const [indices, distances] = nearest_neighbors_radius_raw(x, radius);
    std::vector<vertex_handle> handles;
    handles.reserve(size(indices));
    for (auto const i : indices) {
      handles.emplace_back(static_cast<size_t>(i));
    }
    return handles;
  }
  //============================================================================
  template <typename T>
  auto inverse_distance_weighting_sampler(std::string const& prop_name) const {
    return inverse_distance_weighting_sampler_t<T>{
        *this, vertex_property<T>(prop_name)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T>
  auto inverse_distance_weighting_sampler(
      vertex_property_t<T> const& prop) const {
    return inverse_distance_weighting_sampler_t<T>{*this, prop};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T>
  auto inverse_distance_weighting_sampler(std::string const& prop_name,
                                          Real const         radius) const {
    return inverse_distance_weighting_sampler_t<T>{
        *this, vertex_property<T>(prop_name), radius};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T>
  auto inverse_distance_weighting_sampler(vertex_property_t<T> const& prop,
                                          Real const radius) const {
    return inverse_distance_weighting_sampler_t<T>{*this, prop, radius};
  }
  //============================================================================
  template <typename T>
  struct inverse_distance_weighting_sampler_t {
    using this_t     = inverse_distance_weighting_sampler_t;
    using pointset_t = pointset<Real, N>;
    //==========================================================================
    pointset_t const&           m_pointset;
    vertex_property_t<T> const& m_property;
    Real                        m_radius = 1;
    //==========================================================================
    inverse_distance_weighting_sampler_t(pointset_t const&           ps,
                                         vertex_property_t<T> const& property)
        : m_pointset{ps}, m_property{property} {}
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    inverse_distance_weighting_sampler_t(pointset_t const&           ps,
                                         vertex_property_t<T> const& property,
                                         Real const radius)
        : m_pointset{ps},
          m_property{property},
          m_radius{radius} {}
    //--------------------------------------------------------------------------
    inverse_distance_weighting_sampler_t(
        inverse_distance_weighting_sampler_t const&) = default;
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    inverse_distance_weighting_sampler_t(
        inverse_distance_weighting_sampler_t&&) noexcept = default;
    //--------------------------------------------------------------------------
    auto operator=(inverse_distance_weighting_sampler_t const&)
        -> inverse_distance_weighting_sampler_t& = default;
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    auto operator=(inverse_distance_weighting_sampler_t&&) noexcept
        -> inverse_distance_weighting_sampler_t& = default;
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ~inverse_distance_weighting_sampler_t()      = default;
    //==========================================================================
    auto sample(pos_t const& x) const {
      auto [indices, distances] =
          m_pointset.nearest_neighbors_radius_raw(x, m_radius);
      if (indices.empty()) {
        throw std::runtime_error{
            "[inverse_distance_weighting_sampler] out of domain"};
      }
      T    accumulated_prop_val{};
      Real accumulated_weight = 0;

      auto index_it = begin(indices);
      auto dist_it  = begin(distances);
      for (; index_it != end(indices); ++index_it, ++dist_it) {
        auto const& property_value = m_property[vertex_handle{*index_it}];
        if (*dist_it == 0) {
          return property_value;
        };
        auto const  weight         = 1 / *dist_it;
        accumulated_prop_val += property_value * weight;
        accumulated_weight += weight;
      }
      return accumulated_prop_val / accumulated_weight;
    }
    template <arithmetic... Components>
    requires(sizeof...(Components) == N)
    auto sample(Components const... components) const {
      return sample(pos_t{components...});
    }
    template <arithmetic... Components>
    requires(sizeof...(Components) == N)
    auto operator()(Components const... components) const {
      return sample(pos_t{components...});
    }
    auto operator()(pos_t const& x) const { return sample(x); }
  };
  //============================================================================
  template <typename T>
  requires (num_dimensions() == 2) || (num_dimensions() == 3)
  auto moving_least_squares_sampler(std::string const& prop_name) const {
    return moving_least_squares_sampler_t<T>{*this,
                                             vertex_property<T>(prop_name)};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  requires (num_dimensions() == 2) || (num_dimensions() == 3)
  auto moving_least_squares_sampler(vertex_property_t<T> const& prop) const {
    return moving_least_squares_sampler_t<T>{*this, prop};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  requires (num_dimensions() == 2) || (num_dimensions() == 3)
  auto moving_least_squares_sampler(std::string const& prop_name,
                                    Real const         radius) const {
    return moving_least_squares_sampler_t<T>{
        *this, vertex_property<T>(prop_name), radius};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  requires (num_dimensions() == 2) || (num_dimensions() == 3)
  auto moving_least_squares_sampler(vertex_property_t<T> const& prop,
                                    Real const                  radius) const {
    return moving_least_squares_sampler_t<T>{*this, prop, radius};
  }
  //============================================================================
  template <typename T>
  struct moving_least_squares_sampler_t {
    using this_t     = moving_least_squares_sampler_t;
    using pointset_t = pointset<Real, N>;
    //==========================================================================
    pointset_t const&           m_pointset;
    vertex_property_t<T> const& m_property;
    Real                        m_radius = 1;
    //==========================================================================
    moving_least_squares_sampler_t(pointset_t const&           ps,
                                   vertex_property_t<T> const& property)
        : m_pointset{ps}, m_property{property} {}
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    moving_least_squares_sampler_t(pointset_t const&           ps,
                                   vertex_property_t<T> const& property,
                                   Real const                  radius)
        : m_pointset{ps}, m_property{property}, m_radius{radius} {}
    //--------------------------------------------------------------------------
    moving_least_squares_sampler_t(moving_least_squares_sampler_t const&) =
        default;
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    moving_least_squares_sampler_t(moving_least_squares_sampler_t&&) noexcept =
        default;
    //--------------------------------------------------------------------------
    auto operator=(moving_least_squares_sampler_t const&)
        -> moving_least_squares_sampler_t& = default;
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    auto operator=(moving_least_squares_sampler_t&&) noexcept
        -> moving_least_squares_sampler_t& = default;
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ~moving_least_squares_sampler_t() = default;
    //==========================================================================
    template <typename = void>
    requires (num_dimensions() == 2) || (num_dimensions() == 3)
    auto sample(pos_t const& q) const {
      if constexpr (num_dimensions() == 2) {
        return sample2d(q);
      } else if constexpr (num_dimensions() == 3) {
        return sample3d(q);
      };
    }

   private:
    template <typename = void>
    requires(num_dimensions() == 2)
    auto sample2d(pos_t const& q) const -> T {
      auto const  nn = m_pointset.nearest_neighbors_radius_raw(q, m_radius);
      auto const& indices       = nn.first;
      auto const& distances     = nn.second;
      auto const  num_neighbors = size(indices);

      if (num_neighbors == 0) {
        throw std::runtime_error{"ood"};
      }
      if (num_neighbors == 1) {
        return m_property[vertex_handle{indices[0]}];
      }

      auto w = dynamic_tensor<Real>::zeros(num_neighbors);
      auto F = dynamic_tensor<Real>::zeros(num_neighbors, num_components<T>);
      auto B = [&] {
        if (num_neighbors >= 10) {
          return dynamic_tensor<Real>::ones(num_neighbors, 10);
        }
        if (num_neighbors >= 6) {
          return dynamic_tensor<Real>::ones(num_neighbors, 6);
        }
        if (num_neighbors >= 3) {
          return dynamic_tensor<Real>::ones(num_neighbors, 3);
        }
        return dynamic_tensor<Real>::ones(1, 1);
      }();

      // build w
      for (size_t i = 0; i < num_neighbors; ++i) {
        if (distances[i] == 0) {
          return m_property[vertex_handle{indices[i]}];
        }
        w(i) = 1 / distances[i] - 1 / m_radius;
      }
      // build F
      for (size_t i = 0; i < num_neighbors; ++i) {
        if constexpr (num_components<T> == 1) {
          F(i, 0) = m_property[vertex_handle{indices[i]}];
        } else {
          for (size_t j = 0; j < num_components<T>; ++j) {
            F(i, j) = m_property[vertex_handle{indices[i]}](j);
          }
        }
      }
      // build B
      if (num_neighbors >= 3) {
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 1) = m_pointset.vertex_at(indices[i]).x() - q.x();
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 2) = m_pointset.vertex_at(indices[i]).y() - q.y();
        }
        }
        if (num_neighbors >= 6) {
          for (size_t i = 0; i < num_neighbors; ++i) {
            B(i, 3) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                      (m_pointset.vertex_at(indices[i]).x() - q.x());
          }
          for (size_t i = 0; i < num_neighbors; ++i) {
            B(i, 4) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                      (m_pointset.vertex_at(indices[i]).y() - q.y());
          }
          for (size_t i = 0; i < num_neighbors; ++i) {
            B(i, 5) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                      (m_pointset.vertex_at(indices[i]).y() - q.y());
          }
        }
        if (num_neighbors >= 10) {
          for (size_t i = 0; i < num_neighbors; ++i) {
            B(i, 6) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                      (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                      (m_pointset.vertex_at(indices[i]).x() - q.x());
          }
          for (size_t i = 0; i < num_neighbors; ++i) {
            B(i, 7) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                      (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                      (m_pointset.vertex_at(indices[i]).y() - q.y());
          }
          for (size_t i = 0; i < num_neighbors; ++i) {
            B(i, 8) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                      (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                      (m_pointset.vertex_at(indices[i]).y() - q.y());
          }
          for (size_t i = 0; i < num_neighbors; ++i) {
            B(i, 9) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                      (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                      (m_pointset.vertex_at(indices[i]).y() - q.y());
          }
        }
        auto const BtW = transposed(B) * diag(w);

        if constexpr (num_components<T> == 1) {
          return solve(BtW * B, BtW * F)(0);
        } else {
          T    ret{};
          auto C = solve(BtW * B, BtW * F);
          for (size_t i = 0; i < num_components<T>; ++i) {
            ret(i) = C(0, i);
          }
          return ret;
        }
      }
    template <typename = void>
    requires(num_dimensions() == 3)
    auto sample3d(pos_t const& q) const -> T {
      auto const  nn = m_pointset.nearest_neighbors_radius_raw(q, m_radius);
      auto const& indices       = nn.first;
      auto const& distances     = nn.second;
      auto const  num_neighbors = size(indices);
      if (num_neighbors == 0) {
        throw std::runtime_error{"ood"};
      }
      if (num_neighbors == 1) {
        return m_property[vertex_handle{indices[0]}];
      }

      auto w = dynamic_tensor<Real>::zeros(num_neighbors);
      auto F = dynamic_tensor<Real>::zeros(num_neighbors, num_components<T>);
      auto B = [&] {
        if (num_neighbors >= 20) {
          return dynamic_tensor<Real>::ones(num_neighbors, 20);
        } else if (num_neighbors >= 10) {
          return dynamic_tensor<Real>::ones(num_neighbors, 10);
        } else if (num_neighbors >= 4) {
          return dynamic_tensor<Real>::ones(num_neighbors, 4);
        }
        return dynamic_tensor<Real>::ones(1, 1);
      }();
      // build w
      for (size_t i = 0; i < num_neighbors; ++i) {
        if (distances[i] == 0) {
          return m_property[vertex_handle{indices[i]}];
        }
        w(i) = 1 / distances[i] - 1 / m_radius;
      }
      // build f
      for (size_t i = 0; i < num_neighbors; ++i) {
        if constexpr (num_components<T> == 1) {
          F(i, 0) = m_property[vertex_handle{indices[i]}];
        } else {
          for (size_t j = 0; j < num_components<T>; ++j) {
            F(i, j) = m_property[vertex_handle{indices[i]}](j);
          }
        }
      }
      // build B
      if (num_neighbors >= 4) {
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 1) = m_pointset.vertex_at(indices[i]).x() - q.x();
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 2) = m_pointset.vertex_at(indices[i]).y() - q.y();
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 3) = m_pointset.vertex_at(indices[i]).z() - q.z();
        }
      }
      if (num_neighbors >= 10) {
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 4) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                    (m_pointset.vertex_at(indices[i]).x() - q.x());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 5) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                    (m_pointset.vertex_at(indices[i]).y() - q.y());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 6) = (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                    (m_pointset.vertex_at(indices[i]).z() - q.z());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 7) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                    (m_pointset.vertex_at(indices[i]).y() - q.y());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 8) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                    (m_pointset.vertex_at(indices[i]).z() - q.z());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 9) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                    (m_pointset.vertex_at(indices[i]).z() - q.z());
        }
      }
      if (num_neighbors >= 20) {
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 10) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                     (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                     (m_pointset.vertex_at(indices[i]).x() - q.x());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 11) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                     (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                     (m_pointset.vertex_at(indices[i]).y() - q.y());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 12) = (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                     (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                     (m_pointset.vertex_at(indices[i]).z() - q.z());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 13) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                     (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                     (m_pointset.vertex_at(indices[i]).y() - q.y());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 14) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                     (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                     (m_pointset.vertex_at(indices[i]).z() - q.z());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 15) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                     (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                     (m_pointset.vertex_at(indices[i]).y() - q.y());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 16) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                     (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                     (m_pointset.vertex_at(indices[i]).z() - q.z());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 17) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                     (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                     (m_pointset.vertex_at(indices[i]).z() - q.z());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 18) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                     (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                     (m_pointset.vertex_at(indices[i]).z() - q.z());
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
          B(i, 19) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                     (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                     (m_pointset.vertex_at(indices[i]).z() - q.z());
        }
      }
      auto const BtW = transposed(B) * diag(w);
      if constexpr (num_components<T> == 1) {
        return solve(BtW * B, BtW * F)(0);
      } else {
        T    ret{};
        auto C = solve(BtW * B, BtW * F);
        for (size_t i = 0; i < num_components<T>; ++i) {
          ret(i) = C(0, i);
        }
        return ret;
      }
    }

   public:
    template <arithmetic... Components>
    requires(sizeof...(Components) ==
             N) auto sample(Components const... components) const {
      return sample(pos_t{components...});
    }
    template <arithmetic... Components>
    requires(sizeof...(Components) == N) auto operator()(
        Components const... components) const {
      return sample(pos_t{components...});
    }
    auto operator()(pos_t const& x) const { return sample(x); }
  };
};
//==============================================================================
template <typename Real, size_t N>
auto begin(typename pointset<Real, N>::vertex_container const& verts) {
  return verts.begin();
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto end(typename pointset<Real, N>::vertex_container const& verts) {
  return verts.end();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
