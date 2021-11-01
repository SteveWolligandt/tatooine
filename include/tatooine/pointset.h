#ifndef TATOOINE_POINTSET_H
#define TATOOINE_POINTSET_H
//==============================================================================
#include <tatooine/available_libraries.h>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/algorithm/find.hpp>
#if TATOOINE_FLANN_AVAILABLE
#include <flann/flann.hpp>
#endif
#include <tatooine/field.h>
#include <tatooine/handle.h>
#include <tatooine/polynomial.h>
#include <tatooine/property.h>
#include <tatooine/tensor.h>
#include <tatooine/type_traits.h>
#include <tatooine/vtk_legacy.h>

#include <fstream>
#include <limits>
#include <unordered_set>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t NumDimensions, typename T>
struct moving_least_squares_sampler_t;
//==============================================================================
template <typename Real, size_t NumDimensions, typename T>
struct inverse_distance_weighting_sampler_t;
//==============================================================================
template <typename Real, size_t NumDimensions>
struct pointset {
  // static constexpr size_t triangle_dims = 2;
  // static constexpr size_t tetgen_dims = 3;
  static constexpr auto num_dimensions() { return NumDimensions; }
  using real_t = Real;
  using this_t = pointset<Real, NumDimensions>;
  using pos_t  = vec<Real, NumDimensions>;
#if TATOOINE_FLANN_AVAILABLE
  using flann_index_t = flann::Index<flann::L2<Real>>;
#endif
  //----------------------------------------------------------------------------
  struct vertex_handle : handle<vertex_handle> {
    using handle<vertex_handle>::handle;
    using handle<vertex_handle>::operator=;
  };
  //----------------------------------------------------------------------------
  struct vertex_iterator
      : boost::iterator_facade<vertex_iterator, vertex_handle,
                               boost::bidirectional_traversal_tag,
                               vertex_handle> {
    vertex_iterator(vertex_handle _v, pointset const* _ps) : v{_v}, ps{_ps} {}
    vertex_iterator(vertex_iterator const& other) : v{other.v}, ps{other.ps} {}

   private:
    vertex_handle   v;
    pointset const* ps;

    friend class boost::iterator_core_access;

    auto increment() -> void {
      do
        ++v;
      while (!ps->is_valid(v));
    }
    auto decrement() -> void {
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
      if (!m_pointset->is_valid(*vi)) {
        ++vi;
      }
      return vi;
    }
    //--------------------------------------------------------------------------
    auto end() const {
      return vertex_iterator{vertex_handle{m_pointset->m_vertex_positions.size()},
                             m_pointset};
    }
    //--------------------------------------------------------------------------
    auto size() const {
      return m_pointset->m_vertex_positions.size() -
             m_pointset->m_invalid_vertices.size();
    }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  using vertex_property_t = vector_property_impl<vertex_handle, T>;
  using vertex_property_container_t =
      std::map<std::string, std::unique_ptr<vector_property<vertex_handle>>>;
  //============================================================================
 private:
  std::vector<pos_t>          m_vertex_positions;
  std::vector<vertex_handle>  m_invalid_vertices;
  vertex_property_container_t m_vertex_properties;
#if TATOOINE_FLANN_AVAILABLE
  mutable std::unique_ptr<flann_index_t> m_kd_tree;
#endif
  //============================================================================
 public:
  pointset()  = default;
  ~pointset() = default;
  //----------------------------------------------------------------------------
  pointset(std::initializer_list<pos_t>&& vertices)
      : m_vertex_positions(std::move(vertices)) {}
  //----------------------------------------------------------------------------
  // #ifdef USE_TRIANGLE
  //   pointset(const triangle::io& io) {
  //     for (int i = 0; i < io.numberofpoints; ++i)
  //       insert_vertex(io.pointlist[i * 2], io.pointlist[i * 2 + 1]);
  //   }
  // #endif

  // template <typename = void>
  // requires (NumDimensions == 3)
  // pointset(const tetgenio& io) {
  //   for (int i = 0; i < io.numberofpoints; ++i)
  //     insert_vertex(io.pointlist[i * 3], io.pointlist[i * 3 + 1],
  //                   io.pointlist[i * 3 + 2]);
  // }
  //----------------------------------------------------------------------------
  pointset(pointset const& other)
      : m_vertex_positions(other.m_vertex_positions),
        m_invalid_vertices(other.m_invalid_vertices) {
    m_vertex_properties.clear();
    for (auto const& [name, prop] : other.m_vertex_properties)
      m_vertex_properties.insert(std::pair{name, prop->clone()});
  }
  //----------------------------------------------------------------------------
  pointset(pointset&& other) noexcept
      : m_vertex_positions(std::move(other.m_vertex_positions)),
        m_invalid_vertices(std::move(other.m_invalid_vertices)),
        m_vertex_properties(std::move(other.m_vertex_properties)) {}
  //----------------------------------------------------------------------------
  pointset(std::vector<pos_t> const& vertices) : m_vertex_positions(vertices) {}
  //----------------------------------------------------------------------------
  pointset(std::vector<pos_t>&& vertices) : m_vertex_positions(std::move(vertices)) {}
  //----------------------------------------------------------------------------
  auto operator=(pointset const& other) -> pointset& {
    if (&other == this) {
      return *this;
    }
    m_vertex_properties.clear();
    m_vertex_positions         = other.m_vertex_positions;
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
  auto at(vertex_handle const v) -> auto& { return m_vertex_positions[v.i]; }
  auto at(vertex_handle const v) const -> auto const& {
    return m_vertex_positions[v.i];
  }
  //----------------------------------------------------------------------------
  auto vertex_at(vertex_handle const v) -> auto& { return m_vertex_positions[v.i]; }
  auto vertex_at(vertex_handle const v) const -> auto const& {
    return m_vertex_positions[v.i];
  }
  //----------------------------------------------------------------------------
  auto vertex_at(size_t const i) -> auto& { return m_vertex_positions[i]; }
  auto vertex_at(size_t const i) const -> auto const& { return m_vertex_positions[i]; }
  //----------------------------------------------------------------------------
  auto operator[](vertex_handle const v) -> auto& { return at(v); }
  auto operator[](vertex_handle const v) const -> auto const& { return at(v); }
  //----------------------------------------------------------------------------
  auto vertices() const { return vertex_container{this}; }
  //----------------------------------------------------------------------------
  auto vertex_positions() -> auto& { return m_vertex_positions; }
  auto vertex_positions() const -> auto const& { return m_vertex_positions; }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <arithmetic... Ts>
  requires(sizeof...(Ts) == NumDimensions)
#else
  template <typename... Ts, enable_if<is_arithmetic<Ts...>> = true,
            enable_if<sizeof...(Ts) == NumDimensions> = true>
#endif
      auto insert_vertex(Ts const... ts) {
    m_vertex_positions.push_back(pos_t{static_cast<Real>(ts)...});
    for (auto& [key, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{size(m_vertex_positions) - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t const& v) {
    m_vertex_positions.push_back(v);
    for (auto& [key, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{size(m_vertex_positions) - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& v) {
    m_vertex_positions.emplace_back(std::move(v));
    for (auto& [key, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{size(m_vertex_positions) - 1};
  }
  //----------------------------------------------------------------------------
  /// tidies up invalid vertices
  void tidy_up() {
    for (auto const v : m_invalid_vertices) {
      // decrease deleted vertex indices;
      for (auto& v_to_decrease : m_invalid_vertices)
        if (v_to_decrease.i > v.i) --v_to_decrease.i;

      m_vertex_positions.erase(m_vertex_positions.begin() + v.i);
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
    m_vertex_positions.clear();
    m_vertex_positions.shrink_to_fit();
    m_invalid_vertices.clear();
    m_invalid_vertices.shrink_to_fit();
    for (auto& [key, val] : m_vertex_properties)
      val->clear();
  }
  auto clear() { clear_vertices(); }

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
  //----------------------------------------------------------------------------
  auto resize(size_t const s) {
    m_vertex_positions.resize(s);
    for (auto& [key, prop] : m_vertex_properties) {
      prop->resize(s);
    }
  }

  //#ifdef USE_TRIANGLE
  //----------------------------------------------------------------------------
  //  template <typename = void>
  //  requires (NumDimensions == triangle_dims>>
  //  auto to_triangle_io() const {
  //    triangle::io in;
  //    size_t       i    = 0;
  //    in.numberofpoints = vertices().size();
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
  //--------------------------------------------------------------------------
  //  template <typename vertex_cont_t>
  //  requires (NumDimensions == triangle_dims)
  //  auto to_triangle_io(vertex_cont_t const& vertices) const {
  //    triangle::io in;
  //    size_t       i    = 0;
  //    in.numberofpoints = vertices().size();
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
  // requires (NumDimensions == tetgen_dims)
  // void to_tetgen_io(tetgenio& in) const {
  //   size_t i           = 0;
  //   in.numberofpoints  = vertices().size();
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
    if (auto it = m_vertex_properties.find(name);
        it == end(m_vertex_properties)) {
      return insert_vertex_property<T>(name);
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<vertex_property_t<T>*>(
          m_vertex_properties.at(name).get());
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property(std::string const& name) const -> const auto& {
    if (auto it = m_vertex_properties.find(name);
        it == end(m_vertex_properties)) {
      throw std::runtime_error{"property \"" + name + "\" not found"};
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<vertex_property_t<T>*>(
          m_vertex_properties.at(name).get());
    }
  }
  //----------------------------------------------------------------------------
  auto scalar_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<tatooine::real_t>(name);
  }
  //----------------------------------------------------------------------------
  auto scalar_vertex_property(std::string const& name) -> auto& {
    return vertex_property<tatooine::real_t>(name);
  }
  //----------------------------------------------------------------------------
  auto vec2_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<vec2>(name);
  }
  //----------------------------------------------------------------------------
  auto vec2_vertex_property(std::string const& name) -> auto& {
    return vertex_property<vec2>(name);
  }
  //----------------------------------------------------------------------------
  auto vec3_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<vec3>(name);
  }
  //----------------------------------------------------------------------------
  auto vec3_vertex_property(std::string const& name) -> auto& {
    return vertex_property<vec3>(name);
  }
  //----------------------------------------------------------------------------
  auto vec4_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<vec4>(name);
  }
  //----------------------------------------------------------------------------
  auto vec4_vertex_property(std::string const& name) -> auto& {
    return vertex_property<vec4>(name);
  }
  //----------------------------------------------------------------------------
  auto mat2_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<mat2>(name);
  }
  //----------------------------------------------------------------------------
  auto mat2_vertex_property(std::string const& name) -> auto& {
    return vertex_property<mat2>(name);
  }
  //----------------------------------------------------------------------------
  auto mat3_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<mat3>(name);
  }
  //----------------------------------------------------------------------------
  auto mat3_vertex_property(std::string const& name) -> auto& {
    return vertex_property<mat3>(name);
  }
  //----------------------------------------------------------------------------
  auto mat4_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<mat4>(name);
  }
  //----------------------------------------------------------------------------
  auto mat4_vertex_property(std::string const& name) -> auto& {
    return vertex_property<mat4>(name);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto insert_vertex_property(std::string const& name, T const& value = T{})
      -> auto& {
    auto [it, suc] = m_vertex_properties.insert(
        std::pair{name, std::make_unique<vertex_property_t<T>>(value)});
    auto prop = dynamic_cast<vertex_property_t<T>*>(it->second.get());
    prop->resize(m_vertex_positions.size());
    return *prop;
  }
  //----------------------------------------------------------------------------
  auto insert_scalar_vertex_property(
      std::string const&     name,
      tatooine::real_t const value = tatooine::real_t{}) -> auto& {
    return insert_vertex_property<tatooine::real_t>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_vec2_vertex_property(
      std::string const& name, tatooine::vec2 const value = tatooine::vec2{})
      -> auto& {
    return insert_vertex_property<vec2>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_vec3_vertex_property(
      std::string const& name, tatooine::vec3 const value = tatooine::vec3{})
      -> auto& {
    return insert_vertex_property<vec3>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_vec4_vertex_property(
      std::string const& name, tatooine::vec4 const value = tatooine::vec4{})
      -> auto& {
    return insert_vertex_property<vec4>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_mat2_vertex_property(
      std::string const& name, tatooine::mat2 const value = tatooine::mat2{})
      -> auto& {
    return insert_vertex_property<mat2>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_mat3_vertex_property(
      std::string const& name, tatooine::mat3 const value = tatooine::mat3{})
      -> auto& {
    return insert_vertex_property<mat3>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_mat4_vertex_property(
      std::string const& name, tatooine::mat4 const value = tatooine::mat4{})
      -> auto& {
    return insert_vertex_property<mat4>(name, value);
  }
  //----------------------------------------------------------------------------
  auto write(filesystem::path const& path) {
    if constexpr (NumDimensions == 2 || NumDimensions == 3) {
      if (path.extension() == ".vtk") {
        write_vtk(path);
      }
    }
  }
  //----------------------------------------------------------------------------
#ifndef __cpp_concepts
  template <
      size_t NumDimensions_ = NumDimensions,
      enable_if<NumDimensions_ == NumDimensions(NumDimensions_ == 3 ||
                                                NumDimensions_ == 2)> = true>
#endif
  auto write_vtk(filesystem::path const& path,
                 std::string const&      title = "Tatooine pointset") -> void
#ifdef __cpp_concepts
      requires(NumDimensions == 3 || NumDimensions == 2)
#endif
  {
    vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      std::vector<std::array<Real, 3>> points;
      std::vector<std::vector<size_t>> vertex_indices;
      vertex_indices.emplace_back();
      points.reserve(m_vertex_positions.size());
      size_t i = 0;
      for (auto const& p : m_vertex_positions) {
        if constexpr (NumDimensions == 3) {
          points.push_back({p(0), p(1), p(2)});
        } else {
          points.push_back({p(0), p(1), 0});
        }
        vertex_indices.back().push_back(i++);
      }
      writer.write_points(points);
      writer.write_vertices(vertex_indices);
      if (!m_vertex_properties.empty()) {
        writer.write_point_data(points.size());
      }

      for (auto const& [name, prop] : m_vertex_properties) {
        std::vector<std::vector<Real>> data;
        data.reserve(m_vertex_positions.size());

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
#if TATOOINE_FLANN_AVAILABLE
  auto rebuild_kd_tree() {
    m_kd_tree.reset();
    kd_tree();
  }
  //----------------------------------------------------------------------------
 private:
  auto kd_tree() const -> auto& {
    if (m_kd_tree == nullptr) {
      flann::Matrix<Real> dataset{
          const_cast<Real*>(m_vertex_positions.front().data_ptr()), vertices().size(),
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
    flann::Matrix<Real> qm{const_cast<Real*>(x.data_ptr()), 1,  // NOLINT
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
    flann::Matrix<Real> qm{const_cast<Real*>(x.data_ptr()),  // NOLINT
                           1, num_dimensions()};
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
#endif
  //============================================================================
  template <typename T>
  auto inverse_distance_weighting_sampler(vertex_property_t<T> const& prop,
                                          Real const radius = 1) const {
    return inverse_distance_weighting_sampler_t<Real, NumDimensions, T>{
        *this, prop, radius};
  }
  //============================================================================
  template <typename T
#ifndef __cpp_concepts
            ,
            size_t NumDimensions_ = NumDimensions,
            enable_if<NumDimensions_ == NumDimensions &&
                      (NumDimensions_ == 3 || NumDimensions_ == 2)> = true
#endif
            >
  auto moving_least_squares_sampler(vertex_property_t<T> const& prop,
                                    Real const radius = 1) const
#ifdef __cpp_concepts
      requires(NumDimensions == 3 || NumDimensions == 2)
#endif

  {
    return moving_least_squares_sampler_t<Real, NumDimensions, T>{*this, prop,
                                                                  radius};
  }
};
//============================================================================
template <typename Real, size_t NumDimensions, typename T>
struct inverse_distance_weighting_sampler_t
    : field<inverse_distance_weighting_sampler_t<Real, NumDimensions, T>, Real,
            NumDimensions, T> {
  static_assert(flann_available(),
                "Inverse Distance Weighting Sampler needs FLANN!");
  using this_t   = inverse_distance_weighting_sampler_t<Real, NumDimensions, T>;
  using parent_t = field<this_t, Real, NumDimensions, T>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using pointset_t        = pointset<Real, NumDimensions>;
  using vertex_handle     = typename pointset_t::vertex_handle;
  using vertex_property_t = typename pointset_t::template vertex_property_t<T>;
  //==========================================================================
  pointset_t const&        m_pointset;
  vertex_property_t const& m_property;
  Real                     m_radius = 1;
  //==========================================================================
  inverse_distance_weighting_sampler_t(pointset_t const&        ps,
                                       vertex_property_t const& property,
                                       Real const               radius = 1)
      : m_pointset{ps}, m_property{property}, m_radius{radius} {}
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
  ~inverse_distance_weighting_sampler_t() = default;
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_t const& x, real_t const /*t*/) const
      -> tensor_t {
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
      auto const weight = 1 / *dist_it;
      accumulated_prop_val += property_value * weight;
      accumulated_weight += weight;
    }
    return accumulated_prop_val / accumulated_weight;
  }
};
//============================================================================
/// Moving Least Squares Sampler of scattered data in 2 Dimensions.
/// \see <em>An As-Short-As-Possible Introduction to the Least Squares,
/// Weighted Least Squares and Moving Least Squares Methods for Scattered Data
/// Approximation and Interpolation</em> \cite nealen2004LeastSquaresIntro.
template <typename Real, typename T>
struct moving_least_squares_sampler_t<Real, 2, T>
    : field<moving_least_squares_sampler_t<Real, 2, T>, Real, 2,
            T> {
  static_assert(flann_available(), "Moving Least Squares Sampler needs FLANN!");
  using this_t   = moving_least_squares_sampler_t<Real, 2, T>;
  using parent_t = field<this_t, Real, 2, T>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using pointset_t        = pointset<Real, 2>;
  using vertex_property_t = typename pointset_t::template vertex_property_t<T>;
  using vertex_handle     = typename pointset_t::vertex_handle;
  //==========================================================================
  pointset_t const&        m_pointset;
  vertex_property_t const& m_property;
  Real                     m_radius = 1;
  //==========================================================================
  moving_least_squares_sampler_t(pointset_t const&        ps,
                                 vertex_property_t const& property,
                                 Real const               radius = 1)
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
  [[nodiscard]] auto evaluate(pos_t const& q, Real const /*t*/) const
      -> tensor_t {
    auto        nn      = m_pointset.nearest_neighbors_radius_raw(q, m_radius);
    auto const& indices = nn.first;
    auto&       distances = nn.second;
    for (auto& d : distances) {
      d /= m_radius;
      d = 1 - d;
    }
    auto const  num_neighbors = size(indices);

    if (num_neighbors == 0) {
      if constexpr (is_arithmetic<tensor_t>) {
        return Real(0) / Real(0);
      } else {
        return tensor_t::fill(Real(0) / Real(0));
      }
    }
    if (num_neighbors == 1) {
      return m_property[vertex_handle{indices[0]}];
    }

    auto w = tensor<Real>::zeros(num_neighbors);
    auto F = num_components<T> > 1
                 ? tensor<Real>::zeros(num_neighbors, num_components<T>)
                 : tensor<Real>::zeros(num_neighbors);
    auto B = [&] {
      if (num_neighbors >= 10) {
        return tensor<Real>::ones(num_neighbors, 10);
      }
      if (num_neighbors >= 6) {
        return tensor<Real>::ones(num_neighbors, 6);
      }
      if (num_neighbors >= 3) {
        return tensor<Real>::ones(num_neighbors, 3);
      }
      return tensor<Real>::ones(1, 1);
    }();

    // build w
    auto weighting_function = [&](auto const d) {
       return 1 / d - 1 / m_radius;
       //return std::exp(-d * d);
    };
    for (size_t i = 0; i < num_neighbors; ++i) {
      w(i) = weighting_function(distances[i]);
    }
    // build F
    for (size_t i = 0; i < num_neighbors; ++i) {
      if constexpr (num_components<T> == 1) {
        F(i) = m_property[vertex_handle{indices[i]}];
      } else {
        for (size_t j = 0; j < num_components<T>; ++j) {
          F(i, j) = m_property[vertex_handle{indices[i]}](j);
        }
      }
    }
    // build B
    // linear terms of polynomial
    if (num_neighbors >= 3) {
      for (size_t i = 0; i < num_neighbors; ++i) {
        B(i, 1) = m_pointset.vertex_at(indices[i]).x() - q.x();
      }
      for (size_t i = 0; i < num_neighbors; ++i) {
        B(i, 2) = m_pointset.vertex_at(indices[i]).y() - q.y();
      }
    }
    // quadratic terms of polynomial
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
    // cubic terms of polynomial
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
};
//============================================================================
/// Moving Least Squares Sampler of scattered data in 3 Dimensions.
/// \see <em>An As-Short-As-Possible Introduction to the Least Squares,
/// Weighted Least Squares and Moving Least Squares Methods for Scattered Data
/// Approximation and Interpolation</em> \cite nealen2004LeastSquaresIntro.
template <typename Real, typename T>
struct moving_least_squares_sampler_t<Real, 3, T>
    : field<moving_least_squares_sampler_t<Real, 3, T>, Real, 3,
            T> {
  static_assert(flann_available(), "Moving Least Squares Sampler needs FLANN!");
  using this_t   = moving_least_squares_sampler_t<Real, 3, T>;
  using parent_t = field<this_t, Real, 3, T>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using pointset_t        = pointset<Real, 3>;
  using vertex_handle     = typename pointset_t::vertex_handle;
  using vertex_property_t = typename pointset_t::template vertex_property_t<T>;
  //==========================================================================
  pointset_t const&        m_pointset;
  vertex_property_t const& m_property;
  Real                     m_radius = 1;
  //==========================================================================
  moving_least_squares_sampler_t(pointset_t const&        ps,
                                 vertex_property_t const& property,
                                 Real const               radius = 1)
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
  [[nodiscard]] auto evaluate(pos_t const& q, real_t const /*t*/) const
      -> tensor_t {
    auto const  nn      = m_pointset.nearest_neighbors_radius_raw(q, m_radius);
    auto const& indices = nn.first;
    auto const& distances     = nn.second;
    auto const  num_neighbors = size(indices);
    if (num_neighbors == 0) {
      return T{Real(0) / Real(0)};
    }
    if (num_neighbors == 1) {
      return m_property[vertex_handle{indices[0]}];
    }

    auto w = tensor<Real>::zeros(num_neighbors);
    auto F = num_components<T> > 1
                 ? tensor<Real>::zeros(num_neighbors, num_components<T>)
                 : tensor<Real>::zeros(num_neighbors);
    auto B = [&] {
      if (num_neighbors >= 20) {
        return tensor<Real>::ones(num_neighbors, 20);
      } else if (num_neighbors >= 10) {
        return tensor<Real>::ones(num_neighbors, 10);
      } else if (num_neighbors >= 4) {
        return tensor<Real>::ones(num_neighbors, 4);
      }
      return tensor<Real>::ones(1, 1);
    }();
    // build w
    for (size_t i = 0; i < num_neighbors; ++i) {
      //if (distances[i] == 0) {
      //  return m_property[vertex_handle{indices[i]}];
      //}
      //w(i) = 1 / distances[i] - 1 / m_radius;
      w(i) = std::exp(-(m_radius - distances[i]) * (m_radius - distances[i]));
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
    // linear terms of polynomial
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
    // quadratic terms of polynomial
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
    // cubic terms of polynomial
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
};
//==============================================================================
template <typename Real, size_t NumDimensions>
auto vertices(pointset<Real, NumDimensions> const& ps) {
  return ps.vertices();
}
//------------------------------------------------------------------------------
template <typename Real, size_t NumDimensions>
auto begin(
    typename pointset<Real, NumDimensions>::vertex_container const& verts) {
  return verts.begin();
}
//------------------------------------------------------------------------------
template <typename Real, size_t NumDimensions>
auto end(
    typename pointset<Real, NumDimensions>::vertex_container const& verts) {
  return verts.end();
}
//------------------------------------------------------------------------------
template <typename Real, size_t NumDimensions>
auto size(
    typename pointset<Real, NumDimensions>::vertex_container const& verts) {
  return verts.size();
}
//==============================================================================
template <size_t NumDimensions>
using Pointset  = pointset<real_t, NumDimensions>;
using pointset2 = Pointset<2>;
using pointset3 = Pointset<3>;
using pointset4 = Pointset<4>;
using pointset5 = Pointset<5>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
