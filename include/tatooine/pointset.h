#ifndef TATOOINE_POINTSET_H
#define TATOOINE_POINTSET_H

#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/algorithm/find.hpp>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <vector>
#include "property.h"
#include "tensor.h"
#include "vtk_legacy.h"
#include "strong_indices.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
struct pointset;

//==============================================================================
template <typename Real, size_t N>
struct vertex_iterator
    : boost::iterator_facade<vertex_iterator<Real, N>, vertex,
                             boost::bidirectional_traversal_tag, vertex> {
  vertex_iterator(vertex _v, const pointset<Real, N>* _ps) : v{_v}, ps{_ps} {}
  vertex_iterator(const vertex_iterator& other) : v{other.v}, ps{other.ps} {}

 private:
  vertex                   v;
  const pointset<Real, N>* ps;

  friend class boost::iterator_core_access;

  void increment() {
    do
      ++v.i;
    while (!ps->is_valid(v));
  }
  void decrement() {
    do
      --v.i;
    while (!ps->is_valid(v));
  }

  auto equal(const vertex_iterator& other) const { return v.i == other.v.i; }
  auto dereference() const { return v; }
};

//==============================================================================
template <typename Real, size_t N>
struct pointset_vertex_container {
  using iterator       = vertex_iterator<Real, N>;
  using const_iterator = vertex_iterator<Real, N>;

  const pointset<Real, N>* m_pointset;

  auto begin() const {
    vertex_iterator<Real, N> vi{vertex{0}, m_pointset};
    if (!m_pointset->is_valid(*vi)) {++vi;}
    return vi;
  }

  auto end() const {
    return vertex_iterator<Real, N>{
        vertex{m_pointset->num_vertices() + m_pointset->num_invalid_vertices()},
        m_pointset};
  }
};

//==============================================================================
template <typename Real, size_t N>
struct pointset {
  static constexpr size_t triangle_dims = 2;
  // static constexpr size_t tetgen_dims = 3;
  static constexpr auto num_dimensions() { return N; }
  using real_t = Real;
  using this_t = pointset<Real, N>;
  using pos_t  = vec<Real, N>;

  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  template <typename T>
  struct vertex_property_t : public property_type<T> {
    // inherit all constructors
    vertex_property_t(const vertex_property_t& other) : property_type<T>{other} {}
    using property_type<T>::property_type;
    auto&       at(vertex v) { return property_type<T>::at(v.i); }
    const auto& at(vertex v) const { return property_type<T>::at(v.i); }
    auto&       operator[](vertex v) { return property_type<T>::at(v.i); }
    const auto& operator[](vertex v) const {
      return property_type<T>::operator[](v.i);
    }
    std::unique_ptr<property> clone() const override {
      return std::unique_ptr<vertex_property_t<T>>(new vertex_property_t<T>{*this});
    }
  };

  //============================================================================
 protected:
  std::vector<pos_t>                               m_points;
  std::vector<vertex>                              m_invalid_vertices;
  std::map<std::string, std::unique_ptr<property>> m_vertex_properties;

  //============================================================================
 public:
  pointset() = default;

  //----------------------------------------------------------------------------
  pointset(std::initializer_list<pos_t>&& vertices)
      : m_points(std::move(vertices)) {}

  //----------------------------------------------------------------------------
  // #ifdef USE_TRIANGLE
  //   pointset(const triangle::io& io) {
  //     for (int i = 0; i < io.numberofpoints; ++i)
  //       insert_vertex(io.pointlist[i * 2], io.pointlist[i * 2 + 1]);
  //   }
  // #endif

  // template <size_t _N = N, typename = std::enable_if_t<_N == 3>>
  // pointset(const tetgenio& io) {
  //   for (int i = 0; i < io.numberofpoints; ++i)
  //     insert_vertex(io.pointlist[i * 3], io.pointlist[i * 3 + 1],
  //                   io.pointlist[i * 3 + 2]);
  // }

  //----------------------------------------------------------------------------
  pointset(const pointset& other)
      : m_points(other.m_points), m_invalid_vertices(other.m_invalid_vertices) {
    m_vertex_properties.clear();
    for (const auto& [name, prop] : other.m_vertex_properties)
      m_vertex_properties.insert(std::pair{name, prop->clone()});
  }

  //----------------------------------------------------------------------------
  pointset(pointset&& other)
      : m_points(std::move(other.m_points)),
        m_invalid_vertices(std::move(other.m_invalid_vertices)),
        m_vertex_properties(std::move(other.m_vertex_properties)) {}

  //----------------------------------------------------------------------------
  auto& operator=(const pointset& other) {
    m_vertex_properties.clear();
    m_points         = other.m_points;
    m_invalid_vertices = other.m_invalid_vertices;
    for (const auto& [name, prop] : other.m_vertex_properties)
      m_vertex_properties[name] = prop->clone();
    return *this;
  }

  //----------------------------------------------------------------------------
  auto& operator=(pointset&& other) {
    m_vertex_properties = std::move(other.m_vertex_properties);
    m_invalid_vertices    = std::move(other.m_invalid_vertices);
    return *this;
  }

  //----------------------------------------------------------------------------
  const auto& vertex_properties() const { return m_vertex_properties; }

  //----------------------------------------------------------------------------
  auto&       at(vertex v) { return m_points[v.i]; }
  const auto& at(vertex v) const { return m_points[v.i]; }

  //----------------------------------------------------------------------------
  auto&       operator[](vertex v) { return at(v); }
  const auto& operator[](vertex v) const { return at(v); }

  //----------------------------------------------------------------------------
  auto vertices() const { return pointset_vertex_container<Real, N>{this}; }
  //----------------------------------------------------------------------------
  const auto& invalid_vertices() const { return m_invalid_vertices; }

  //----------------------------------------------------------------------------
  auto&       points() { return m_points; }
  const auto& points() const { return m_points; }

  //----------------------------------------------------------------------------
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) == N>>
  auto insert_vertex(Ts&&... ts) {
    points().push_back({static_cast<Real>(std::forward<Ts>(ts))...});
    for (auto& [key, prop] : m_vertex_properties) prop->push_back();
    return vertex{m_points.size() - 1};
  }

  //----------------------------------------------------------------------------
  auto insert_vertex(const pos_t& v) {
    points().push_back(v);
    for (auto& [key, prop] : m_vertex_properties) prop->push_back();
    return vertex{m_points.size() - 1};
  }

  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& v) {
    points().push_back(std::move(v));
    for (auto& [key, prop] : m_vertex_properties) prop->push_back();
    return vertex{m_points.size() - 1};
  }

  //----------------------------------------------------------------------------
  //! tidies up invalid vertices
  void tidy_up() {
    for (const auto v : m_invalid_vertices) {
      // decrease deleted vertex indices;
      for (auto& v_to_decrease : m_invalid_vertices){
        if (v_to_decrease.i > v.i) {--v_to_decrease.i;}}

      m_points.erase(m_points.begin() + v.i);
      for (const auto& [key, prop] : m_vertex_properties) { prop->erase(v.i); }
    }
    m_invalid_vertices.clear();
  }
  
  //----------------------------------------------------------------------------
  //! tidies up invalid vertices
  template <typename... Ts>
  void tidy_up(std::tuple<Ts&...> additional_vertex_properties) {
    for (const auto v : m_invalid_vertices) {
      // decrease deleted vertex indices;
      for (auto& v_to_decrease : m_invalid_vertices){
        if (v_to_decrease.i > v.i) {--v_to_decrease.i;}}

      m_points.erase(m_points.begin() + v.i);
      for (const auto& [key, prop] : m_vertex_properties) { prop->erase(v.i); }
      for (auto& add_prop : additional_vertex_properties) {
        add_prop.erase(v.i);
      }
    }
    m_invalid_vertices.clear();
  }

  //----------------------------------------------------------------------------
  void remove(vertex v) {
    if (is_valid(v) &&
        boost::find(m_invalid_vertices, v) == m_invalid_vertices.end())
      m_invalid_vertices.push_back(v);
  }

  //----------------------------------------------------------------------------
  constexpr bool is_valid(vertex v) const {
    return boost::find(m_invalid_vertices, v) == m_invalid_vertices.end();
  }

  //----------------------------------------------------------------------------
  void clear_vertices() {
    m_points.clear();
    m_points.shrink_to_fit();
    m_invalid_vertices.clear();
    m_invalid_vertices.shrink_to_fit();
    for (auto& [key, val] : m_vertex_properties) val->clear();
  }
  void clear() { clear_vertices(); }

  //----------------------------------------------------------------------------
  auto num_vertices() const {
    return m_points.size() - m_invalid_vertices.size();
  }

  //----------------------------------------------------------------------------
  auto num_invalid_vertices() const {
    return m_invalid_vertices.size();
  }

  //----------------------------------------------------------------------------
  auto join(const this_t& other) {
    for (auto v : other.vertices()) { insert_vertex(other[v]); }
  };

  //----------------------------------------------------------------------------
  auto find_duplicates(Real eps = 1e-6) {
    std::vector<std::pair<vertex, vertex>> duplicates;
    for (auto v0 = vertices().begin(); v0 != vertices().end(); ++v0)
      for (auto v1 = next(v0); v1 != vertices().end(); ++v1)
        if (approx_equal(at(v0), at(v1), eps)) duplicates.emplace_back(v0, v1);

    return duplicates;
  }

#ifdef USE_TRIANGLE
  //----------------------------------------------------------------------------
  template <size_t _N = N, typename = std::enable_if_t<_N == triangle_dims>>
  auto to_triangle_io() const {
    triangle::io in;
    size_t       i    = 0;
    in.numberofpoints = num_vertices();
    in.pointlist      = new triangle::Real[in.numberofpoints * triangle_dims];
    for (auto v : vertices()) {
      for (size_t j = 0; j < triangle_dims; ++j) {
        in.pointlist[i++] = at(v)(j);
      }
    }

    return in;
  }

  //----------------------------------------------------------------------------
  template <typename vertex_cont_t, size_t _N = N,
            typename = std::enable_if_t<_N == triangle_dims>>
  auto to_triangle_io(const vertex_cont_t& vertices) const {
    triangle::io in;
    size_t       i    = 0;
    in.numberofpoints = num_vertices();
    in.pointlist      = new triangle::Real[in.numberofpoints * triangle_dims];
    for (auto v : vertices()) {
      for (size_t j = 0; j < triangle_dims; ++j) {
        in.pointlist[i++] = at(v)(j);
      }
    }

    return in;
  }
#endif

  //----------------------------------------------------------------------------
  // template <size_t _N = N, typename = std::enable_if_t<_N == tetgen_dims>>
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
  //! using specified vertices of point_set
  // template <typename vertex_cont_t>
  // auto to_tetgen_io(const vertex_cont_t& vertices) const {
  //   tetgenio io;
  //   size_t       i    = 0;
  //   io.numberofpoints = vertices.size();
  //   io.pointlist      = new tetgen_real_t[io.numberofpoints * 3];
  //   for (auto v : vertices) {
  //     const auto& x       = at(v);
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
  auto& vertex_property(const std::string& name) {
    auto prop        = m_vertex_properties.at(name).get();
    auto casted_prop = dynamic_cast<vertex_property_t<T>*>(prop);
    return *casted_prop;
  }

  //----------------------------------------------------------------------------
  template <typename T>
  const auto& vertex_property(const std::string& name) const {
    return *dynamic_cast<vertex_property_t<T>*>(m_vertex_properties.at(name).get());
  }

  //----------------------------------------------------------------------------
  template <typename T>
  auto& add_vertex_property(const std::string& name, const T& value = T{}) {
    auto [it, suc] = m_vertex_properties.insert(
        std::pair{name, std::make_unique<vertex_property_t<T>>(value)});
    auto prop = dynamic_cast<vertex_property_t<T>*>(it->second.get());
    prop->resize(m_points.size());
    return *prop;
  }

  //----------------------------------------------------------------------------
  template <typename = std::enable_if_t<N == 3 || N == 2>>
  void write_vtk(const std::string& path,
                 const std::string& title = "Tatooine pointset") {
    vtk::legacy_file_writer writer(path, vtk::POLYDATA);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      std::vector<std::array<Real, 3>> points;
      points.reserve(m_points.size());
      for (const auto& p : m_points)
        if constexpr (N == 3)
          points.push_back({p(0), p(1), p(2)});
        else
          points.push_back({p(0), p(1), 0});
      writer.write_points(points);
      writer.write_point_data(m_points.size());
      for (const auto& [name, prop] : m_vertex_properties) {
        std::vector<std::vector<Real>> data;
        data.reserve(m_points.size());

        if (prop->type() == typeid(vec<Real, 4>)) {
          for (const auto& v4 :
               *dynamic_cast<const vertex_property_t<vec<Real, 4>>*>(prop.get()))
            data.push_back({v4(0), v4(1), v4(2), v4(3)});

        } else if (prop->type() == typeid(vec<Real, 3>)) {
          for (const auto& v3 :
               *dynamic_cast<const vertex_property_t<vec<Real, 3>>*>(prop.get()))
            data.push_back({v3(0), v3(1), v3(2)});

        } else if (prop->type() == typeid(vec<Real, 2>)) {
          for (const auto& v2 :
               *dynamic_cast<const vertex_property_t<vec<Real, 2>>*>(prop.get()))
            data.push_back({v2(0), v2(1)});

        } else if (prop->type() == typeid(Real)) {
          for (const auto& scalar :
               *dynamic_cast<const vertex_property_t<Real>*>(prop.get()))
            data.push_back({scalar});
        }

        writer.write_scalars(name, data);
      }
      writer.close();
    }
  }
};

//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto begin(const pointset_vertex_container<Real, N>& verts) {
  return verts.begin();
}

//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto end(const pointset_vertex_container<Real, N>& verts) {
  return verts.end();
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
