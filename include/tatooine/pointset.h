#ifndef TATOOINE_POINTSET_H
#define TATOOINE_POINTSET_H
//==============================================================================
#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/algorithm/find.hpp>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <vector>

#include "handle.h"
#include "property.h"
#include "tensor.h"
#include "type_traits.h"
#include "vtk_legacy.h"

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
  //----------------------------------------------------------------------------
  struct vertex_index : handle {
    using handle::handle;
    using handle::operator=;
    bool operator==(vertex_index other) const { return this->i == other.i; }
    bool operator!=(vertex_index other) const { return this->i != other.i; }
    bool operator<(vertex_index other) const { return this->i < other.i; }
    static constexpr auto invalid() {
      return vertex_index{handle::invalid_idx};
    }
  };
  //----------------------------------------------------------------------------
  struct vertex_iterator
      : boost::iterator_facade<vertex_iterator, vertex_index,
                               boost::bidirectional_traversal_tag,
                               vertex_index> {
    vertex_iterator(vertex_index _v, pointset const* _ps) : v{_v}, ps{_ps} {}
    vertex_iterator(vertex_iterator const& other) : v{other.v}, ps{other.ps} {}

   private:
    vertex_index    v;
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
      vertex_iterator vi{vertex_index{0}, m_pointset};
      if (!m_pointset->is_valid(*vi)) ++vi;
      return vi;
    }
    //--------------------------------------------------------------------------
    auto end() const {
      return vertex_iterator{vertex_index{m_pointset->m_vertices.size()},
                             m_pointset};
    }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  using vertex_property_t = vector_property_impl<vertex_index, T>;
  using vertex_property_container_t =
      std::map<std::string, std::unique_ptr<vector_property<vertex_index>>>;
  //============================================================================
 private:
  std::vector<pos_t>          m_vertices;
  std::vector<vertex_index>   m_invalid_vertices;
  vertex_property_container_t m_vertex_properties;
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
  auto at(vertex_index const v) -> auto& { return m_vertices[v.i]; }
  auto at(vertex_index const v) const -> auto const& { return m_vertices[v.i]; }
  //----------------------------------------------------------------------------
  auto vertex_at(vertex_index const v) -> auto& { return m_vertices[v.i]; }
  auto vertex_at(vertex_index const v) const -> auto const& {
    return m_vertices[v.i];
  }
  //----------------------------------------------------------------------------
  auto vertex_at(size_t const i) -> auto& { return m_vertices[i]; }
  auto vertex_at(size_t const i) const -> auto const& { return m_vertices[i]; }
  //----------------------------------------------------------------------------
  auto operator[](vertex_index const v) -> auto& { return at(v); }
  auto operator[](vertex_index const v) const -> auto const& { return at(v); }
  //----------------------------------------------------------------------------
  auto vertices() const { return vertex_container{this}; }
  //----------------------------------------------------------------------------
  auto vertex_data() -> auto& { return m_vertices; }
  auto vertex_data() const -> auto const& { return m_vertices; }
  //----------------------------------------------------------------------------
  template <real_number... Ts>
  requires(sizeof...(Ts) == N)
  auto insert_vertex(Ts const... ts) {
    m_vertices.push_back(pos_t{static_cast<Real>(ts)...});
    for (auto& [key, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_index{size(m_vertices) - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t const& v) {
    m_vertices.push_back(v);
    for (auto& [key, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_index{size(m_vertices) - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& v) {
    m_vertices.emplace_back(std::move(v));
    for (auto& [key, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_index{size(m_vertices) - 1};
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
  void remove(vertex_index v) {
    if (is_valid(v) &&
        boost::find(m_invalid_vertices, v) == m_invalid_vertices.end())
      m_invalid_vertices.push_back(v);
  }

  //----------------------------------------------------------------------------
  constexpr auto is_valid(vertex_index v) const -> bool {
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
    std::vector<std::pair<vertex_index, vertex_index>> duplicates;
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
  requires(N == 3 || N == 2) auto write_vtk(
      std::string const& path, std::string const& title = "Tatooine pointset") {
    vtk::legacy_file_writer writer(path, vtk::POLYDATA);
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
};
//------------------------------------------------------------------------------
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
