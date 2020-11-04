#ifndef TATOOINE_TRIANGULAR_MESH_H
#define TATOOINE_TRIANGULAR_MESH_H
//==============================================================================
#include <tatooine/pointset.h>
#include <tatooine/delaunator.h>
#include <tatooine/property.h>
#include <tatooine/vtk_legacy.h>

#include <boost/range/algorithm/copy.hpp>
#include <filesystem>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
class triangular_mesh : public pointset<Real, N> {
 public:
  using this_t   = triangular_mesh<Real, N>;
  using parent_t = pointset<Real, N>;
  using parent_t::at;
  using parent_t::points;
  using parent_t::num_vertices;
  using parent_t::vertex_properties;
  using typename parent_t::vertex_index;
  using parent_t::operator[];
  using parent_t::is_valid;
  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  //----------------------------------------------------------------------------
  struct triangle_index : handle {
    using handle::handle;
    constexpr bool operator==(triangle_index other) const {
      return this->i == other.i;
    }
    constexpr bool operator!=(triangle_index other) const {
      return this->i != other.i;
    }
    constexpr bool operator<(triangle_index other) const {
      return this->i < other.i;
    }
    static constexpr auto invalid() {
      return triangle_index{handle::invalid_idx};
    }
  };
  //----------------------------------------------------------------------------
  struct triangle_iterator
      : boost::iterator_facade<triangle_iterator, triangle_index,
                               boost::bidirectional_traversal_tag,
                               triangle_index> {
    triangle_iterator(triangle_index i, triangular_mesh const* mesh)
        : m_index{i}, m_mesh{mesh} {}
    triangle_iterator(triangle_iterator const& other)
        : m_index{other.m_index}, m_mesh{other.m_mesh} {}

   private:
    triangle_index       m_index;
    triangular_mesh const* m_mesh;

    friend class boost::iterator_core_access;

    auto increment() {
      do
        ++m_index;
      while (!m_mesh->is_valid(m_index));
    }
    auto decrement() {
      do
        --m_index;
      while (!m_mesh->is_valid(m_index));
    }

    auto equal(triangle_iterator const& other) const {
      return m_index == other.m_index;
    }
    auto dereference() const { return m_index; }
  };
  //----------------------------------------------------------------------------
  struct triangle_container {
    using iterator       = triangle_iterator;
    using const_iterator = triangle_iterator;

    triangular_mesh const* m_mesh;

    auto begin() const {
      triangle_iterator vi{triangle_index{0}, m_mesh};
      if (!m_mesh->is_valid(*vi)) { ++vi; }
      return vi;
    }

    auto end() const {
      return triangle_iterator{triangle_index{m_mesh->num_triangles()}, m_mesh};
    }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  using triangle_property_t = vector_property_impl<triangle_index, T>;
  using triangle_property_container_t =
      std::map<std::string,
               std::unique_ptr<vector_property<triangle_index>>>;
  //============================================================================
 private:
  std::vector<vertex_index>             m_triangle_indices;
  std::vector<triangle_index>           m_invalid_triangles;
  triangle_property_container_t         m_triangle_properties;

 public:
  //============================================================================
  constexpr triangular_mesh() = default;
  //============================================================================
 public:
  triangular_mesh(triangular_mesh const& other)
      : parent_t{other}, m_triangle_indices{other.m_triangle_indices} {
    for (auto const& [key, fprop] : other.m_triangle_properties) {
      m_triangle_properties.insert(std::pair{key, fprop->clone()});
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  triangular_mesh(triangular_mesh&& other) noexcept = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(triangular_mesh const& other) -> triangular_mesh& {
    parent_t::operator=(other);
    m_triangle_properties.clear();
    m_triangle_indices    = other.m_triangle_indices;
    for (auto const& [key, fprop] : other.m_triangle_properties) {
      m_triangle_properties.insert(std::pair{key, fprop->clone()});
    }
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(triangular_mesh&& other) noexcept
    -> triangular_mesh& = default;
  //----------------------------------------------------------------------------
  triangular_mesh(std::filesystem::path const& file) { read(file); }
  //============================================================================
  auto operator[](triangle_index t) const { return at(t); }
  auto operator[](triangle_index t) { return at(t); }
  //----------------------------------------------------------------------------
  auto at(triangle_index t) const {
    return std::tuple<size_t const&, size_t const&, size_t const&>{
        t.i * 3, t.i * 3 + 1, t.i * 3 + 2};
  }
  auto at(triangle_index t)  { 
    return std::tuple<size_t&, size_t&, size_t&>{t.i * 3, t.i * 3 + 1,
                                                 t.i * 3 + 2};
  }
  //----------------------------------------------------------------------------
  auto insert_triangle(vertex_index v0, vertex_index v1, vertex_index v2) {
    m_triangle_indices.push_back(v0.i);
    m_triangle_indices.push_back(v1.i);
    m_triangle_indices.push_back(v2.i);
    for (auto& [key, prop] : m_triangle_properties) { prop->push_back(); }
  }
  //----------------------------------------------------------------------------
  auto insert_triangle(size_t v0, size_t v1, size_t v2) {
    insert_triangle(vertex_index{v0}, vertex_index{v1}, vertex_index{v2});
  }
  //----------------------------------------------------------------------------
  auto insert_triangle(std::array<vertex_index, 3> const& t) {
    m_triangle_indices.push_back(t);
    for (auto& [key, prop] : m_triangle_properties) { prop->push_back(); }
  }
  //----------------------------------------------------------------------------
  auto clear() {
    parent_t::clear();
    m_triangle_indices.clear();
  }
  //----------------------------------------------------------------------------
  auto triangles() const { return triangle_container{this}; }
  //----------------------------------------------------------------------------
  auto num_triangles() const { return m_triangle_indices.size() / 3; }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires (N == 2)
  auto triangulate_delaunay() {
    delaunator::Delaunator d{this->points()};
    set_triangle_indices(std::move(d.triangles));
  }
  //----------------------------------------------------------------------------
  auto set_triangle_indices(std::vector<size_t> && is) {
    m_triangle_indices =
        std::move(*reinterpret_cast<std::vector<vertex_index>*>(&is));
  }
  //----------------------------------------------------------------------------
  auto set_triangle_indices(std::vector<size_t> const& is) {
    m_triangle_indices = *reinterpret_cast<std::vector<vertex_index>*>(&is);
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires (N == 2) || (N == 3)
  auto write_vtk(std::string const& path,
                 std::string const& title = "tatooine triangular mesh") const
      -> bool {
    using boost::copy;
    using boost::adaptors::transformed;
    vtk::legacy_file_writer writer(path, vtk::POLYDATA);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      if constexpr (N == 2) {
        auto three_dims = [](vec<Real, 2> const& v2) {
          return vec<Real, 3>{v2(0), v2(1), 0};
        };
        std::vector<vec<Real, 3>> v3s(num_vertices());
        auto                      three_dimensional = transformed(three_dims);
        copy(points() | three_dimensional, begin(v3s));
        writer.write_points(v3s);

      } else if constexpr (N == 3) {
        writer.write_points(points());
      }

      std::vector<std::vector<size_t>> polygons;
      polygons.reserve(num_triangles());
      for (size_t i = 0; i < size(m_triangle_indices); i+=3) {
        polygons.push_back(std::vector{m_triangle_indices[i].i, m_triangle_indices[i + 1].i,
                                       m_triangle_indices[i + 2].i});
      }
      writer.write_polygons(polygons);

      // write vertex_index data
      writer.write_point_data(this->num_vertices());
      for (auto const& [name, prop] : vertex_properties()) {
        if (prop->type() == typeid(vec<Real, 4>)) {
        } else if (prop->type() == typeid(vec<Real, 3>)) {
        } else if (prop->type() == typeid(vec<Real, 2>)) {
          auto const& casted_prop =
              *dynamic_cast<vertex_property_t<vec<Real, 2>> const*>(prop.get());
          writer.write_scalars(name, casted_prop.container());
        } else if (prop->type() == typeid(Real)) {
          auto const& casted_prop =
              *dynamic_cast<vertex_property_t<Real> const*>(prop.get());
          writer.write_scalars(name, casted_prop.container());
        }
      }

      writer.close();
      return true;
    } else {
      return false;
    }
  }
  //----------------------------------------------------------------------------
  auto read(std::filesystem::path const& path) {
    auto const ext = path.extension();
    if constexpr (N == 2 || N == 3) {
      if (ext == "vtk") { read_vtk(path); }
    }
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires (N == 2) || (N == 3)
  auto read_vtk(std::filesystem::path const& path) {
    struct listener_t : vtk::legacy_file_listener {
      triangular_mesh& mesh;

      listener_t(triangular_mesh& _mesh) : mesh(_mesh) {}

      void on_dataset_type(vtk::DatasetType t) override {
        if (t != vtk::POLYDATA) {
          throw std::runtime_error{
              "[triangular_mesh] need polydata when reading vtk legacy"};
        }
      }

      void on_points(std::vector<std::array<float, 3>> const& ps) override {
        for (auto& p : ps) { mesh.insert_vertex(p[0], p[1], p[2]); }
      }
      void on_points(std::vector<std::array<double, 3>> const& ps) override {
        for (auto& p : ps) { mesh.insert_vertex(p[0], p[1], p[2]); }
      }
      void on_polygons(std::vector<int> const& ps) override {
        for (size_t i = 0; i < ps.size();) {
          auto const& size = ps[i++];
          if (size == 4) {
            mesh.insert_triangle(size_t(ps[i]), size_t(ps[i + 1]),
                                    size_t(ps[i + 2]), size_t(ps[i + 2]));
          }
          i += size;
        }
      }
      void on_scalars(std::string const& data_name,
                      std::string const& /*lookup_table_name*/,
                      size_t num_comps, std::vector<double> const& scalars,
                      vtk::ReaderData data) override {
        if (data == vtk::POINT_DATA) {
          if (num_comps == 1) {
            auto& prop = mesh.template add_vertex_property<double>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) { prop[i] = scalars[i]; }
          } else if (num_comps == 2) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 2>>(data_name);

            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[i][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 3) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 3>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[i][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 4) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 4>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[i][j] = scalars[i * num_comps + j];
              }
            }
          }
        }
      }
    } listener{*this};
    vtk::legacy_file f{path};
    f.add_listener(listener);
    f.read();
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto triangle_property(std::string const& name) -> auto& {
    auto prop = m_triangle_properties.at(name).get();
    auto casted_prop =  dynamic_cast<triangle_property_t<T>*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto triangle_property(std::string const& name) const -> auto const& {
    auto prop = m_triangle_properties.at(name).get();
    auto casted_prop =  dynamic_cast<triangle_property_t<T> const*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto add_triangle_property(std::string const& name,
                                 T const&           value = T{}) -> auto& {
    auto [it, suc] = m_triangle_properties.insert(
        std::pair{name, std::make_unique<triangle_property_t<T>>(value)});
    auto fprop = dynamic_cast<triangle_property_t<T>*>(it->second.get());
    fprop->resize(num_triangles());
    return *fprop;
  }
  //----------------------------------------------------------------------------
  constexpr bool is_valid(triangle_index t) const {
    return boost::find(m_invalid_triangles, t) ==
           end(m_invalid_triangles);
  }
};
//==============================================================================
triangular_mesh()->triangular_mesh<double, 3>;
triangular_mesh(std::string const&)->triangular_mesh<double, 3>;
//==============================================================================
namespace detail {
template <typename MeshCont>
auto write_mesh_container_to_vtk(MeshCont const&    meshes,
                                 std::string const& path,
                                 std::string const& title) {
  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
  if (writer.is_open()) {
    size_t num_pts   = 0;
    size_t cur_first = 0;
    for (auto const& m : meshes) {
      num_pts += m.num_vertices();
    }
    std::vector<std::array<typename MeshCont::value_type::real_t, 3>> points;
    std::vector<std::vector<size_t>>                                  triangles;
    points.reserve(num_pts);
    triangles.reserve(meshes.size());

    for (auto const& m : meshes) {
      // add points
      for (auto const& v : m.vertices()) {
        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
      }

      // add triangles
      for (auto t : m.triangles()) {
        triangles.emplace_back();
        triangles.back().push_back(cur_first + m[t][0].i);
        triangles.back().push_back(cur_first + m[t][1].i);
        triangles.back().push_back(cur_first + m[t][2].i);
      }
      cur_first += m.num_vertices();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_polygons(triangles);
    // writer.write_point_data(num_pts);
    writer.close();
  }
}
}  // namespace detail
//==============================================================================
template <typename Real>
auto write_vtk(std::vector<triangular_mesh<Real, 3>> const& meshes,
               std::string const&                           path,
               std::string const& title = "tatooine meshes") {
  detail::write_mesh_container_to_vtk(meshes, path, title);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif

