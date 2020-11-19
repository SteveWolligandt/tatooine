#ifndef TATOOINE_TRIANGULAR_MESH_H
#define TATOOINE_TRIANGULAR_MESH_H
//==============================================================================
#include <CGAL/Cartesian.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <tatooine/grid.h>
#include <tatooine/octree.h>
#include <tatooine/pointset.h>
#include <tatooine/property.h>
#include <tatooine/quadtree.h>
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
  using parent_t::num_vertices;
  using parent_t::vertex_data;
  using parent_t::vertex_properties;
  using typename parent_t::vertex_index;
  using parent_t::operator[];
  using parent_t::is_valid;
  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  using typename parent_t::pos_t;

  template <typename T>
  struct vertex_property_sampler_t {
    this_t const&               m_mesh;
    vertex_property_t<T> const& m_prop;

    [[nodiscard]] auto operator()(pos_t const& x) const { return sample(x); }
    [[nodiscard]] auto sample(pos_t const& x) const -> T {
      auto tris = m_mesh.hierarchy().nearby_triangles_ptr(x);
      if (tris == nullptr || tris->empty()) {
        throw std::runtime_error{
            "[vertex_property_sampler_t::sample] out of domain"};
      }
      for (auto ti : *tris) {
        auto [vi0, vi1, vi2]     = m_mesh.triangle_at(ti);
        auto const&           v0 = m_mesh.vertex_at(vi0);
        auto const&           v1 = m_mesh.vertex_at(vi1);
        auto const&           v2 = m_mesh.vertex_at(vi2);
        mat<Real, 3, 3> const A{{v0(0), v1(0), v2(0)},
                                {v0(1), v1(1), v2(1)},
                                {Real(1), Real(1), Real(1)}};
        vec<Real, 3> const    b{x(0), x(1), 1};
        auto const            abc = solve(A, b);
        if (abc(0) >= 0 && abc(0) <= 1 && abc(1) >= 0 && abc(1) <= 1 &&
            abc(2) >= 0 && abc(2) <= 1) {
          return m_prop[vi0] * abc(0) + m_prop[vi1] * abc(1) +
                 m_prop[vi2] * abc(2);
        }
      }
      return T{};
    }
  };
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
    triangle_index         m_index;
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
      if (!m_mesh->is_valid(*vi)) {
        ++vi;
      }
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
      std::map<std::string, std::unique_ptr<vector_property<triangle_index>>>;
  using hierarchy_t =
      std::conditional_t<N == 2, quadtree<Real>,
                         std::conditional_t<N == 3, octree<Real>, void>>;
  //============================================================================
 private:
  std::vector<vertex_index>            m_triangle_indices;
  std::vector<triangle_index>          m_invalid_triangles;
  triangle_property_container_t        m_triangle_properties;
  mutable std::unique_ptr<hierarchy_t> m_hierarchy;

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
    m_triangle_indices = other.m_triangle_indices;
    for (auto const& [key, fprop] : other.m_triangle_properties) {
      m_triangle_properties.insert(std::pair{key, fprop->clone()});
    }
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator           =(triangular_mesh&& other) noexcept
      -> triangular_mesh& = default;
  //----------------------------------------------------------------------------
  template <indexable_space DimX, indexable_space DimY>
  requires(N == 2) triangular_mesh(grid<DimX, DimY> const& g) {
    for (auto v : g.vertices()) {
      insert_vertex(v);
    }
    for (size_t j = 0; j < g.size(1) - 1; ++j) {
      for (size_t i = 0; i < g.size(0) - 1; ++i) {
        insert_face(vertex_index{i + j * g.size(0)},
                    vertex_index{(i + 1) + j * g.size(0)},
                    vertex_index{i + (j + 1) * g.size(0)});
        insert_face(vertex_index{(i + 1) + j * g.size(0)},
                    vertex_index{(i + 1) + (j + 1) * g.size(0)},
                    vertex_index{i + (j + 1) * g.size(0)});
      }
    }
    auto copy_prop = [&]<typename T>(auto const& name, auto const& prop) {
      if (prop->type() == typeid(T)) {
        auto const& grid_prop = g.template vertex_property<T>(name);
        auto&       tri_prop  = this->template add_vertex_property<T>(name);
        g.loop_over_vertex_indices([&](auto const... is) {
          std::array is_arr{is...};
          tri_prop[vertex_index{is_arr[0] + is_arr[1] * g.size(0)}] =
              grid_prop(is...);
        });
      }
    };
    for (auto const& [name, prop] : g.vertex_properties()) {
      copy_prop.template operator()<vec<double, 4>>(name, prop);
      copy_prop.template operator()<vec<double, 3>>(name, prop);
      copy_prop.template operator()<vec<double, 2>>(name, prop);
      copy_prop.template operator()<double>(name, prop);
      copy_prop.template operator()<vec<float, 4>>(name, prop);
      copy_prop.template operator()<vec<float, 3>>(name, prop);
      copy_prop.template operator()<vec<float, 2>>(name, prop);
      copy_prop.template operator()<float>(name, prop);
    }
  }
  //----------------------------------------------------------------------------
  triangular_mesh(std::filesystem::path const& file) { read(file); }
  //============================================================================
  auto operator[](triangle_index const t) const { return triangle_at(t.i); }
  auto operator[](triangle_index const t) { return triangle_at(t.i); }
  //----------------------------------------------------------------------------
  auto at(triangle_index t) const { return triangle_at(t.i); }
  auto at(triangle_index t) { return triangle_at(t.i); }
  //----------------------------------------------------------------------------
  auto triangle_at(triangle_index const t) const { return triangle_at(t.i); }
  auto triangle_at(triangle_index const t) { return triangle_at(t.i); }
  //----------------------------------------------------------------------------
  auto triangle_at(size_t const i) const
      -> std::tuple<vertex_index const&, vertex_index const&,
                    vertex_index const&> {
    return {m_triangle_indices[i * 3], m_triangle_indices[i * 3 + 1],
            m_triangle_indices[i * 3 + 2]};
  }
  auto triangle_at(size_t const i)
      -> std::tuple<vertex_index&, vertex_index&, vertex_index&> {
    return {m_triangle_indices[i * 3], m_triangle_indices[i * 3 + 1],
            m_triangle_indices[i * 3 + 2]};
  }
  //----------------------------------------------------------------------------
  template <real_number... Ts>
  requires(sizeof...(Ts) == N) auto insert_vertex(Ts const... ts) {
    auto const vi = parent_t::insert_vertex(ts...);
    if (m_hierarchy != nullptr) {
      if (!m_hierarchy->insert_vertex(*this, vi.i)) {
        clear_hierarchy();
        build_hierarchy();
      }
    }
    return vi;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t const& v) {
    auto const vi = parent_t::insert_vertex(v);
    if (m_hierarchy != nullptr) {
      if (!m_hierarchy->insert_vertex(*this, vi.i)) {
        clear_hierarchy();
        build_hierarchy();
      }
    }
    return vi;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& v) {
    auto const vi = parent_t::insert_vertex(std::move(v));
    if (m_hierarchy != nullptr) {
      if (!m_hierarchy->insert_vertex(*this, vi.i)) {
        clear_hierarchy();
        build_hierarchy();
      }
    }
    return vi;
  }
  //----------------------------------------------------------------------------
  auto insert_face(vertex_index v0, vertex_index v1, vertex_index v2) {
    m_triangle_indices.push_back(v0);
    m_triangle_indices.push_back(v1);
    m_triangle_indices.push_back(v2);
    auto const ti = triangle_index{size(m_triangle_indices) / 3 - 1};
    for (auto& [key, prop] : m_triangle_properties) {
      prop->push_back();
    }
    if (m_hierarchy != nullptr) {
      if (!m_hierarchy->insert_face(*this, ti.i)) {
        clear_hierarchy();
        build_hierarchy();
      }
    }
    return ti;
  }
  //----------------------------------------------------------------------------
  auto insert_face(size_t v0, size_t v1, size_t v2) {
    return insert_face(vertex_index{v0}, vertex_index{v1}, vertex_index{v2});
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
  template <typename = void> requires(N == 2)
  auto triangulate_delaunay() -> void {
    m_triangle_indices.clear();
    using Kernel = CGAL::Cartesian<Real>;
    using Vb     = CGAL::Triangulation_vertex_base_with_info_2<vertex_index, Kernel>;
    using Tds    = CGAL::Triangulation_data_structure_2<Vb>;
    using Triangulation = CGAL::Delaunay_triangulation_2<Kernel, Tds>;
    using Point         = typename Kernel::Point_2;
    std::vector<std::pair<Point, vertex_index>> points;
    points.reserve(this->num_vertices());
    for (auto v : this->vertices()) {
      points.emplace_back(Point{at(v)(0), at(v)(1)}, v);
    }

    Triangulation dt{begin(points), end(points)};
    for (auto it = dt.finite_faces_begin(); it != dt.finite_faces_end(); ++it) {
      insert_face(vertex_index{it->vertex(0)->info()},
                  vertex_index{it->vertex(1)->info()},
                  vertex_index{it->vertex(2)->info()});
    }
  }
  //----------------------------------------------------------------------------
  auto set_triangle_indices(std::vector<size_t>&& is) {
    m_triangle_indices =
        std::move(*reinterpret_cast<std::vector<vertex_index>*>(&is));
  }
  //----------------------------------------------------------------------------
  auto set_triangle_indices(std::vector<size_t> const& is) {
    m_triangle_indices = *reinterpret_cast<std::vector<vertex_index>*>(&is);
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires(N == 2 || N == 3) auto build_hierarchy() const {
    auto& h = hierarchy();
    for (auto v : this->vertices()) {
      h.insert_vertex(*this, v.i);
    }
    for (auto t : triangles()) {
      h.insert_face(*this, t.i);
    }
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires(N == 2 || N == 3) auto clear_hierarchy() {
    m_hierarchy.reset();
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires(N == 2 || N == 3) auto hierarchy() const -> auto& {
    if (m_hierarchy == nullptr) {
      auto min = pos_t::ones() * std::numeric_limits<Real>::infinity();
      auto max = -pos_t::ones() * std::numeric_limits<Real>::infinity();
      for (auto v : this->vertices()) {
        for (size_t i = 0; i < N; ++i) {
          min(i) = std::min(min(i), at(v)(i));
          max(i) = std::max(max(i), at(v)(i));
        }
      }
      m_hierarchy = std::make_unique<hierarchy_t>(min, max);
    }
    return *m_hierarchy;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property_sampler(vertex_property_t<T> const& prop) const {
    if (m_hierarchy == nullptr) {
      build_hierarchy();
    }
    return vertex_property_sampler_t<T>{*this, prop};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property_sampler(std::string const& name) const {
    if (m_hierarchy == nullptr) {
      build_hierarchy();
    }
    return vertex_property_sampler_t<T>{
        *this, this->template vertex_property<T>(name)};
  }
  //----------------------------------------------------------------------------
  template <typename = void>
      requires(N == 2) ||
      (N == 3) auto write_vtk(
          std::string const& path,
          std::string const& title = "tatooine triangular mesh") const -> bool {
    using boost::copy;
    using boost::adaptors::transformed;
    vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      if constexpr (N == 2) {
        auto three_dims = [](vec<Real, 2> const& v2) {
          return vec<Real, 3>{v2(0), v2(1), 0};
        };
        std::vector<vec<Real, 3>> v3s(num_vertices());
        auto                      three_dimensional = transformed(three_dims);
        copy(vertex_data() | three_dimensional, begin(v3s));
        writer.write_points(v3s);

      } else if constexpr (N == 3) {
        writer.write_points(vertex_data());
      }

      std::vector<std::vector<size_t>> polygons;
      polygons.reserve(num_triangles());
      for (size_t i = 0; i < size(m_triangle_indices); i += 3) {
        polygons.push_back(std::vector{m_triangle_indices[i].i,
                                       m_triangle_indices[i + 1].i,
                                       m_triangle_indices[i + 2].i});
      }
      writer.write_polygons(polygons);

      // write vertex_index data
      writer.write_point_data(this->num_vertices());
      for (auto const& [name, prop] : vertex_properties()) {
        if (prop->type() == typeid(vec<Real, 4>)) {
          auto const& casted_prop =
              *dynamic_cast<vertex_property_t<vec<Real, 4>> const*>(prop.get());
          writer.write_scalars(name, casted_prop.container());
        } else if (prop->type() == typeid(vec<Real, 3>)) {
          auto const& casted_prop =
              *dynamic_cast<vertex_property_t<vec<Real, 3>> const*>(prop.get());
          writer.write_scalars(name, casted_prop.container());
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
      if (ext == ".vtk") {
        read_vtk(path);
      }
    }
  }
  //----------------------------------------------------------------------------
  template <typename = void>
      requires(N == 2) ||
      (N == 3) auto read_vtk(std::filesystem::path const& path) {
    struct listener_t : vtk::legacy_file_listener {
      triangular_mesh& mesh;

      listener_t(triangular_mesh& _mesh) : mesh(_mesh) {}

      void on_dataset_type(vtk::dataset_type t) override {
        if (t != vtk::dataset_type::polydata) {
          throw std::runtime_error{
              "[triangular_mesh] need polydata when reading vtk legacy"};
        }
      }

      void on_points(std::vector<std::array<float, 3>> const& ps) override {
        for (auto& p : ps) {
          if constexpr (N == 3) {
            mesh.insert_vertex(p[0], p[1], p[2]);
          } else if constexpr (N == 2) {
            mesh.insert_vertex(p[0], p[1]);
          }
        }
      }
      void on_points(std::vector<std::array<double, 3>> const& ps) override {
        for (auto& p : ps) {
          if constexpr (N == 3) {
            mesh.insert_vertex(p[0], p[1], p[2]);
          } else if constexpr (N == 2) {
            mesh.insert_vertex(p[0], p[1]);
          }
        }
      }
      void on_polygons(std::vector<int> const& ps) override {
        for (size_t i = 0; i < ps.size();) {
          auto const& size = ps[i++];
          mesh.insert_face(size_t(ps[i]), size_t(ps[i + 1]), size_t(ps[i + 2]));
          i += size;
        }
      }
      void on_scalars(std::string const& data_name,
                      std::string const& /*lookup_table_name*/,
                      size_t num_comps, std::vector<float> const& /*scalars*/,
                      vtk::reader_data /*data*/) override {
        std::cerr << data_name << " " << num_comps << '\n';
      }
      void on_scalars(std::string const& data_name,
                      std::string const& /*lookup_table_name*/,
                      size_t num_comps, std::vector<double> const& scalars,
                      vtk::reader_data data) override {
        std::cerr << data_name << " " << num_comps << '\n';
        if (data == vtk::reader_data::point_data) {
          if (num_comps == 1) {
            auto& prop = mesh.template add_vertex_property<double>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              prop[vertex_index{i}] = scalars[i];
            }
          } else if (num_comps == 2) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 2>>(data_name);

            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[vertex_index{i}][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 3) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 3>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[vertex_index{i}][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 4) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 4>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[vertex_index{i}][j] = scalars[i * num_comps + j];
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
    auto prop        = m_triangle_properties.at(name).get();
    auto casted_prop = dynamic_cast<triangle_property_t<T>*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto triangle_property(std::string const& name) const -> auto const& {
    auto prop        = m_triangle_properties.at(name).get();
    auto casted_prop = dynamic_cast<triangle_property_t<T> const*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto add_triangle_property(std::string const& name, T const& value = T{})
      -> auto& {
    auto [it, suc] = m_triangle_properties.insert(
        std::pair{name, std::make_unique<triangle_property_t<T>>(value)});
    auto fprop = dynamic_cast<triangle_property_t<T>*>(it->second.get());
    fprop->resize(num_triangles());
    return *fprop;
  }
  //----------------------------------------------------------------------------
  constexpr bool is_valid(triangle_index t) const {
    return boost::find(m_invalid_triangles, t) == end(m_invalid_triangles);
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
  vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
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

