#ifndef TATOOINE_TRIANGULAR_MESH_H
#define TATOOINE_TRIANGULAR_MESH_H
//==============================================================================
#ifdef TATOOINE_HAS_CGAL_SUPPORT
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#endif

#include <tatooine/grid.h>
#include <tatooine/octree.h>
#include <tatooine/pointset.h>
#include <tatooine/property.h>
#include <tatooine/quadtree.h>
#include <tatooine/vtk_legacy.h>

#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/transformed.hpp>
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
  using typename parent_t::vertex_handle;
  using parent_t::operator[];
  using parent_t::is_valid;
  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  using typename parent_t::pos_t;

  template <typename T>
  struct vertex_property_sampler_t {
    this_t const&               m_mesh;
    vertex_property_t<T> const& m_prop;
    //--------------------------------------------------------------------------
    auto mesh() const -> auto const& { return m_mesh; }
    auto property() const -> auto const& { return m_prop; }
    //--------------------------------------------------------------------------
    [[nodiscard]] auto operator()(Real x, Real y) const { return sample(pos_t{x, y}); }
    [[nodiscard]] auto operator()(pos_t const& x) const { return sample(x); }
    [[nodiscard]] auto sample(Real x, Real y) const { return sample(pos_t{x, y}); }
    [[nodiscard]] auto sample(pos_t const& x) const -> T {
      auto face_handles = m_mesh.hierarchy().nearby_faces(x);
      if (face_handles.empty()) {
        throw std::runtime_error{
            "[vertex_property_sampler_t::sample] out of domain"};
      }
      for (auto f : face_handles) {
        auto [vi0, vi1, vi2]     = m_mesh.face_at(f);
        auto const&           v0 = m_mesh.vertex_at(vi0);
        auto const&           v1 = m_mesh.vertex_at(vi1);
        auto const&           v2 = m_mesh.vertex_at(vi2);
        mat<Real, 3, 3> const A{{v0(0), v1(0), v2(0)},
                                {v0(1), v1(1), v2(1)},
                                {Real(1), Real(1), Real(1)}};
        vec<Real, 3> const    b{x(0), x(1), 1};
        auto const            abc = solve(A, b);
        if (abc(0) >= -1e-6 && abc(0) <= 1 + 1e-6 &&
            abc(1) >= -1e-6 && abc(1) <= 1 + 1e-6 &&
            abc(2) >= -1e-6 && abc(2) <= 1 + 1e-6) {
          return m_prop[vi0] * abc(0) +
                 m_prop[vi1] * abc(1) +
                 m_prop[vi2] * abc(2);
        }
      }
      throw std::runtime_error{
          "[vertex_property_sampler_t::sample] out of domain"};
      return T{};
    }
  };
  //----------------------------------------------------------------------------
  struct face_handle : handle {
    using handle::handle;
    constexpr bool operator==(face_handle other) const {
      return this->i == other.i;
    }
    constexpr bool operator!=(face_handle other) const {
      return this->i != other.i;
    }
    constexpr bool operator<(face_handle other) const {
      return this->i < other.i;
    }
    static constexpr auto invalid() {
      return face_handle{handle::invalid_idx};
    }
  };
  //----------------------------------------------------------------------------
  struct face_iterator
      : boost::iterator_facade<face_iterator, face_handle,
                               boost::bidirectional_traversal_tag,
                               face_handle> {
    face_iterator(face_handle i, triangular_mesh const* mesh)
        : m_index{i}, m_mesh{mesh} {}
    face_iterator(face_iterator const& other)
        : m_index{other.m_index}, m_mesh{other.m_mesh} {}

   private:
    face_handle         m_index;
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

    auto equal(face_iterator const& other) const {
      return m_index == other.m_index;
    }
    auto dereference() const { return m_index; }
  };
  //----------------------------------------------------------------------------
  struct face_container {
    using iterator       = face_iterator;
    using const_iterator = face_iterator;

    triangular_mesh const* m_mesh;

    auto begin() const {
      face_iterator vi{face_handle{0}, m_mesh};
      if (!m_mesh->is_valid(*vi)) {
        ++vi;
      }
      return vi;
    }

    auto end() const {
      return face_iterator{face_handle{m_mesh->num_faces()}, m_mesh};
    }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  using face_property_t = vector_property_impl<face_handle, T>;
  using face_property_container_t =
      std::map<std::string, std::unique_ptr<vector_property<face_handle>>>;
  using hierarchy_t =
      std::conditional_t<N == 2, quadtree<Real>,
                         std::conditional_t<N == 3, octree<Real>, void>>;
  //============================================================================
 private:
  std::vector<vertex_handle>            m_face_indices;
  std::vector<face_handle>          m_invalid_faces;
  face_property_container_t        m_face_properties;
  mutable std::unique_ptr<hierarchy_t> m_hierarchy;

 public:
  //============================================================================
  constexpr triangular_mesh() = default;
  //============================================================================
 public:
  triangular_mesh(triangular_mesh const& other)
      : parent_t{other}, m_face_indices{other.m_face_indices} {
    for (auto const& [key, fprop] : other.m_face_properties) {
      m_face_properties.insert(std::pair{key, fprop->clone()});
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  triangular_mesh(triangular_mesh&& other) noexcept = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(triangular_mesh const& other) -> triangular_mesh& {
    parent_t::operator=(other);
    m_face_properties.clear();
    m_face_indices = other.m_face_indices;
    for (auto const& [key, fprop] : other.m_face_properties) {
      m_face_properties.insert(std::pair{key, fprop->clone()});
    }
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator           =(triangular_mesh&& other) noexcept
      -> triangular_mesh& = default;
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <indexable_space DimX, indexable_space DimY>
  requires(N == 2)
#else
  template <typename DimX, typename DimY, size_t _N = N,
            enable_if<_N == 2> = true>
#endif
  triangular_mesh(grid<DimX, DimY> const& g) {
    for (auto v : g.vertices()) {
      insert_vertex(v);
    }
    for (size_t j = 0; j < g.size(1) - 1; ++j) {
      for (size_t i = 0; i < g.size(0) - 1; ++i) {
        insert_face(vertex_handle{i + j * g.size(0)},
                    vertex_handle{(i + 1) + j * g.size(0)},
                    vertex_handle{i + (j + 1) * g.size(0)});
        insert_face(vertex_handle{(i + 1) + j * g.size(0)},
                    vertex_handle{(i + 1) + (j + 1) * g.size(0)},
                    vertex_handle{i + (j + 1) * g.size(0)});
      }
    }
    auto copy_prop = [&]<typename T>(auto const& name, auto const& prop) {
      if (prop->type() == typeid(T)) {
        auto const& grid_prop = g.template vertex_property<T>(name);
        auto&       tri_prop  = this->template add_vertex_property<T>(name);
        g.loop_over_vertex_indices([&](auto const... is) {
          std::array is_arr{is...};
          tri_prop[vertex_handle{is_arr[0] + is_arr[1] * g.size(0)}] =
              grid_prop(is...);
        });
      }
    };
    for (auto const& [name, prop] : g.vertex_properties()) {
      copy_prop.template operator()<vec4d>(name, prop);
      copy_prop.template operator()<vec3d>(name, prop);
      copy_prop.template operator()<vec2d>(name, prop);
      copy_prop.template operator()<vec4f>(name, prop);
      copy_prop.template operator()<vec3f>(name, prop);
      copy_prop.template operator()<vec2f>(name, prop);
      copy_prop.template operator()<double>(name, prop);
      copy_prop.template operator()<float>(name, prop);
    }
  }
  //----------------------------------------------------------------------------
  triangular_mesh(std::filesystem::path const& file) { read(file); }
  //============================================================================
  auto operator[](face_handle const t) const { return face_at(t.i); }
  auto operator[](face_handle const t) { return face_at(t.i); }
  //----------------------------------------------------------------------------
  auto at(face_handle t) const { return face_at(t.i); }
  auto at(face_handle t) { return face_at(t.i); }
  //----------------------------------------------------------------------------
  auto face_at(face_handle const t) const { return face_at(t.i); }
  auto face_at(face_handle const t) { return face_at(t.i); }
  //----------------------------------------------------------------------------
  auto face_at(size_t const i) const
      -> std::tuple<vertex_handle const&, vertex_handle const&,
                    vertex_handle const&> {
    return {m_face_indices[i * 3], m_face_indices[i * 3 + 1],
            m_face_indices[i * 3 + 2]};
  }
  auto face_at(size_t const i)
      -> std::tuple<vertex_handle&, vertex_handle&, vertex_handle&> {
    return {m_face_indices[i * 3], m_face_indices[i * 3 + 1],
            m_face_indices[i * 3 + 2]};
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <arithmetic... Ts>
  requires(sizeof...(Ts) == N)
#else
  template <typename... Ts, enable_if<is_arithmetic<Ts...>> = true,
            enable_if<sizeof...(Ts) == N> = true>
#endif
  auto insert_vertex(Ts const... ts) {
    auto const vi = parent_t::insert_vertex(ts...);
    if (m_hierarchy != nullptr) {
      if (!m_hierarchy->insert_vertex(*this, vi.i)) {
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
        build_hierarchy();
      }
    }
    return vi;
  }
  //----------------------------------------------------------------------------
  auto insert_face(vertex_handle v0, vertex_handle v1, vertex_handle v2) {
    m_face_indices.push_back(v0);
    m_face_indices.push_back(v1);
    m_face_indices.push_back(v2);
    auto const ti = face_handle{size(m_face_indices) / 3 - 1};
    for (auto& [key, prop] : m_face_properties) {
      prop->push_back();
    }
    if (m_hierarchy != nullptr) {
      if (!m_hierarchy->insert_face(*this, ti.i)) {
        build_hierarchy();
      }
    }
    return ti;
  }
  //----------------------------------------------------------------------------
  auto insert_face(size_t v0, size_t v1, size_t v2) {
    return insert_face(vertex_handle{v0}, vertex_handle{v1}, vertex_handle{v2});
  }
  //----------------------------------------------------------------------------
  auto clear() {
    parent_t::clear();
    m_face_indices.clear();
  }
  //----------------------------------------------------------------------------
  auto faces() const { return face_container{this}; }
  //----------------------------------------------------------------------------
  auto num_faces() const { return m_face_indices.size() / 3; }
  //----------------------------------------------------------------------------
#ifdef TATOOINE_HAS_CGAL_SUPPORT
  template <typename = void> requires(N == 2)
  auto triangulate_delaunay() -> void {
    m_face_indices.clear();
    //using Kernel = CGAL::Cartesian<Real>;
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vb     = CGAL::Triangulation_vertex_base_with_info_2<vertex_handle, Kernel>;
    using Tds    = CGAL::Triangulation_data_structure_2<Vb>;
    using Triangulation = CGAL::Delaunay_triangulation_2<Kernel, Tds>;
    using Point         = typename Kernel::Point_2;
    std::vector<std::pair<Point, vertex_handle>> points;
    points.reserve(this->num_vertices());
    for (auto v : this->vertices()) {
      points.emplace_back(Point{at(v)(0), at(v)(1)}, v);
    }

    Triangulation dt{begin(points), end(points)};
    for (auto it = dt.finite_faces_begin(); it != dt.finite_faces_end(); ++it) {
      insert_face(vertex_handle{it->vertex(0)->info()},
                  vertex_handle{it->vertex(1)->info()},
                  vertex_handle{it->vertex(2)->info()});
    }
  }
#endif
  //----------------------------------------------------------------------------
  auto set_face_indices(std::vector<size_t>&& is) {
    m_face_indices =
        std::move(*reinterpret_cast<std::vector<vertex_handle>*>(&is));
  }
  //----------------------------------------------------------------------------
  auto set_face_indices(std::vector<size_t> const& is) {
    m_face_indices = *reinterpret_cast<std::vector<vertex_handle>*>(&is);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires(N == 2 || N == 3)
#else
  template <size_t _N = N, enable_if<(_N == 2 || _N == 3)> = true>
#endif
  auto build_hierarchy() const {
    clear_hierarchy();
    auto& h = hierarchy();
    for (auto v : this->vertices()) {
      h.insert_vertex(*this, v.i);
    }
    for (auto t : faces()) {
      h.insert_face(*this, t.i);
    }
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires(N == 2 || N == 3)
#else
  template <size_t _N = N, enable_if<(_N == 2 || _N == 3)> = true>
#endif
  auto clear_hierarchy() const {
    m_hierarchy.reset();
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires(N == 2 || N == 3)
#else
  template <size_t _N = N, enable_if<(_N == 2 || _N == 3)> = true>
#endif
  auto hierarchy() const -> auto& {
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
  auto sampler(vertex_property_t<T> const& prop) const {
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
#ifdef __cpp_concepts
  template <typename = void>
  requires(N == 2 || N == 3)
#else
  template <size_t _N = N, enable_if<(_N == 2 || _N == 3)> = true>
#endif
  auto write_vtk(std::string const& path,
                 std::string const& title = "tatooine triangular mesh") const
      -> bool {
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
      polygons.reserve(num_faces());
      for (size_t i = 0; i < size(m_face_indices); i += 3) {
        polygons.push_back(std::vector{m_face_indices[i].i,
                                       m_face_indices[i + 1].i,
                                       m_face_indices[i + 2].i});
      }
      writer.write_polygons(polygons);

      // write vertex_handle data
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
#ifdef __cpp_concepts
  template <typename = void>
  requires(N == 2 || N == 3)
#else
  template <size_t _N = N, enable_if<(_N == 2 || _N == 3)> = true>
#endif
  auto read_vtk(std::filesystem::path const& path) {
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
              prop[vertex_handle{i}] = scalars[i];
            }
          } else if (num_comps == 2) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 2>>(data_name);

            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[vertex_handle{i}][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 3) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 3>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[vertex_handle{i}][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 4) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 4>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[vertex_handle{i}][j] = scalars[i * num_comps + j];
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
  auto face_property(std::string const& name) -> auto& {
    auto prop        = m_face_properties.at(name).get();
    auto casted_prop = dynamic_cast<face_property_t<T>*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto face_property(std::string const& name) const -> auto const& {
    auto prop        = m_face_properties.at(name).get();
    auto casted_prop = dynamic_cast<face_property_t<T> const*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto add_face_property(std::string const& name, T const& value = T{})
      -> auto& {
    auto [it, suc] = m_face_properties.insert(
        std::pair{name, std::make_unique<face_property_t<T>>(value)});
    auto fprop = dynamic_cast<face_property_t<T>*>(it->second.get());
    fprop->resize(num_faces());
    return *fprop;
  }
  //----------------------------------------------------------------------------
  constexpr bool is_valid(face_handle t) const {
    return boost::find(m_invalid_faces, t) == end(m_invalid_faces);
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
    std::vector<std::vector<size_t>>                                  faces;
    points.reserve(num_pts);
    faces.reserve(meshes.size());

    for (auto const& m : meshes) {
      // add points
      for (auto const& v : m.vertices()) {
        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
      }

      // add faces
      for (auto t : m.faces()) {
        faces.emplace_back();
        faces.back().push_back(cur_first + m[t][0].i);
        faces.back().push_back(cur_first + m[t][1].i);
        faces.back().push_back(cur_first + m[t][2].i);
      }
      cur_first += m.num_vertices();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_polygons(faces);
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

