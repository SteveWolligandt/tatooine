#ifndef TATOOINE_TETRAHEDRAL_MESH_H
#define TATOOINE_TETRAHEDRAL_MESH_H
//==============================================================================
#ifdef TATOOINE_HAS_CGAL_SUPPORT
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#endif

#include <tatooine/grid.h>
#include <tatooine/pointset.h>
#include <tatooine/property.h>
#include <tatooine/vtk_legacy.h>

#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
class tetrahedral_mesh : public pointset<Real, N> {
  static_assert(N >= 3, "Tetrahedal mesh needs to have at least 3 dimensions.");

 public:
  using this_t   = tetrahedral_mesh<Real, N>;
  using parent_t = pointset<Real, N>;
  using parent_t::at;
  using typename parent_t::vertex_handle;
  using typename parent_t::pos_t;
  using parent_t::operator[];
  using parent_t::is_valid;
  using parent_t::vertices;
  using parent_t::vertex_data;
  using parent_t::vertex_properties;
  using parent_t::insert_vertex;
  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  using hierarchy_t       = std::conditional_t<(N == 3), octree<Real>, void>;
  //----------------------------------------------------------------------------
  template <typename T>
  struct vertex_property_sampler_t {
    this_t const&               m_mesh;
    vertex_property_t<T> const& m_prop;
    //--------------------------------------------------------------------------
    auto mesh() const -> auto const& { return m_mesh; }
    auto property() const -> auto const& { return m_prop; }
    //--------------------------------------------------------------------------
    [[nodiscard]] auto operator()(Real x, Real y, Real z) const {
      return sample(pos_t{x, y, z});
    }
    [[nodiscard]] auto operator()(pos_t const& x) const { return sample(x); }
    [[nodiscard]] auto sample(Real x, Real y, Real z) const {
      return sample(pos_t{x, y, z});
    }
    [[nodiscard]] auto sample(pos_t const& x) const -> T {
      auto tet_handles = m_mesh.hierarchy().nearby_tetrahedrons(x);
      if (tet_handles.empty()) {
        throw std::runtime_error{
            "[vertex_property_sampler_t::sample] out of domain"};
      }
      for (auto t : tet_handles) {
        auto const [vi0, vi1, vi2, vi3] = m_mesh.tetrahedron_at(t);

        auto const& v0 = m_mesh.vertex_at(vi0);
        auto const& v1 = m_mesh.vertex_at(vi1);
        auto const& v2 = m_mesh.vertex_at(vi2);
        auto const& v3 = m_mesh.vertex_at(vi3);

        mat<Real, 4, 4> const A{{v0(0), v1(0), v2(0), v3(0)},
                                {v0(1), v1(1), v2(1), v3(1)},
                                {v0(2), v1(2), v2(2), v3(2)},
                                {Real(1), Real(1), Real(1), Real(1)}};
        vec<Real, 4> const    b{x(0), x(1), x(2), 1};
        auto const            abcd = solve(A, b);
        if (abcd(0) >= -1e-6 && abcd(0) <= 1 + 1e-6 &&
            abcd(1) >= -1e-6 && abcd(1) <= 1 + 1e-6 &&
            abcd(2) >= -1e-6 && abcd(2) <= 1 + 1e-6 &&
            abcd(3) >= -1e-6 && abcd(3) <= 1 + 1e-6) {
          return m_prop[vi0] * abcd(0) +
                 m_prop[vi1] * abcd(1) +
                 m_prop[vi2] * abcd(2) +
                 m_prop[vi3] * abcd(3);
        }
      }
      throw std::runtime_error{
          "[vertex_property_sampler_t::sample] out of domain"};
      return T{};
    }
  };
  //----------------------------------------------------------------------------
  struct tetrahedron_index : handle {
    using handle::handle;
    constexpr bool operator==(tetrahedron_index other) const {
      return this->i == other.i;
    }
    constexpr bool operator!=(tetrahedron_index other) const {
      return this->i != other.i;
    }
    constexpr bool operator<(tetrahedron_index other) const {
      return this->i < other.i;
    }
    static constexpr auto invalid() {
      return tetrahedron_index{handle::invalid_idx};
    }
  };
  //----------------------------------------------------------------------------
  struct tetrahedron_iterator
      : boost::iterator_facade<tetrahedron_iterator, tetrahedron_index,
                               boost::bidirectional_traversal_tag,
                               tetrahedron_index> {
    tetrahedron_iterator(tetrahedron_index i, tetrahedral_mesh const* mesh)
        : m_index{i}, m_mesh{mesh} {}
    tetrahedron_iterator(tetrahedron_iterator const& other)
        : m_index{other.m_index}, m_mesh{other.m_mesh} {}

   private:
    tetrahedron_index       m_index;
    tetrahedral_mesh const* m_mesh;

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

    auto equal(tetrahedron_iterator const& other) const {
      return m_index == other.m_index;
    }
    auto dereference() const { return m_index; }
  };
  //----------------------------------------------------------------------------
  struct tetrahedron_container {
    using iterator       = tetrahedron_iterator;
    using const_iterator = tetrahedron_iterator;
    //--------------------------------------------------------------------------
    tetrahedral_mesh const* m_mesh;
    //--------------------------------------------------------------------------
    auto begin() const {
      tetrahedron_iterator vi{tetrahedron_index{0}, m_mesh};
      if (!m_mesh->is_valid(*vi)) {
        ++vi;
      }
      return vi;
    }
    //--------------------------------------------------------------------------
    auto end() const {
      return tetrahedron_iterator{
          tetrahedron_index{size()}, m_mesh};
    }
    //--------------------------------------------------------------------------
    auto size() const { return m_mesh->m_tet_indices.size() / 4; }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  using tetrahedron_property_t = vector_property_impl<tetrahedron_index, T>;
  using tetrahedron_property_container_t =
      std::map<std::string,
               std::unique_ptr<vector_property<tetrahedron_index>>>;
  //============================================================================
 private:
  std::vector<vertex_handle>           m_tet_indices;
  std::vector<tetrahedron_index>       m_invalid_tetrahedrons;
  tetrahedron_property_container_t     m_tetrahedron_properties;
  mutable std::unique_ptr<hierarchy_t> m_hierarchy;

 public:
  //============================================================================
  constexpr tetrahedral_mesh() = default;
  //============================================================================
 public:
  tetrahedral_mesh(tetrahedral_mesh const& other)
      : parent_t{other}, m_tet_indices{other.m_tet_indices} {
    for (auto const& [key, prop] : other.m_tetrahedron_properties) {
      m_tetrahedron_properties.insert(std::pair{key, prop->clone()});
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  tetrahedral_mesh(tetrahedral_mesh&& other) noexcept = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(tetrahedral_mesh const& other) -> tetrahedral_mesh& {
    parent_t::operator=(other);
    m_tetrahedron_properties.clear();
    m_tet_indices = other.m_tet_indices;
    for (auto const& [key, prop] : other.m_tetrahedron_properties) {
      m_tetrahedron_properties.insert(std::pair{key, prop->clone()});
    }
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(tetrahedral_mesh&& other) noexcept
      -> tetrahedral_mesh& = default;
  //----------------------------------------------------------------------------
  tetrahedral_mesh(std::string const& file) { read(file); }
  //----------------------------------------------------------------------------
  tetrahedral_mesh(std::vector<vec<Real, N>> const& positions) : parent_t{positions} {}
  tetrahedral_mesh(std::vector<vec<Real, N>>&& positions)
      : parent_t{std::move(positions)} {}
  //----------------------------------------------------------------------------
 private:
  //template <typename T, typename Prop, typename Grid>
  //auto copy_prop(std::string const& name, Prop const& prop, Grid const& g) {
  //  if (prop->type() == typeid(T)) {
  //    auto const& grid_prop = g.template vertex_property<T>(name);
  //    auto&       tri_prop  = this->template add_vertex_property<T>(name);
  //    g.loop_over_vertex_indices([&](auto const... is) {
  //      std::array is_arr{is...};
  //      tri_prop[vertex_handle{is_arr[0] + is_arr[1] * g.size(0)}] =
  //          grid_prop(is...);
  //    });
  //  }
  //}

 public:
#ifdef __cpp_concepts
  template <indexable_space DimX, indexable_space DimY, indexable_space DimZ>
  requires(N == 3)
#else
  template <typename DimX, typename DimY, typename DimZ, size_t N_ = N,
            enable_if<N_ == 3> = true>
#endif
      tetrahedral_mesh(grid<DimX, DimY, DimZ> const& g) {

    constexpr auto turned = [](size_t const ix, size_t const iy,
                     size_t const iz) -> bool {
      bool const xodd = ix % 2 == 0;
      bool const yodd = iy % 2 == 0;
      bool const zodd = iz % 2 == 0;

      bool turned = xodd;
      if (yodd) {
        turned = !turned;
      }
      if (zodd) {
        turned = !turned;
      }
      return turned;
    };

    for (auto v : g.vertices()) {
      insert_vertex(v);
    }
    for (size_t k = 0; k < g.size(2) - 1; ++k) {
      for (size_t j = 0; j < g.size(1) - 1; ++j) {
        for (size_t i = 0; i < g.size(0) - 1; ++i) {
          auto const le_bo_fr =
              vertex_handle{i + j * g.size(0) + k * g.size(0) * g.size(1)};
          auto const ri_bo_fr =
            vertex_handle{(i + 1) + j * g.size(0) + k * g.size(0) * g.size(1)};
          auto const le_to_fr =
              vertex_handle{i + (j+1) * g.size(0) + k * g.size(0) * g.size(1)};
          auto const ri_to_fr =
            vertex_handle{(i + 1) + (j+1) * g.size(0) + k * g.size(0) * g.size(1)};
          auto const le_bo_ba =
              vertex_handle{i + j * g.size(0) + (k+1) * g.size(0) * g.size(1)};
          auto const ri_bo_ba =
            vertex_handle{(i + 1) + j * g.size(0) + (k+1) * g.size(0) * g.size(1)};
          auto const le_to_ba =
              vertex_handle{i + (j+1) * g.size(0) + (k+1) * g.size(0) * g.size(1)};
          auto const ri_to_ba =
            vertex_handle{(i + 1) + (j+1) * g.size(0) + (k+1) * g.size(0) * g.size(1)};
          if (turned(i, j, k)) {
            insert_tetrahedron(
                le_bo_fr, ri_bo_ba, ri_to_fr, le_to_ba);  // inner
            insert_tetrahedron(
                le_bo_fr, ri_bo_fr, ri_to_fr, ri_bo_ba);  // right front
            insert_tetrahedron(
                le_bo_fr, ri_to_fr, le_to_fr, le_to_ba);  // left front
            insert_tetrahedron(
                ri_to_fr, ri_bo_ba, ri_to_ba, le_to_ba);  // right back
            insert_tetrahedron(
                le_bo_fr, le_bo_ba, ri_bo_ba, le_to_ba);  // left back
          } else {
            insert_tetrahedron(
                le_to_fr, ri_bo_fr, le_bo_ba, ri_to_ba);  // inner
            insert_tetrahedron(
                le_bo_fr, ri_bo_fr, le_to_fr, le_bo_ba);  // left front
            insert_tetrahedron(
                ri_bo_fr, ri_to_fr, le_to_fr, ri_to_ba);  // right front
            insert_tetrahedron(
                le_to_fr, le_to_ba, ri_to_ba, le_bo_ba);  // left back
            insert_tetrahedron(
                ri_bo_fr, ri_bo_ba, ri_to_ba, le_bo_ba);  // right back
          }
        }
      }
    }
    //for (auto const& [name, prop] : g.vertex_properties()) {
    //  copy_prop<vec4d>(name, prop, g);
    //  copy_prop<vec3d>(name, prop, g);
    //  copy_prop<vec2d>(name, prop, g);
    //  copy_prop<vec4f>(name, prop, g);
    //  copy_prop<vec3f>(name, prop, g);
    //  copy_prop<vec2f>(name, prop, g);
    //  copy_prop<double>(name, prop, g);
    //  copy_prop<float>(name, prop, g);
    //}
  }
  //============================================================================
  auto operator[](tetrahedron_index t) const -> auto {
    return tetrahedron_at(t.i);
  }
  auto operator[](tetrahedron_index t) -> auto { return tetrahedron_at(t.i); }
  //----------------------------------------------------------------------------
  auto at(tetrahedron_index t) const -> auto { return tetrahedron_at(t.i); }
  auto at(tetrahedron_index t) -> auto { return tetrahedron_at(t.i); }
  //----------------------------------------------------------------------------
  auto tetrahedron_at(tetrahedron_index t) const -> auto {
    return tetrahedron_at(t.i);
  }
  auto tetrahedron_at(tetrahedron_index t) -> auto {
    return tetrahedron_at(t.i);
  }
  //----------------------------------------------------------------------------
  auto tetrahedron_at(size_t const i) const
      -> std::tuple<vertex_handle const&, vertex_handle const&,
                    vertex_handle const&, vertex_handle const&> {
    return {m_tet_indices[i * 4], m_tet_indices[i * 4 + 1],
            m_tet_indices[i * 4 + 2], m_tet_indices[i * 4 + 3]};
  }
  auto tetrahedron_at(size_t const i)
      -> std::tuple<vertex_handle&, vertex_handle&, vertex_handle&,
                    vertex_handle&> {
    return {m_tet_indices[i * 4], m_tet_indices[i * 4 + 1],
            m_tet_indices[i * 4 + 2], m_tet_indices[i * 4 + 3]};
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
  auto insert_tetrahedron(vertex_handle const v0, vertex_handle const v1,
                          vertex_handle const v2, vertex_handle const v3) {
    m_tet_indices.push_back(v0);
    m_tet_indices.push_back(v1);
    m_tet_indices.push_back(v2);
    m_tet_indices.push_back(v3);
    for (auto& [key, prop] : m_tetrahedron_properties) {
      prop->push_back();
    }
    tetrahedron_index ti{tetrahedrons().size() - 1};
    if (m_hierarchy != nullptr) {
      if (!m_hierarchy->insert_tetrahedron(*this, ti.i)) {
        build_hierarchy();
      }
    }
    return ti;
  }
  //----------------------------------------------------------------------------
  auto insert_tetrahedron(size_t const v0, size_t const v1, size_t const v2,
                          size_t const v3) {
    return insert_tetrahedron(vertex_handle{v0}, vertex_handle{v1},
                              vertex_handle{v2}, vertex_handle{v3});
  }
  //----------------------------------------------------------------------------
  auto insert_tetrahedron(std::array<vertex_handle, 4> const& t) {
    return insert_tetrahedron(vertex_handle{t[0]}, vertex_handle{t[1]},
                              vertex_handle{t[2]}, vertex_handle{t[3]});
  }
  //----------------------------------------------------------------------------
  auto clear() {
    parent_t::clear();
    m_tet_indices.clear();
  }
  //----------------------------------------------------------------------------
  auto tetrahedrons() const { return tetrahedron_container{this}; }
  //----------------------------------------------------------------------------
#ifdef TATOOINE_HAS_CGAL_SUPPORT
#ifdef __cpp_concepts
  template <typename = void> requires(N == 3)
#else
  template <size_t N_ = N, enable_if<N_ == 3> = true>
#endif
  auto build_delaunay_mesh() -> void {
    m_tet_indices.clear();
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vb =
        CGAL::Triangulation_vertex_base_with_info_3<vertex_handle, Kernel>;
    using Tds           = CGAL::Triangulation_data_structure_3<Vb>;
    using Triangulation = CGAL::Delaunay_triangulation_3<Kernel, Tds>;
    using Point         = typename Kernel::Point_3;
    std::vector<std::pair<Point, vertex_handle>> points;
    points.reserve(vertices().size());
    for (auto v : vertices()) {
      points.emplace_back(Point{at(v)(0), at(v)(1), at(v)(2)}, v);
    }

    Triangulation dt{begin(points), end(points)};
    for (auto it = dt.finite_cells_begin(); it != dt.finite_cells_end(); ++it) {
      insert_tetrahedron(vertex_handle{it->vertex(0)->info()},
                         vertex_handle{it->vertex(1)->info()},
                         vertex_handle{it->vertex(2)->info()},
                         vertex_handle{it->vertex(3)->info()});
    }
  }
#endif
  //----------------------------------------------------------------------------
  template <size_t _N = N, enable_if<_N == 3> = true>
  auto write_vtk(std::string const& path,
                 std::string const& title = "tatooine tetrahedral mesh") const
      -> bool {
    using boost::copy;
    using boost::adaptors::transformed;
    vtk::legacy_file_writer writer(path, vtk::dataset_type::unstructured_grid);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      writer.write_points(vertex_data());

      std::vector<std::vector<size_t>> polygons;
      polygons.reserve(tetrahedrons().size());
      std::vector<vtk::cell_type>      polygon_types(tetrahedrons().size(),
                                                vtk::cell_type::tetra);
      for (auto const t : tetrahedrons()) {
        auto const [v0, v1, v2, v3] = at(t);
        polygons.push_back(std::vector{v0.i, v1.i, v2.i, v3.i});
      }
      writer.write_cells(polygons);
      writer.write_cell_types(polygon_types);

      // write vertex_handle data
      writer.write_point_data(vertices().size());
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
  auto read(std::string const& path) {
    auto ext = path.substr(path.find_last_of(".") + 1);
    if constexpr (N == 3) {
      if (ext == "vtk") {
        read_vtk(path);
      }
    }
  }
  //----------------------------------------------------------------------------
  template <size_t _N = N, enable_if<_N == 3> = true>
  auto read_vtk(std::string const& path) {
    struct listener_t : vtk::legacy_file_listener {
      tetrahedral_mesh& mesh;

      listener_t(tetrahedral_mesh& _mesh) : mesh(_mesh) {}

      void on_dataset_type(vtk::dataset_type t) override {
        if (t != vtk::dataset_type::polydata) {
          throw std::runtime_error{
              "[tetrahedral_mesh] need polydata when reading vtk legacy"};
        }
      }

      void on_points(std::vector<std::array<float, 3>> const& ps) override {
        for (auto& p : ps) {
          mesh.insert_vertex(p[0], p[1], p[2]);
        }
      }
      void on_points(std::vector<std::array<double, 3>> const& ps) override {
        for (auto& p : ps) {
          mesh.insert_vertex(p[0], p[1], p[2]);
        }
      }
      void on_polygons(std::vector<int> const& ps) override {
        for (size_t i = 0; i < ps.size();) {
          auto const& size = ps[i++];
          if (size == 4) {
            mesh.insert_tetrahedron(size_t(ps[i]), size_t(ps[i + 1]),
                                    size_t(ps[i + 2]), size_t(ps[i + 2]));
          }
          i += size;
        }
      }
      void on_scalars(std::string const& data_name,
                      std::string const& /*lookup_table_name*/,
                      size_t num_comps, std::vector<double> const& scalars,
                      vtk::reader_data data) override {
        if (data == vtk::reader_data::point_data) {
          if (num_comps == 1) {
            auto& prop = mesh.template add_vertex_property<double>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              prop[i] = scalars[i];
            }
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
  auto tetrahedron_property(std::string const& name) -> auto& {
    auto prop        = m_tetrahedron_properties.at(name).get();
    auto casted_prop = dynamic_cast<tetrahedron_property_t<T>*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto tetrahedron_property(std::string const& name) const -> auto const& {
    auto prop        = m_tetrahedron_properties.at(name).get();
    auto casted_prop = dynamic_cast<tetrahedron_property_t<T> const*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto add_tetrahedron_property(std::string const& name, T const& value = T{})
      -> auto& {
    auto [it, suc] = m_tetrahedron_properties.insert(
        std::pair{name, std::make_unique<tetrahedron_property_t<T>>(value)});
    auto prop = dynamic_cast<tetrahedron_property_t<T>*>(it->second.get());
    prop->resize(m_tet_indices.size());
    return *prop;
  }
  //----------------------------------------------------------------------------
  constexpr bool is_valid(tetrahedron_index t) const {
    return boost::find(m_invalid_tetrahedrons, t) ==
           end(m_invalid_tetrahedrons);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires(N == 3)
#else
  template <size_t N_ = N, enable_if<(N_ == 3)> = true>
#endif
  auto build_hierarchy() const {
    clear_hierarchy();
    auto& h = hierarchy();
    for (auto v : vertices()) {
      h.insert_vertex(*this, v.i);
    }
    for (auto t : tetrahedrons()) {
      h.insert_tetrahedron(*this, t.i);
    }
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires(N == 3)
#else
  template <size_t N_ = N, enable_if<(N_ == 3)> = true>
#endif
  auto clear_hierarchy() const {
    m_hierarchy.reset();
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires(N == 3)
#else
  template <size_t N_ = N, enable_if<(N_ == 3)> = true>
#endif
  auto hierarchy() const -> auto& {
    if (m_hierarchy == nullptr) {
      auto min = pos_t::ones() * std::numeric_limits<Real>::infinity();
      auto max = -pos_t::ones() * std::numeric_limits<Real>::infinity();
      for (auto v : vertices()) {
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
};
//==============================================================================
tetrahedral_mesh()->tetrahedral_mesh<double, 3>;
tetrahedral_mesh(std::string const&)->tetrahedral_mesh<double, 3>;
template <typename... Dims>
tetrahedral_mesh(grid<Dims...> const& g)
    -> tetrahedral_mesh<typename grid<Dims...>::real_t, sizeof...(Dims)>;
//==============================================================================
// namespace detail {
// template <typename MeshCont>
// auto write_mesh_container_to_vtk(MeshContc onst& meshes, std::string const&
// path,
//                                 std::string const& title) {
//  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
//  if (writer.is_open()) {
//    size_t num_pts = 0;
//    size_t cur_first = 0;
//    for (auto const& m : meshes) { num_pts += m.num_vertices(); }
//    std::vector<std::array<typename MeshCont::value_type::real_t, 3>> points;
//    std::vector<std::vector<size_t>> tetrahedrons; points.reserve(num_pts);
//    tetrahedrons.reserve(meshes.size());
//
//    for (auto const& m : meshes) {
//      // add points
//      for (auto const& v : m.vertices()) {
//        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
//      }
//
//      // add tetrahedrons
//      for (auto t : m.tetrahedrons()) {
//        tetrahedrons.emplace_back();
//        tetrahedrons.back().push_back(cur_first + m[t][0].i);
//        tetrahedrons.back().push_back(cur_first + m[t][1].i);
//        tetrahedrons.back().push_back(cur_first + m[t][2].i);
//      }
//      cur_first += m.num_vertices();
//    }
//
//    // write
//    writer.set_title(title);
//    writer.write_header();
//    writer.write_points(points);
//    writer.write_polygons(tetrahedrons);
//    //writer.write_point_data(num_pts);
//    writer.close();
//  }
//}
//}  // namespace detail
////==============================================================================
// template <typename Real>
// auto write_vtk(std::vector<tetrahedral_mesh<Real, 3>> const& meshes,
// std::string const& path,
//               std::string const& title = "tatooine meshes") {
//  detail::write_mesh_container_to_vtk(meshes, path, title);
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif

