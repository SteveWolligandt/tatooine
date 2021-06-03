#ifndef TATOOINE_SIMPLEX_MESH_H
#define TATOOINE_SIMPLEX_MESH_H
//==============================================================================
#ifdef TATOOINE_HAS_CGAL_SUPPORT
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#endif

#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/celltree.h>
#include <tatooine/grid.h>
#include <tatooine/kdtree.h>
#include <tatooine/octree.h>
#include <tatooine/pointset.h>
#include <tatooine/property.h>
#include <tatooine/quadtree.h>
#include <tatooine/vtk_legacy.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename VertexHandle, size_t NumVerticesPerSimplex, size_t I = 0,
          typename... Ts>
struct simplex_mesh_cell_at_return_type_impl {
  using type = typename simplex_mesh_cell_at_return_type_impl<
      VertexHandle, NumVerticesPerSimplex, I + 1, Ts..., VertexHandle>::type;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename VertexHandle, size_t NumVerticesPerSimplex, typename... Ts>
struct simplex_mesh_cell_at_return_type_impl<VertexHandle, NumVerticesPerSimplex,
                                             NumVerticesPerSimplex, Ts...> {
  using type = std::tuple<Ts...>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename VertexHandle, size_t NumVerticesPerSimplex>
using simplex_mesh_cell_at_return_type =
    typename simplex_mesh_cell_at_return_type_impl<VertexHandle,
                                                   NumVerticesPerSimplex>::type;
//==============================================================================
template <typename Mesh, typename Real, size_t NumDimensions, size_t SimplexDim>
struct simplex_mesh_hierarchy {
  using type = void;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real, size_t NumDimensions, size_t SimplexDim>
using simplex_mesh_hierarchy_t =
    typename simplex_mesh_hierarchy<Mesh, Real, NumDimensions,
                                    SimplexDim>::type;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real>
struct simplex_mesh_hierarchy<Mesh, Real, 3, 3> {
  using type = celltree<Mesh>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real>
struct simplex_mesh_hierarchy<Mesh, Real, 2, 2> {
  using type = celltree<Mesh>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real>
struct simplex_mesh_hierarchy<Mesh, Real, 3, 2> {
  using type = celltree<Mesh>;
};
//==============================================================================
template <typename Mesh, typename Real, size_t NumDimensions, size_t SimplexDim>
struct simplex_mesh_parent : pointset<Real, NumDimensions> {
  using typename pointset<Real, NumDimensions>::vertex_handle;
  using hierarchy_t =
      simplex_mesh_hierarchy_t<Mesh, Real, NumDimensions, SimplexDim>;
  using const_cell_at_return_type =
      simplex_mesh_cell_at_return_type<vertex_handle const&, SimplexDim + 1>;
  using cell_at_return_type =
      simplex_mesh_cell_at_return_type<vertex_handle&, SimplexDim + 1>;
};
//==============================================================================
template <typename Mesh, typename Real>
struct simplex_mesh_parent<Mesh, Real, 3, 2> : pointset<Real, 3>, ray_intersectable<Real> {
  using real_t = Real;
  using typename pointset<real_t, 3>::vertex_handle;
  using hierarchy_t = simplex_mesh_hierarchy_t<Mesh, real_t, 3, 2>;
  using const_cell_at_return_type =
      simplex_mesh_cell_at_return_type<vertex_handle const&, 3>;
  using cell_at_return_type =
      simplex_mesh_cell_at_return_type<vertex_handle&, 3>;

  using typename ray_intersectable<real_t>::ray_t;
  using typename ray_intersectable<real_t>::intersection_t;
  using typename ray_intersectable<real_t>::optional_intersection_t;
  //----------------------------------------------------------------------------
  auto as_mesh() const -> auto const& {
    return *dynamic_cast<Mesh const*>(this);
  }
  //----------------------------------------------------------------------------
  auto check_intersection(ray_t const& r, real_t const min_t = 0) const
      -> optional_intersection_t override {
    constexpr double eps          = 1e-6;
    auto const&      mesh         = as_mesh();
    auto             global_min_t = std::numeric_limits<real_t>::max();
    auto             inters       = optional_intersection_t{};
    if (!mesh.m_hierarchy) {
      mesh.build_hierarchy();
    }
    auto const possible_cells =
        mesh.m_hierarchy->collect_possible_intersections(r);
    for (auto const cell_handle : possible_cells) {
      auto const [vi0, vi1, vi2] = mesh.cell_at(cell_handle);
      auto const&      v0        = mesh.at(vi0);
      auto const&      v1        = mesh.at(vi1);
      auto const&      v2        = mesh.at(vi2);
      auto const       v0v1      = v1 - v0;
      auto const       v0v2      = v2 - v0;
      auto const       pvec      = cross(r.direction(), v0v2);
      auto const       det       = dot(v0v1, pvec);
      // r and triangle are parallel if det is close to 0
      if (std::abs(det) < eps) {
        continue;
      }
      auto const inv_det = 1 / det;

      auto const tvec = r.origin() - v0;
      auto const u    = dot(tvec, pvec) * inv_det;
      if (u < 0 || u > 1) {
        continue;
      }

      auto const qvec = cross(tvec, v0v1);
      auto const v    = dot(r.direction(), qvec) * inv_det;
      if (v < 0 || u + v > 1) {
        continue;
      }

      auto const t                 = dot(v0v2, qvec) * inv_det;
      auto const barycentric_coord = vec<real_t, 3>{1 - u - v, u, v};
      if (t > min_t) {
        auto const pos = barycentric_coord(0) * v0 +
                         barycentric_coord(1) * v1 +
                         barycentric_coord(2) * v2;

        if (t < global_min_t) {
          global_min_t = t;
          inters       = intersection_t{
              this, r, t, pos, normalize(cross(v0v1, v2 - v1)), vec2{0, 0}};
        }
      }
    }

    return inters;
  }
};
//==============================================================================
template <typename Real, size_t NumDimensions,
          size_t SimplexDim = NumDimensions>
class simplex_mesh
    : public simplex_mesh_parent<simplex_mesh<Real, NumDimensions, SimplexDim>,
                                 Real, NumDimensions, SimplexDim> {
 public:
  using this_t   = simplex_mesh<Real, NumDimensions, SimplexDim>;
  using parent_t = simplex_mesh_parent<this_t, Real, NumDimensions, SimplexDim>;
  friend struct simplex_mesh_parent<this_t, Real, NumDimensions, SimplexDim>;
  using parent_t::at;
  using parent_t::num_dimensions;
  using typename parent_t::pos_t;
  using typename parent_t::vertex_handle;
  using parent_t::operator[];
  using parent_t::insert_vertex;
  using parent_t::is_valid;
  using parent_t::vertex_data;
  using parent_t::vertex_properties;
  using parent_t::vertices;

  using typename parent_t::cell_at_return_type;
  using typename parent_t::const_cell_at_return_type;

  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  using hierarchy_t =
      simplex_mesh_hierarchy_t<this_t, Real, NumDimensions, SimplexDim>;
  static constexpr auto num_vertices_per_simplex() { return SimplexDim + 1; }
  static constexpr auto simplex_dimension() { return SimplexDim; }
  //----------------------------------------------------------------------------
  // template <typename T>
  // struct vertex_property_sampler_t {
  //  this_t const&               m_mesh;
  //  vertex_property_t<T> const& m_prop;
  //  //--------------------------------------------------------------------------
  //  auto mesh() const -> auto const& { return m_mesh; }
  //  auto property() const -> auto const& { return m_prop; }
  //  //--------------------------------------------------------------------------
  //  [[nodiscard]] auto operator()(Real const x, Real const y,
  //                                Real const z) const {
  //    return sample(pos_t{x, y, z});
  //  }
  //  [[nodiscard]] auto operator()(pos_t const& x) const { return sample(x); }
  //  //--------------------------------------------------------------------------
  //  [[nodiscard]] auto sample(Real const x, Real const y, Real const z) const
  //  {
  //    return sample(pos_t{x, y, z});
  //  }
  //  [[nodiscard]] auto sample(pos_t const& x) const -> T {
  //    auto cell_handles = m_mesh.hierarchy().nearby_cells(x);
  //    if (cell_handles.empty()) {
  //      throw std::runtime_error{
  //          "[vertex_property_sampler_t::sample] out of domain"};
  //    }
  //    for (auto t : cell_handles) {
  //      auto const [vi0, vi1, vi2, vi3] = m_mesh.cell_at(t);
  //
  //      auto const& v0 = m_mesh.vertex_at(vi0);
  //      auto const& v1 = m_mesh.vertex_at(vi1);
  //      auto const& v2 = m_mesh.vertex_at(vi2);
  //      auto const& v3 = m_mesh.vertex_at(vi3);
  //
  //      mat<Real, 4, 4> const A{{v0(0), v1(0), v2(0), v3(0)},
  //                              {v0(1), v1(1), v2(1), v3(1)},
  //                              {v0(2), v1(2), v2(2), v3(2)},
  //                              {Real(1), Real(1), Real(1), Real(1)}};
  //      vec<Real, 4> const    b{x(0), x(1), x(2), 1};
  //      auto const            abcd = solve(A, b);
  //      if (abcd(0) >= -1e-8 && abcd(0) <= 1 + 1e-8 && abcd(1) >= -1e-8 &&
  //          abcd(1) <= 1 + 1e-8 && abcd(2) >= -1e-8 && abcd(2) <= 1 + 1e-8 &&
  //          abcd(3) >= -1e-8 && abcd(3) <= 1 + 1e-8) {
  //        return m_prop[vi0] * abcd(0) + m_prop[vi1] * abcd(1) +
  //               m_prop[vi2] * abcd(2) + m_prop[vi3] * abcd(3);
  //      }
  //    }
  //    throw std::runtime_error{
  //        "[vertex_property_sampler_t::sample] out of domain"};
  //    return T{};
  //  }
  //};
  //----------------------------------------------------------------------------
  struct cell_handle : handle {
    using handle::handle;
    constexpr bool operator==(cell_handle other) const {
      return this->i == other.i;
    }
    constexpr bool operator!=(cell_handle other) const {
      return this->i != other.i;
    }
    constexpr bool operator<(cell_handle other) const {
      return this->i < other.i;
    }
    static constexpr auto invalid() { return cell_handle{handle::invalid_idx}; }
  };
  //----------------------------------------------------------------------------
  struct cell_iterator
      : boost::iterator_facade<cell_iterator, cell_handle,
                               boost::bidirectional_traversal_tag,
                               cell_handle> {
    cell_iterator(cell_handle i, simplex_mesh const* mesh)
        : m_index{i}, m_mesh{mesh} {}
    cell_iterator(cell_iterator const& other)
        : m_index{other.m_index}, m_mesh{other.m_mesh} {}

   private:
    cell_handle         m_index;
    simplex_mesh const* m_mesh;

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

    auto equal(cell_iterator const& other) const {
      return m_index == other.m_index;
    }
    auto dereference() const { return m_index; }
  };
  //----------------------------------------------------------------------------
  struct cell_container {
    using iterator       = cell_iterator;
    using const_iterator = cell_iterator;
    //--------------------------------------------------------------------------
    simplex_mesh const* m_mesh;
    //--------------------------------------------------------------------------
    auto begin() const {
      cell_iterator vi{cell_handle{0}, m_mesh};
      if (!m_mesh->is_valid(*vi)) {
        ++vi;
      }
      return vi;
    }
    //--------------------------------------------------------------------------
    auto end() const { return cell_iterator{cell_handle{size()}, m_mesh}; }
    //--------------------------------------------------------------------------
    auto size() const {
      return m_mesh->m_cell_indices.size() / num_vertices_per_simplex();
    }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  using cell_property_t = vector_property_impl<cell_handle, T>;
  using cell_property_container_t =
      std::map<std::string, std::unique_ptr<vector_property<cell_handle>>>;
  //============================================================================
 private:
  std::vector<vertex_handle>           m_cell_indices;
  std::vector<cell_handle>             m_invalid_cells;
  cell_property_container_t            m_cell_properties;
  mutable std::unique_ptr<hierarchy_t> m_hierarchy;

 public:
  //============================================================================
  constexpr simplex_mesh() = default;
  //============================================================================
 public:
  simplex_mesh(simplex_mesh const& other)
      : parent_t{other}, m_cell_indices{other.m_cell_indices} {
    for (auto const& [key, prop] : other.m_cell_properties) {
      m_cell_properties.insert(std::pair{key, prop->clone()});
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  simplex_mesh(simplex_mesh&& other) noexcept = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(simplex_mesh const& other) -> simplex_mesh& {
    parent_t::operator=(other);
    m_cell_properties.clear();
    m_cell_indices = other.m_cell_indices;
    for (auto const& [key, prop] : other.m_cell_properties) {
      m_cell_properties.insert(std::pair{key, prop->clone()});
    }
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(simplex_mesh&& other) noexcept -> simplex_mesh& = default;
  //----------------------------------------------------------------------------
  // simplex_mesh(std::string const& file) { read(file); }
  //----------------------------------------------------------------------------
  simplex_mesh(std::initializer_list<pos_t>&& vertices)
      : parent_t{std::move(vertices)} {}
  simplex_mesh(std::vector<vec<Real, NumDimensions>> const& positions)
      : parent_t{positions} {}
  simplex_mesh(std::vector<vec<Real, NumDimensions>>&& positions)
      : parent_t{std::move(positions)} {}
  //----------------------------------------------------------------------------
 private:
  // template <typename T, typename Prop, typename Grid>
  // auto copy_prop(std::string const& name, Prop const& prop, Grid const& g) {
  //  if (prop->type() == typeid(T)) {
  //    auto const& grid_prop = g.template vertex_property<T>(name);
  //    auto&       tri_prop  = this->template add_vertex_property<T>(name);
  //    g.iterate_over_vertex_indices([&](auto const... is) {
  //      std::array is_arr{is...};
  //      tri_prop[vertex_handle{is_arr[0] + is_arr[1] * g.size(0)}] =
  //          grid_prop(is...);
  //    });
  //  }
  //}

 public:
#ifdef __cpp_concepts
  template <indexable_space DimX, indexable_space DimY, indexable_space DimZ>
  requires(NumDimensions == 3)
#else
  template <typename DimX, typename DimY, typename DimZ,
            size_t N_ = NumDimensions, enable_if<N_ == 3> = true>
#endif
      simplex_mesh(grid<DimX, DimY, DimZ> const& g) {

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
          auto const ri_bo_fr = vertex_handle{(i + 1) + j * g.size(0) +
                                              k * g.size(0) * g.size(1)};
          auto const le_to_fr = vertex_handle{i + (j + 1) * g.size(0) +
                                              k * g.size(0) * g.size(1)};
          auto const ri_to_fr = vertex_handle{(i + 1) + (j + 1) * g.size(0) +
                                              k * g.size(0) * g.size(1)};
          auto const le_bo_ba = vertex_handle{i + j * g.size(0) +
                                              (k + 1) * g.size(0) * g.size(1)};
          auto const ri_bo_ba = vertex_handle{(i + 1) + j * g.size(0) +
                                              (k + 1) * g.size(0) * g.size(1)};
          auto const le_to_ba = vertex_handle{i + (j + 1) * g.size(0) +
                                              (k + 1) * g.size(0) * g.size(1)};
          auto const ri_to_ba = vertex_handle{(i + 1) + (j + 1) * g.size(0) +
                                              (k + 1) * g.size(0) * g.size(1)};
          if (turned(i, j, k)) {
            insert_cell(le_bo_fr, ri_bo_ba, ri_to_fr,
                        le_to_ba);  // inner
            insert_cell(le_bo_fr, ri_bo_fr, ri_to_fr,
                        ri_bo_ba);  // right front
            insert_cell(le_bo_fr, ri_to_fr, le_to_fr,
                        le_to_ba);  // left front
            insert_cell(ri_to_fr, ri_bo_ba, ri_to_ba,
                        le_to_ba);  // right back
            insert_cell(le_bo_fr, le_bo_ba, ri_bo_ba,
                        le_to_ba);  // left back
          } else {
            insert_cell(le_to_fr, ri_bo_fr, le_bo_ba,
                        ri_to_ba);  // inner
            insert_cell(le_bo_fr, ri_bo_fr, le_to_fr,
                        le_bo_ba);  // left front
            insert_cell(ri_bo_fr, ri_to_fr, le_to_fr,
                        ri_to_ba);  // right front
            insert_cell(le_to_fr, le_to_ba, ri_to_ba,
                        le_bo_ba);  // left back
            insert_cell(ri_bo_fr, ri_bo_ba, ri_to_ba,
                        le_bo_ba);  // right back
          }
        }
      }
    }
    // for (auto const& [name, prop] : g.vertex_properties()) {
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
  auto operator[](cell_handle t) const -> auto { return cell_at(t.i); }
  auto operator[](cell_handle t) -> auto { return cell_at(t.i); }
  //----------------------------------------------------------------------------
  auto at(cell_handle t) const -> auto { return cell_at(t.i); }
  auto at(cell_handle t) -> auto { return cell_at(t.i); }
  //----------------------------------------------------------------------------
  auto cell_at(cell_handle t) const -> auto { return cell_at(t.i); }
  auto cell_at(cell_handle t) -> auto { return cell_at(t.i); }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  auto cell_at(size_t const i) const {
    return cell_at(i, std::make_index_sequence<num_vertices_per_simplex()>{});
  }
  template <size_t... Seq>
  auto cell_at(size_t const i) {
    return cell_at(i, std::make_index_sequence<num_vertices_per_simplex()>{});
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Seq>
  auto cell_at(size_t const i, std::index_sequence<Seq...> /*seq*/) const
      -> const_cell_at_return_type {
    return {m_cell_indices[i * num_vertices_per_simplex() + Seq]...};
  }
  template <size_t... Seq>
  auto cell_at(size_t const i, std::index_sequence<Seq...> /*seq*/)
      -> cell_at_return_type {
    return {m_cell_indices[i * num_vertices_per_simplex() + Seq]...};
  }
  //----------------------------------------------------------------------------
 public:
#ifdef __cpp_concepts
  template <arithmetic... Ts>
  requires(sizeof...(Ts) == NumDimensions)
#else
  template <typename... Ts, enable_if<is_arithmetic<Ts...>> = true,
            enable_if<sizeof...(Ts) == NumDimensions> = true>
#endif
      auto insert_vertex(Ts const... ts) {
    auto const vi = parent_t::insert_vertex(ts...);
    // if (m_hierarchy != nullptr) {
    //  if (!m_hierarchy->insert_vertex(vi.i)) {
    //    build_hierarchy();
    //  }
    //}
    return vi;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t const& v) {
    auto const vi = parent_t::insert_vertex(v);
    // if (m_hierarchy != nullptr) {
    //  if (!m_hierarchy->insert_vertex(vi.i)) {
    //    build_hierarchy();
    //  }
    //}
    return vi;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& v) {
    auto const vi = parent_t::insert_vertex(std::move(v));
    // if (m_hierarchy != nullptr) {
    //  if (!m_hierarchy->insert_vertex(vi.i)) {
    //    build_hierarchy();
    //  }
    //}
    return vi;
  }
  //----------------------------------------------------------------------------
  template <typename... Handles>
  auto insert_cell(Handles const... handles) {
    static_assert(sizeof...(Handles) == num_vertices_per_simplex(),
                  "wrong number of vertices for simplex");
    (m_cell_indices.push_back(handles), ...);
    for (auto& [key, prop] : m_cell_properties) {
      prop->push_back();
    }
    return cell_handle{cells().size() - 1};
  }
  //----------------------------------------------------------------------------
  auto clear() {
    parent_t::clear();
    m_cell_indices.clear();
  }
  //----------------------------------------------------------------------------
  auto cells() const { return cell_container{this}; }
  //----------------------------------------------------------------------------
#ifdef TATOOINE_HAS_CGAL_SUPPORT
#ifndef __cpp_concepts
  template <size_t N_ = NumDimensions, enable_if<N_ == 2 || N_ == 3> = true>
#endif
      auto build_delaunay_mesh() -> void
#ifdef __cpp_concepts
      requires(NumDimensions == 2) ||
      (NumDimensions == 3)
#endif
  {
    build_delaunay_mesh(std::make_index_sequence<NumDimensions>{});
  }

 private:
#ifdef __cpp_concepts
  template <size_t... Seq>
      requires(NumDimensions == 2) ||
      (NumDimensions == 3)
#else
  template <size_t... Seq, size_t N_ = NumDimensions,
            enable_if<N_ == 2 || N_ == 3> = true>
#endif
          auto build_delaunay_mesh(std::index_sequence<Seq...> /*seq*/)
              -> void {
    m_cell_indices.clear();
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vb     = std::conditional_t<
        NumDimensions == 2,
        CGAL::Triangulation_vertex_base_with_info_2<vertex_handle, Kernel>,
        CGAL::Triangulation_vertex_base_with_info_3<vertex_handle, Kernel>>;
    using Tds = std::conditional_t<NumDimensions == 2,
                                   CGAL::Triangulation_data_structure_2<Vb>,
                                   CGAL::Triangulation_data_structure_3<Vb>>;
    using Triangulation =
        std::conditional_t<NumDimensions == 2,
                           CGAL::Delaunay_triangulation_2<Kernel, Tds>,
                           CGAL::Delaunay_triangulation_3<Kernel, Tds>>;
    using Point =
        std::conditional_t<NumDimensions == 2, typename Kernel::Point_2,
                           typename Kernel::Point_3>;
    std::vector<std::pair<Point, vertex_handle>> points;
    points.reserve(vertices().size());
    for (auto v : vertices()) {
      points.emplace_back(Point{at(v)(Seq)...}, v);
    }

    Triangulation dt{begin(points), end(points)};
    if constexpr (NumDimensions == 2) {
      for (auto it = dt.finite_faces_begin(); it != dt.finite_faces_end();
           ++it) {
        insert_cell(vertex_handle{it->vertex(0)->info()},
                    vertex_handle{it->vertex(Seq + 1)->info()}...);
      }
    } else if constexpr (NumDimensions == 3) {
      for (auto it = dt.finite_cells_begin(); it != dt.finite_cells_end();
           ++it) {
        insert_cell(vertex_handle{it->vertex(0)->info()},
                    vertex_handle{it->vertex(Seq + 1)->info()}...);
      }
    }
  }
#endif
  //----------------------------------------------------------------------------
 public:
  auto write_vtk(std::string const& path,
                 std::string const& title = "tatooine mesh") const {
    if constexpr (SimplexDim == 2) {
      write_triangular_mesh_vtk(path, title);
    } else if constexpr (SimplexDim == 3) {
      write_tetrahedral_mesh_vtk(path, title);
    }
  }

 private:
  template <size_t _N = SimplexDim, enable_if<_N == 2> = true>
  auto write_triangular_mesh_vtk(std::string const& path,
                                 std::string const& title) const -> bool {
    using boost::copy;
    using boost::adaptors::transformed;
    vtk::legacy_file_writer writer(path, vtk::dataset_type::unstructured_grid);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      if constexpr (NumDimensions == 2) {
        auto three_dims = [](vec<Real, 2> const& v2) {
          return vec<Real, 3>{v2(0), v2(1), 0};
        };
        std::vector<vec<Real, 3>> v3s(vertices().size());
        auto                      three_dimensional = transformed(three_dims);
        copy(vertex_data() | three_dimensional, begin(v3s));
        writer.write_points(v3s);

      } else if constexpr (NumDimensions == 3) {
        writer.write_points(vertex_data());
      }

      std::vector<std::vector<size_t>> vertices_per_cell;
      vertices_per_cell.reserve(cells().size());
      std::vector<vtk::cell_type> cell_types(cells().size(),
                                             vtk::cell_type::triangle);
      for (auto const c : cells()) {
        auto const [v0, v1, v2] = at(c);
        vertices_per_cell.push_back(std::vector{v0.i, v1.i, v2.i});
      }
      writer.write_cells(vertices_per_cell);
      writer.write_cell_types(cell_types);

      // write vertex_handle data
      writer.write_point_data(vertices().size());
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
  template <size_t _N = SimplexDim, enable_if<_N == 3> = true>
  auto write_tetrahedral_mesh_vtk(std::string const& path,
                                  std::string const& title) const -> bool {
    using boost::copy;
    using boost::adaptors::transformed;
    vtk::legacy_file_writer writer(path, vtk::dataset_type::unstructured_grid);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      writer.write_points(vertex_data());

      std::vector<std::vector<size_t>> vertices_per_cell;
      vertices_per_cell.reserve(cells().size());
      std::vector<vtk::cell_type> cell_types(cells().size(),
                                             vtk::cell_type::tetra);
      for (auto const t : cells()) {
        auto const [v0, v1, v2, v3] = at(t);
        vertices_per_cell.push_back(std::vector{v0.i, v1.i, v2.i, v3.i});
      }
      writer.write_cells(vertices_per_cell);
      writer.write_cell_types(cell_types);

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
 public:
  auto read(std::string const& path) {
    auto ext = path.substr(path.find_last_of(".") + 1);
    if constexpr (NumDimensions == 3) {
      if (ext == "vtk") {
        read_vtk(path);
      }
    }
  }
  //----------------------------------------------------------------------------
  template <size_t _N = NumDimensions, enable_if<_N == 3> = true>
  auto read_vtk(std::string const& path) {
    struct listener_t : vtk::legacy_file_listener {
      simplex_mesh& mesh;

      listener_t(simplex_mesh& _mesh) : mesh(_mesh) {}

      void on_dataset_type(vtk::dataset_type t) override {
        if (t != vtk::dataset_type::polydata) {
          throw std::runtime_error{
              "[simplex_mesh] need polydata when reading vtk legacy"};
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
            mesh.insert_cell(size_t(ps[i]), size_t(ps[i + 1]),
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
  auto cell_property(std::string const& name) -> auto& {
    auto prop        = m_cell_properties.at(name).get();
    auto casted_prop = dynamic_cast<cell_property_t<T>*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto cell_property(std::string const& name) const -> auto const& {
    auto prop        = m_cell_properties.at(name).get();
    auto casted_prop = dynamic_cast<cell_property_t<T> const*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto add_cell_property(std::string const& name, T const& value = T{})
      -> auto& {
    auto [it, suc] = m_cell_properties.insert(
        std::pair{name, std::make_unique<cell_property_t<T>>(value)});
    auto prop = dynamic_cast<cell_property_t<T>*>(it->second.get());
    prop->resize(m_cell_indices.size());
    return *prop;
  }
  //----------------------------------------------------------------------------
  constexpr bool is_valid(cell_handle t) const {
    return boost::find(m_invalid_cells, t) == end(m_invalid_cells);
  }
  //----------------------------------------------------------------------------
  auto build_hierarchy() const {
    clear_hierarchy();
    auto& h = hierarchy();
    if constexpr (is_quadtree<hierarchy_t>() || is_octree<hierarchy_t>()) {
      for (auto v : vertices()) {
        h.insert_vertex(v.i);
      }
      for (auto c : cells()) {
        h.insert_triangle(c.i);
      }
    }
  }
  //----------------------------------------------------------------------------
  auto clear_hierarchy() const { m_hierarchy.reset(); }
  //----------------------------------------------------------------------------
  auto hierarchy() const -> auto& {
    if (m_hierarchy == nullptr) {
      auto const bb = bounding_box();
      m_hierarchy   = std::make_unique<hierarchy_t>(*this, bb.min(), bb.max());
    }
    return *m_hierarchy;
  }
  //----------------------------------------------------------------------------
  // template <typename T>
  // auto sampler(vertex_property_t<T> const& prop) const {
  //  if (m_hierarchy == nullptr) {
  //    build_hierarchy();
  //  }
  //  return vertex_property_sampler_t<T>{*this, prop};
  //}
  ////--------------------------------------------------------------------------
  // template <typename T>
  // auto vertex_property_sampler(std::string const& name) const {
  //  return sampler<T>(this->template vertex_property<T>(name));
  //}
  constexpr auto bounding_box() const {
    auto bb = axis_aligned_bounding_box<Real, num_dimensions()>{};
    for (auto const v : vertices()) {
      bb += at(v);
    }
    return bb;
  }
};
//==============================================================================
simplex_mesh()->simplex_mesh<double, 3>;
simplex_mesh(std::string const&)->simplex_mesh<double, 3>;
template <typename... Dims>
simplex_mesh(grid<Dims...> const& g)
    -> simplex_mesh<typename grid<Dims...>::real_t, sizeof...(Dims)>;
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
//    std::vector<std::vector<size_t>> cells; points.reserve(num_pts);
//    cells.reserve(meshes.size());
//
//    for (auto const& m : meshes) {
//      // add points
//      for (auto const& v : m.vertices()) {
//        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
//      }
//
//      // add cells
//      for (auto t : m.cells()) {
//        cells.emplace_back();
//        cells.back().push_back(cur_first + m[t][0].i);
//        cells.back().push_back(cur_first + m[t][1].i);
//        cells.back().push_back(cur_first + m[t][2].i);
//      }
//      cur_first += m.num_vertices();
//    }
//
//    // write
//    writer.set_title(title);
//    writer.write_header();
//    writer.write_points(points);
//    writer.write_polygons(cells);
//    //writer.write_point_data(num_pts);
//    writer.close();
//  }
//}
//}  // namespace detail
////==============================================================================
// template <typename Real>
// auto write_vtk(std::vector<simplex_mesh<Real, 3>> const& meshes,
// std::string const& path,
//               std::string const& title = "tatooine meshes") {
//  detail::write_mesh_container_to_vtk(meshes, path, title);
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
