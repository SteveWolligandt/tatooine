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
#include <tatooine/pointset.h>
#include <tatooine/property.h>
#include <tatooine/uniform_tree_hierarchy.h>
#include <tatooine/vtk_legacy.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <filesystem>
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
struct simplex_mesh_cell_at_return_type_impl<
    VertexHandle, NumVerticesPerSimplex, NumVerticesPerSimplex, Ts...> {
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
  using type = uniform_tree_hierarchy<Mesh>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real>
struct simplex_mesh_hierarchy<Mesh, Real, 2, 2> {
  using type = uniform_tree_hierarchy<Mesh>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real>
struct simplex_mesh_hierarchy<Mesh, Real, 3, 2> {
  using type = uniform_tree_hierarchy<Mesh>;
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
struct simplex_mesh_parent<Mesh, Real, 3, 2> : pointset<Real, 3>,
                                               ray_intersectable<Real, 3> {
  using real_t = Real;
  using typename pointset<real_t, 3>::vertex_handle;
  using hierarchy_t = simplex_mesh_hierarchy_t<Mesh, real_t, 3, 2>;
  using const_cell_at_return_type =
      simplex_mesh_cell_at_return_type<vertex_handle const&, 3>;
  using cell_at_return_type =
      simplex_mesh_cell_at_return_type<vertex_handle&, 3>;

  using typename ray_intersectable<real_t, 3>::ray_t;
  using typename ray_intersectable<real_t, 3>::intersection_t;
  using typename ray_intersectable<real_t, 3>::optional_intersection_t;
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
      auto const& v0             = mesh.at(vi0);
      auto const& v1             = mesh.at(vi1);
      auto const& v2             = mesh.at(vi2);
      auto const  v0v1           = v1 - v0;
      auto const  v0v2           = v2 - v0;
      auto const  pvec           = cross(r.direction(), v0v2);
      auto const  det            = dot(v0v1, pvec);
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
        auto const pos = barycentric_coord(0) * v0 + barycentric_coord(1) * v1 +
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
  template <typename T>
  struct vertex_property_sampler_t : field<vertex_property_sampler_t<T>, Real,
                                           parent_t::num_dimensions(), T> {
   private:
    using mesh_t = simplex_mesh<Real, NumDimensions, SimplexDim>;
    using this_t = vertex_property_sampler_t<T>;

    mesh_t const&               m_mesh;
    vertex_property_t<T> const& m_prop;
    //--------------------------------------------------------------------------
   public:
    vertex_property_sampler_t(mesh_t const&               mesh,
                              vertex_property_t<T> const& prop)
        : m_mesh{mesh}, m_prop{prop} {}
    //--------------------------------------------------------------------------
    auto mesh() const -> auto const& { return m_mesh; }
    auto property() const -> auto const& { return m_prop; }
    //--------------------------------------------------------------------------
    [[nodiscard]] auto evaluate(pos_t const& x, real_t const /*t*/) const -> T {
      return evaluate(x,
                      std::make_index_sequence<num_vertices_per_simplex()>{});
    }
    //--------------------------------------------------------------------------
    template <size_t... VertexSeq>
    [[nodiscard]] auto evaluate(pos_t const& x,
                                std::index_sequence<VertexSeq...> /*seq*/) const
        -> T {
      auto cell_handles = m_mesh.hierarchy().nearby_cells(x);
      if (cell_handles.empty()) {
        throw std::runtime_error{
            "[vertex_property_sampler_t::sample] out of domain"};
      }
      for (auto t : cell_handles) {
        auto const            vs = m_mesh.cell_at(t);
        static constexpr auto NV = num_vertices_per_simplex();
        auto                  A  = mat<Real, NV, NV>::ones();
        auto                  b  = vec<Real, NV>::ones();
        for (size_t r = 0; r < num_dimensions(); ++r) {
          (
              [&]() {
                if (VertexSeq > 0) {
                  A(r, VertexSeq) = m_mesh[std::get<VertexSeq>(vs)](r) -
                                    m_mesh[std::get<0>(vs)](r);
                } else {
                  ((A(r, VertexSeq) = 0), ...);
                }
              }(),
              ...);

          b(r) = x(r) - m_mesh[std::get<0>(vs)](r);
        }
        auto const   barycentric_coord = solve(A, b);
        real_t const eps               = 1e-8;
        if (((barycentric_coord(VertexSeq) >= -eps) && ...) &&
            ((barycentric_coord(VertexSeq) <= 1 + eps) && ...)) {
          return (
              (m_prop[std::get<VertexSeq>(vs)] * barycentric_coord(VertexSeq)) +
              ...);
        }
      }
      throw std::runtime_error{
          "[vertex_property_sampler_t::sample] out of domain"};
      return T{};
    }
  };
  //----------------------------------------------------------------------------
  struct cell_handle : handle<cell_handle> {
    using handle<cell_handle>::handle;
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
  simplex_mesh(std::filesystem::path const& path) { read(path); }
  //----------------------------------------------------------------------------
  simplex_mesh(std::initializer_list<pos_t>&& vertices)
      : parent_t{std::move(vertices)} {}
  simplex_mesh(std::vector<vec<Real, NumDimensions>> const& positions)
      : parent_t{positions} {}
  simplex_mesh(std::vector<vec<Real, NumDimensions>>&& positions)
      : parent_t{std::move(positions)} {}
  //----------------------------------------------------------------------------
 private:
  template <typename... TypesToCheck, typename Prop, typename Grid>
  auto copy_prop(std::string const& name, Prop const& prop, Grid const& g) {
    (([&]() {
       if (prop->type() == typeid(TypesToCheck)) {
         auto const& grid_prop = g.template vertex_property<TypesToCheck>(name);
         auto& mesh_prop = this->template vertex_property<TypesToCheck>(name);
         auto  vi        = vertex_handle{0};
         g.vertices().iterate_indices(
             [&](auto const... is) { mesh_prop[vi++] = grid_prop(is...); });
       }
     }()),
     ...);
  }

 public:
#ifdef __cpp_concepts
  template <indexable_space DimX, indexable_space DimY>
  requires(NumDimensions == 2) && (SimplexDim == 2)
#else
  template <
      typename DimX, typename DimY, size_t NumDimensions_ = NumDimensions,
      size_t SimplexDim_                               = SimplexDim,
      enable_if<NumDimensions == NumDimensions_, SimplexDim == SimplexDim_,
                NumDimensions_ == 2, SimplexDim_ == 2> = true>
#endif
                                      simplex_mesh(grid<DimX, DimY> const& g) {
    auto const gv = g.vertices();
    for (auto v : gv) {
      insert_vertex(gv[v]);
    }
    auto const gc = g.cells();
    auto const s0 = g.size(0);
    gc.iterate_indices([&](auto const i, auto const j) {
      auto const le_bo = vertex_handle{i + j * s0};
      auto const ri_bo = vertex_handle{(i + 1) + j * s0};
      auto const le_to = vertex_handle{i + (j + 1) * s0};
      auto const ri_to = vertex_handle{(i + 1) + (j + 1) * s0};
      insert_cell(le_bo, ri_bo, le_to);
      insert_cell(ri_bo, ri_to, le_to);
    });
    for (auto const& [name, prop] : g.vertex_properties()) {
      copy_prop<mat4d, mat3d, mat2d, mat4f, mat3f, mat2f, vec4d, vec3d, vec2d,
                vec4f, vec3f, vec2f, double, float, std::int8_t, std::uint8_t,
                std::int16_t, std::uint16_t, std::int32_t, std::uint32_t,
                std::int64_t, std::uint64_t>(name, prop, g);
    }
  }

#ifdef __cpp_concepts
  template <indexable_space DimX, indexable_space DimY, indexable_space DimZ>
  requires(NumDimensions == 3) &&
      (SimplexDim == 3)
#else
  template <
      typename DimX, typename DimY, typename DimZ,
      size_t NumDimensions_ = NumDimensions, size_t SimplexDim_ = SimplexDim,
      enable_if<NumDimensions == NumDimensions_, SimplexDim == SimplexDim_,
                NumDimensions_ == 3, SimplexDim_ == 3> = true>
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

    auto const gv = g.vertices();
    for (auto v : gv) {
      insert_vertex(gv[v]);
    }
    auto const gc   = g.cells();
    auto const s0   = g.size(0);
    auto const s1   = g.size(1);
    auto const s0s1 = s0 * s1;
    gc.iterate_vertices([&](auto const i, auto const j, auto const k) {
      for (size_t i = 0; i < s0 - 1; ++i) {
        auto const le_bo_fr = vertex_handle{i + j * s0 + k * s0s1};
        auto const ri_bo_fr = vertex_handle{(i + 1) + j * s0 + k * s0s1};
        auto const le_to_fr = vertex_handle{i + (j + 1) * s0 + k * s0s1};
        auto const ri_to_fr = vertex_handle{(i + 1) + (j + 1) * s0 + k * s0s1};
        auto const le_bo_ba = vertex_handle{i + j * s0 + (k + 1) * s0s1};
        auto const ri_bo_ba = vertex_handle{(i + 1) + j * s0 + (k + 1) * s0s1};
        auto const le_to_ba = vertex_handle{i + (j + 1) * s0 + (k + 1) * s0s1};
        auto const ri_to_ba =
            vertex_handle{(i + 1) + (j + 1) * s0 + (k + 1) * s0s1};
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
    });
    for (auto const& [name, prop] : g.vertex_properties()) {
      copy_prop<mat4d, mat3d, mat2d, mat4f, mat3f, mat2f, vec4d, vec3d, vec2d,
                vec4f, vec3f, vec2f, double, float, std::int8_t, std::uint8_t,
                std::int16_t, std::uint16_t, std::int32_t, std::uint32_t,
                std::int64_t, std::uint64_t>(name, prop, g);
    }
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
  template <size_t NumDimensions_ = NumDimensions,
            enable_if<NumDimensions_ == 2 || NumDimensions_ == 3> = true>
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
  template <size_t... Seq, size_t NumDimensions_ = NumDimensions,
            enable_if<NumDimensions_ == 2 || NumDimensions_ == 3> = true>
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
  template <typename T>
  auto cell_property(std::string const& name) -> auto& {
    if (auto it = m_cell_properties.find(name); it == end(m_cell_properties)) {
      return insert_cell_property<T>(name);
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<cell_property_t<T>*>(
          m_cell_properties.at(name).get());
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto cell_property(std::string const& name) const -> const auto& {
    if (auto it = m_cell_properties.find(name); it == end(m_cell_properties)) {
      throw std::runtime_error{"property \"" + name + "\" not found"};
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<cell_property_t<T>*>(
          m_cell_properties.at(name).get());
    }
  }
  //----------------------------------------------------------------------------
  auto scalar_cell_property(std::string const& name) const -> auto const& {
    return cell_property<tatooine::real_t>(name);
  }
  //----------------------------------------------------------------------------
  auto scalar_cell_property(std::string const& name) -> auto& {
    return cell_property<tatooine::real_t>(name);
  }
  //----------------------------------------------------------------------------
  auto vec2_cell_property(std::string const& name) const -> auto const& {
    return cell_property<vec2>(name);
  }
  //----------------------------------------------------------------------------
  auto vec2_cell_property(std::string const& name) -> auto& {
    return cell_property<vec2>(name);
  }
  //----------------------------------------------------------------------------
  auto vec3_cell_property(std::string const& name) const -> auto const& {
    return cell_property<vec3>(name);
  }
  //----------------------------------------------------------------------------
  auto vec3_cell_property(std::string const& name) -> auto& {
    return cell_property<vec3>(name);
  }
  //----------------------------------------------------------------------------
  auto vec4_cell_property(std::string const& name) const -> auto const& {
    return cell_property<vec4>(name);
  }
  //----------------------------------------------------------------------------
  auto vec4_cell_property(std::string const& name) -> auto& {
    return cell_property<vec4>(name);
  }
  //----------------------------------------------------------------------------
  auto mat2_cell_property(std::string const& name) const -> auto const& {
    return cell_property<mat2>(name);
  }
  //----------------------------------------------------------------------------
  auto mat2_cell_property(std::string const& name) -> auto& {
    return cell_property<mat2>(name);
  }
  //----------------------------------------------------------------------------
  auto mat3_cell_property(std::string const& name) const -> auto const& {
    return cell_property<mat3>(name);
  }
  //----------------------------------------------------------------------------
  auto mat3_cell_property(std::string const& name) -> auto& {
    return cell_property<mat3>(name);
  }
  //----------------------------------------------------------------------------
  auto mat4_cell_property(std::string const& name) const -> auto const& {
    return cell_property<mat4>(name);
  }
  //----------------------------------------------------------------------------
  auto mat4_cell_property(std::string const& name) -> auto& {
    return cell_property<mat4>(name);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto insert_cell_property(std::string const& name, T const& value = T{})
      -> auto& {
    auto [it, suc] = m_cell_properties.insert(
        std::pair{name, std::make_unique<cell_property_t<T>>(value)});
    auto prop = dynamic_cast<cell_property_t<T>*>(it->second.get());
    prop->resize(cells().size());
    return *prop;
  }
  auto write_vtk(std::filesystem::path const& path,
                 std::string const&           title = "tatooine mesh") const {
    if constexpr (SimplexDim == 2) {
      write_triangular_mesh_vtk(path, title);
    } else if constexpr (SimplexDim == 3) {
      write_tetrahedral_mesh_vtk(path, title);
    }
  }

 private:
  template <size_t SimplexDim_ = SimplexDim, enable_if<SimplexDim_ == 2> = true>
  auto write_triangular_mesh_vtk(std::filesystem::path const& path,
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
  template <size_t SimplexDim_ = SimplexDim, enable_if<SimplexDim_ == 3> = true>
  auto write_tetrahedral_mesh_vtk(std::filesystem::path const& path,
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
  auto read(std::filesystem::path const& path) {
    auto ext = path.extension();
    if constexpr (NumDimensions == 2 || NumDimensions == 3) {
      if (ext == ".vtk") {
        read_vtk(path);
      }
    }
  }
  //----------------------------------------------------------------------------
  template <size_t NumDimensions_ = NumDimensions,
            enable_if<NumDimensions_ == 2 || NumDimensions_ == 3> = true>
  auto read_vtk(std::filesystem::path const& path) {
    struct listener_t : vtk::legacy_file_listener {
      simplex_mesh&          mesh;
      std::vector<int>       cells;

      listener_t(simplex_mesh& _mesh) : mesh(_mesh) {}
      auto add_cells(std::vector<int> const& cells) -> void {
        size_t i = 0;
        while (i < size(cells)) {
          auto const num_vertices = cells[i++];
          if (num_vertices != num_vertices_per_simplex()) {
            throw std::runtime_error{
                "Number of vertices in file does not match number of vertices "
                "per simplex."};
          }
          for (size_t j = 0; j < static_cast<size_t>(num_vertices); ++j) {
            mesh.m_cell_indices.push_back(vertex_handle{cells[i++]});
          }
          for (auto& [key, prop] : mesh.m_cell_properties) {
            prop->push_back();
          }
        }
      }
      auto on_cells(std::vector<int> const& cells) -> void override {
        add_cells(cells);
      }
      auto on_dataset_type(vtk::dataset_type t) -> void override {
        if (t != vtk::dataset_type::unstructured_grid &&
            t != vtk::dataset_type::polydata) {
          throw std::runtime_error{
              "[simplex_mesh] need polydata or unstructured_grid when reading vtk legacy"};
        }
      }

      auto on_points(std::vector<std::array<float, 3>> const& ps)
          -> void override {
        for (auto& p : ps) {
          if constexpr (NumDimensions == 2) {
            mesh.insert_vertex(static_cast<Real>(p[0]),
                               static_cast<Real>(p[1]));
          }
          if constexpr (NumDimensions == 3) {
            mesh.insert_vertex(static_cast<Real>(p[0]), static_cast<Real>(p[1]),
                               static_cast<Real>(p[2]));
          }
        }
      }
      auto on_points(std::vector<std::array<double, 3>> const& ps)
          -> void override {
        for (auto& p : ps) {
          if constexpr (NumDimensions == 2) {
            mesh.insert_vertex(static_cast<Real>(p[0]),
                               static_cast<Real>(p[1]));
          }
          if constexpr (NumDimensions == 3) {
            mesh.insert_vertex(static_cast<Real>(p[0]), static_cast<Real>(p[1]),
                               static_cast<Real>(p[2]));
          }
        }
      }
      auto on_polygons(std::vector<int> const& ps) -> void override {
        add_cells(ps);
      }
      auto on_scalars(std::string const& data_name,
                      std::string const& /*lookup_table_name*/,
                      size_t num_comps, std::vector<double> const& scalars,
                      vtk::reader_data data) -> void override {
        if (data == vtk::reader_data::point_data) {
          if (num_comps == 1) {
            auto& prop =
                mesh.template insert_vertex_property<double>(data_name);
            for (auto v = vertex_handle{0}; v < vertex_handle{prop.size()}; ++v) {
              prop[v] = scalars[v.i];
            }
          } else if (num_comps == 2) {
            auto& prop =
                mesh.template insert_vertex_property<vec<double, 2>>(data_name);

            for (auto v = vertex_handle{0}; v < vertex_handle{prop.size()}; ++v) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[v][j] = scalars[v.i * num_comps + j];
              }
            }
          } else if (num_comps == 3) {
            auto& prop =
                mesh.template insert_vertex_property<vec<double, 3>>(data_name);
            for (auto v = vertex_handle{0}; v < vertex_handle{prop.size()}; ++v) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[v][j] = scalars[v.i * num_comps + j];
              }
            }
          } else if (num_comps == 4) {
            auto& prop =
                mesh.template insert_vertex_property<vec<double, 4>>(data_name);
            for (auto v = vertex_handle{0}; v < vertex_handle{prop.size()}; ++v) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[v][j] = scalars[v.i * num_comps + j];
              }
            }
          }
        }
        else if (data == vtk::reader_data::cell_data) {
          if (num_comps == 1) {
            auto& prop =
                mesh.template insert_cell_property<double>(data_name);
            for (auto c = cell_handle{0}; c < cell_handle{prop.size()}; ++c) {
              prop[c] = scalars[c.i];
            }
          } else if (num_comps == 2) {
            auto& prop =
                mesh.template insert_cell_property<vec<double, 2>>(data_name);

            for (auto c = cell_handle{0}; c < cell_handle{prop.size()}; ++c) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[c][j] = scalars[c.i * num_comps + j];
              }
            }
          } else if (num_comps == 3) {
            auto& prop =
                mesh.template insert_cell_property<vec<double, 3>>(data_name);
            for (auto c = cell_handle{0}; c < cell_handle{prop.size()}; ++c) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[c][j] = scalars[c.i * num_comps + j];
              }
            }
          } else if (num_comps == 4) {
            auto& prop =
                mesh.template insert_cell_property<vec<double, 4>>(data_name);
            for (auto c = cell_handle{0}; c < cell_handle{prop.size()}; ++c) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[c][j] = scalars[c.i * num_comps + j];
              }
            }
          }
        }
      }
    } listener{*this};
    auto f = vtk::legacy_file{path};
    f.add_listener(listener);
    f.read();
  }
  //----------------------------------------------------------------------------
  constexpr bool is_valid(cell_handle t) const {
    return boost::find(m_invalid_cells, t) == end(m_invalid_cells);
  }
  //----------------------------------------------------------------------------
  auto build_hierarchy() const {
    clear_hierarchy();
    auto& h = hierarchy();
    if constexpr (is_uniform_tree_hierarchy<hierarchy_t>()) {
      for (auto v : vertices()) {
        h.insert_vertex(v);
      }
      for (auto c : cells()) {
        h.insert_cell(c);
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
  template <typename T>
  auto sampler(vertex_property_t<T> const& prop) const {
    if (m_hierarchy == nullptr) {
      build_hierarchy();
    }
    return vertex_property_sampler_t<T>{*this, prop};
  }
  //--------------------------------------------------------------------------
  template <typename T>
  auto vertex_property_sampler(std::string const& name) const {
    return sampler<T>(this->template vertex_property<T>(name));
  }
  //--------------------------------------------------------------------------
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
//    std::vector<std::array<typename MeshCont::value_type::real_t, 3>>
//    points; std::vector<std::vector<size_t>> cells; points.reserve(num_pts);
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
