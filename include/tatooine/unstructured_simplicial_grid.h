#ifndef TATOOINE_UNSTRUCTURED_SIMPLICIAL_GRID_H
#define TATOOINE_UNSTRUCTURED_SIMPLICIAL_GRID_H
//==============================================================================
#ifdef TATOOINE_CDT_AVAILABLE
#include <CDT.h>
#endif
#ifdef TATOOINE_HAS_CGAL_SUPPORT
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#endif

#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/celltree.h>
#include <tatooine/kdtree.h>
#include <tatooine/packages.h>
#include <tatooine/pointset.h>
#include <tatooine/property.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/uniform_tree_hierarchy.h>
#include <tatooine/vtk/xml/data_array.h>
#include <tatooine/vtk_legacy.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <filesystem>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail::unstructured_simplicial_grid {
//==============================================================================
template <typename VertexHandle, std::size_t NumVerticesPerSimplex,
          std::size_t I = 0, typename... Ts>
struct cell_at_return_type_impl {
  using type =
      typename cell_at_return_type_impl<VertexHandle, NumVerticesPerSimplex,
                                        I + 1, Ts..., VertexHandle>::type;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename VertexHandle, std::size_t NumVerticesPerSimplex,
          typename... Ts>
struct cell_at_return_type_impl<VertexHandle, NumVerticesPerSimplex,
                                NumVerticesPerSimplex, Ts...> {
  using type = std::tuple<Ts...>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename VertexHandle, std::size_t NumVerticesPerSimplex>
using cell_at_return_type_t =
    typename cell_at_return_type_impl<VertexHandle,
                                      NumVerticesPerSimplex>::type;
//==============================================================================
template <typename Mesh, typename Real, std::size_t NumDimensions,
          std::size_t SimplexDim>
struct hierarchy_impl {
  using type = void;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real, std::size_t NumDimensions,
          std::size_t SimplexDim>
using hierarchy =
    typename hierarchy_impl<Mesh, Real, NumDimensions, SimplexDim>::type;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real>
struct hierarchy_impl<Mesh, Real, 3, 3> {
  using type = uniform_tree_hierarchy<Mesh>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real>
struct hierarchy_impl<Mesh, Real, 2, 2> {
  using type = uniform_tree_hierarchy<Mesh>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, typename Real>
struct hierarchy_impl<Mesh, Real, 3, 2> {
  using type = uniform_tree_hierarchy<Mesh>;
};
//==============================================================================
using tatooine::pointset;
template <typename Mesh, typename Real, std::size_t NumDimensions,
          std::size_t SimplexDim>
struct parent : pointset<Real, NumDimensions> {
  using typename pointset<Real, NumDimensions>::vertex_handle;
  using hierarchy_t = hierarchy<Mesh, Real, NumDimensions, SimplexDim>;
  using const_cell_at_return_type =
      cell_at_return_type_t<vertex_handle const&, SimplexDim + 1>;
  using cell_at_return_type =
      cell_at_return_type_t<vertex_handle&, SimplexDim + 1>;
};
//==============================================================================
template <typename Mesh, typename Real>
struct parent<Mesh, Real, 3, 2> : pointset<Real, 3>,
                                  ray_intersectable<Real, 3> {
  using real_t = Real;
  using typename pointset<real_t, 3>::vertex_handle;
  using hierarchy_t = hierarchy<Mesh, real_t, 3, 2>;
  using const_cell_at_return_type =
      cell_at_return_type_t<vertex_handle const&, 3>;
  using cell_at_return_type = cell_at_return_type_t<vertex_handle&, 3>;

  using typename ray_intersectable<real_t, 3>::ray_t;
  using typename ray_intersectable<real_t, 3>::intersection_t;
  using typename ray_intersectable<real_t, 3>::optional_intersection_t;
  //----------------------------------------------------------------------------
  virtual ~parent() = default;
  auto as_grid() const -> auto const& {
    return *dynamic_cast<Mesh const*>(this);
  }
  //----------------------------------------------------------------------------
  auto check_intersection(ray_t const& r, real_t const min_t = 0) const
      -> optional_intersection_t override {
    constexpr double eps          = 1e-6;
    auto const&      grid         = as_grid();
    auto             global_min_t = std::numeric_limits<real_t>::max();
    auto             inters       = optional_intersection_t{};
    if (!grid.m_hierarchy) {
      grid.build_hierarchy();
    }
    auto const possible_cells =
        grid.m_hierarchy->collect_possible_intersections(r);
    for (auto const cell_handle : possible_cells) {
      auto const [vi0, vi1, vi2] = grid.cell_at(cell_handle);
      auto const& v0             = grid.at(vi0);
      auto const& v1             = grid.at(vi1);
      auto const& v2             = grid.at(vi2);
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
          inters =
              intersection_t{this, r, t, pos, normalize(cross(v0v1, v2 - v1))};
        }
      }
    }

    return inters;
  }
};
//==============================================================================
template <typename Real, std::size_t NumDimensions, std::size_t SimplexDim>
struct cell_container;
//==============================================================================
}  // namespace detail::unstructured_simplicial_grid
//==============================================================================
template <typename Real, std::size_t NumDimensions,
          std::size_t SimplexDim = NumDimensions>
struct unstructured_simplicial_grid
    : detail::unstructured_simplicial_grid::parent<
          unstructured_simplicial_grid<Real, NumDimensions, SimplexDim>, Real,
          NumDimensions, SimplexDim> {
  using this_t = unstructured_simplicial_grid<Real, NumDimensions, SimplexDim>;
  using parent_t =
      detail::unstructured_simplicial_grid::parent<this_t, Real, NumDimensions,
                                                   SimplexDim>;
  friend struct detail::unstructured_simplicial_grid::parent<
      this_t, Real, NumDimensions, SimplexDim>;
  using parent_t::at;
  using parent_t::num_dimensions;
  using typename parent_t::pos_t;
  using typename parent_t::vertex_handle;
  using parent_t::operator[];
  using parent_t::insert_vertex;
  using parent_t::invalid_vertices;
  using parent_t::is_valid;
  using parent_t::vertex_position_data;
  using parent_t::vertex_properties;
  using parent_t::vertices;

  using typename parent_t::cell_at_return_type;
  using typename parent_t::const_cell_at_return_type;

  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  using hierarchy_t       = typename parent_t::hierarchy_t;
  static constexpr auto num_vertices_per_simplex() { return SimplexDim + 1; }
  static constexpr auto simplex_dimension() { return SimplexDim; }
  //----------------------------------------------------------------------------
  template <typename T>
  struct vertex_property_sampler_t : field<vertex_property_sampler_t<T>, Real,
                                           parent_t::num_dimensions(), T> {
   private:
    using grid_t =
        unstructured_simplicial_grid<Real, NumDimensions, SimplexDim>;
    using this_t = vertex_property_sampler_t<T>;

    grid_t const&               m_grid;
    vertex_property_t<T> const& m_prop;
    //--------------------------------------------------------------------------
   public:
    vertex_property_sampler_t(grid_t const&               grid,
                              vertex_property_t<T> const& prop)
        : m_grid{grid}, m_prop{prop} {}
    //--------------------------------------------------------------------------
    auto grid() const -> auto const& { return m_grid; }
    auto property() const -> auto const& { return m_prop; }
    //--------------------------------------------------------------------------
    [[nodiscard]] auto evaluate(pos_t const& x, real_t const /*t*/) const -> T {
      return evaluate(x,
                      std::make_index_sequence<num_vertices_per_simplex()>{});
    }
    //--------------------------------------------------------------------------
    template <std::size_t... VertexSeq>
    [[nodiscard]] auto evaluate(pos_t const& x,
                                std::index_sequence<VertexSeq...> /*seq*/) const
        -> T {
      auto cell_handles = m_grid.hierarchy().nearby_cells(x);
      if (cell_handles.empty()) {
        std::stringstream ss;
        ss << "[unstructured_simplicial_grid::vertex_property_sampler_t::"
              "sample]"
              "\n";
        ss << "  out of domain: " << x;
        throw std::runtime_error{ss.str()};
      }
      for (auto t : cell_handles) {
        auto const            vs = m_grid.cell_at(t);
        static constexpr auto NV = num_vertices_per_simplex();
        auto                  A  = mat<Real, NV, NV>::ones();
        auto                  b  = vec<Real, NV>::ones();
        for (std::size_t r = 0; r < num_dimensions(); ++r) {
          (
              [&]() {
                if (VertexSeq > 0) {
                  A(r, VertexSeq) = m_grid[std::get<VertexSeq>(vs)](r) -
                                    m_grid[std::get<0>(vs)](r);
                } else {
                  ((A(r, VertexSeq) = 0), ...);
                }
              }(),
              ...);

          b(r) = x(r) - m_grid[std::get<0>(vs)](r);
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
      std::stringstream ss;
      ss << "[unstructured_simplicial_grid::vertex_property_sampler_t::sample]"
            "\n";
      ss << "  out of domain: " << x;
      throw std::runtime_error{ss.str()};
      return T{};
    }
  };
  //----------------------------------------------------------------------------
  struct cell_handle : handle<cell_handle> {
    using handle<cell_handle>::handle;
  };
  using cell_container =
      detail::unstructured_simplicial_grid::cell_container<Real, NumDimensions,
                                                           SimplexDim>;
  friend struct detail::unstructured_simplicial_grid::cell_container<
      Real, NumDimensions, SimplexDim>;
  //----------------------------------------------------------------------------
  template <typename T>
  using cell_property_t = vector_property_impl<cell_handle, T>;
  using cell_property_container_t =
      std::map<std::string, std::unique_ptr<vector_property<cell_handle>>>;
  //============================================================================
 private:
  std::vector<vertex_handle>           m_cell_index_data;
  std::set<cell_handle>             m_invalid_cells;
  cell_property_container_t            m_cell_properties;
  mutable std::unique_ptr<hierarchy_t> m_hierarchy;

 protected:
  auto cell_index_data() const -> auto const& { return m_cell_index_data; }
  auto cell_index_data() -> auto& { return m_cell_index_data; }

  auto invalid_cells() const -> auto const& { return m_invalid_cells; }
  auto invalid_cells() -> auto& { return m_invalid_cells; }

  auto cell_properties() const -> auto const& { return m_cell_properties; }
  auto cell_properties() -> auto& { return m_cell_properties; }

 public:
  //============================================================================
  constexpr unstructured_simplicial_grid() = default;
  //============================================================================
  unstructured_simplicial_grid(unstructured_simplicial_grid const& other)
      : parent_t{other}, m_cell_index_data{other.m_cell_index_data} {
    for (auto const& [key, prop] : other.cell_properties()) {
      cell_properties().insert(std::pair{key, prop->clone()});
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  unstructured_simplicial_grid(unstructured_simplicial_grid&& other) noexcept =
      default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(unstructured_simplicial_grid const& other)
      -> unstructured_simplicial_grid& {
    parent_t::operator=(other);
    cell_properties().clear();
    m_cell_index_data = other.m_cell_index_data;
    for (auto const& [key, prop] : other.cell_properties()) {
      cell_properties().insert(std::pair{key, prop->clone()});
    }
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(unstructured_simplicial_grid&& other) noexcept
      -> unstructured_simplicial_grid& = default;
  //----------------------------------------------------------------------------
  explicit unstructured_simplicial_grid(std::filesystem::path const& path) {
    read(path);
  }
  //----------------------------------------------------------------------------
  unstructured_simplicial_grid(std::initializer_list<pos_t>&& vertices)
      : parent_t{std::move(vertices)} {}
  explicit unstructured_simplicial_grid(
      std::vector<vec<Real, NumDimensions>> const& positions)
      : parent_t{positions} {}
  explicit unstructured_simplicial_grid(
      std::vector<vec<Real, NumDimensions>>&& positions)
      : parent_t{std::move(positions)} {}
  //----------------------------------------------------------------------------
 private:
  template <typename... TypesToCheck, typename Prop, typename Grid>
  auto copy_prop(std::string const& name, Prop const& prop,
                 Grid const& other_grid) {
    (([&]() {
       if (prop->type() == typeid(TypesToCheck)) {
         auto const& other_prop =
             other_grid.template vertex_property<TypesToCheck>(name);
         auto& prop = this->template vertex_property<TypesToCheck>(name);
         auto  vi   = vertex_handle{0};
         other_grid.vertices().iterate_indices(
             [&](auto const... is) { prop[vi++] = other_prop(is...); });
       }
     }()),
     ...);
  }

 public:
  template <indexable_space DimX, indexable_space DimY>
  requires(NumDimensions == 2) &&
      (SimplexDim == 2) explicit unstructured_simplicial_grid(
          rectilinear_grid<DimX, DimY> const& g) {
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

  template <indexable_space DimX, indexable_space DimY, indexable_space DimZ>
  requires(NumDimensions == 3) &&
      (SimplexDim == 3) explicit unstructured_simplicial_grid(
          rectilinear_grid<DimX, DimY, DimZ> const& g) {
    constexpr auto turned = [](std::size_t const ix, std::size_t const iy,
                               std::size_t const iz) -> bool {
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
    gc.iterate_vertices([&](auto const ix, auto const iy, auto const iz) {
      auto const le_bo_fr = vertex_handle{ix + iy * s0 + iz * s0s1};
      auto const ri_bo_fr = vertex_handle{(ix + 1) + iy * s0 + iz * s0s1};
      auto const le_to_fr = vertex_handle{ix + (iy + 1) * s0 + iz * s0s1};
      auto const ri_to_fr = vertex_handle{(ix + 1) + (iy + 1) * s0 + iz * s0s1};
      auto const le_bo_ba = vertex_handle{ix + iy * s0 + (iz + 1) * s0s1};
      auto const ri_bo_ba = vertex_handle{(ix + 1) + iy * s0 + (iz + 1) * s0s1};
      auto const le_to_ba = vertex_handle{ix + (iy + 1) * s0 + (iz + 1) * s0s1};
      auto const ri_to_ba =
          vertex_handle{(ix + 1) + (iy + 1) * s0 + (iz + 1) * s0s1};
      if (turned(ix, iy, iz)) {
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
    });
    for (auto const& [name, prop] : g.vertex_properties()) {
      copy_prop<mat4d, mat3d, mat2d, mat4f, mat3f, mat2f, vec4d, vec3d, vec2d,
                vec4f, vec3f, vec2f, double, float, std::int8_t, std::uint8_t,
                std::int16_t, std::uint16_t, std::int32_t, std::uint32_t,
                std::int64_t, std::uint64_t>(name, prop, g);
    }
  }
  //============================================================================
  auto operator[](cell_handle t) const -> auto { return cell_at(t.index()); }
  auto operator[](cell_handle t) -> auto { return cell_at(t.index()); }
  //----------------------------------------------------------------------------
  auto at(cell_handle t) const -> auto { return cell_at(t.index()); }
  auto at(cell_handle t) -> auto { return cell_at(t.index()); }
  //----------------------------------------------------------------------------
  auto cell_at(cell_handle t) const -> auto { return cell_at(t.index()); }
  auto cell_at(cell_handle t) -> auto { return cell_at(t.index()); }
  //----------------------------------------------------------------------------
  template <std::size_t... Seq>
  auto cell_at(std::size_t const i) const {
    return cell_at(i, std::make_index_sequence<num_vertices_per_simplex()>{});
  }
  template <std::size_t... Seq>
  auto cell_at(std::size_t const i) {
    return cell_at(i, std::make_index_sequence<num_vertices_per_simplex()>{});
  }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Seq>
  auto cell_at(std::size_t const i, std::index_sequence<Seq...> /*seq*/) const
      -> const_cell_at_return_type {
    return {cell_index_data()[i * num_vertices_per_simplex() + Seq]...};
  }
  template <std::size_t... Seq>
  auto cell_at(std::size_t const i, std::index_sequence<Seq...> /*seq*/)
      -> cell_at_return_type {
    return {cell_index_data()[i * num_vertices_per_simplex() + Seq]...};
  }
  //----------------------------------------------------------------------------
 public:
  auto insert_vertex(arithmetic auto const... comps) requires(
      sizeof...(comps) == NumDimensions) {
    auto const vi = parent_t::insert_vertex(comps...);
    // if (m_hierarchy != nullptr) {
    //  if (!m_hierarchy->insert_vertex(vi.index())) {
    //    build_hierarchy();
    //  }
    //}
    return vi;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t const& v) {
    auto const vi = parent_t::insert_vertex(v);
    // if (m_hierarchy != nullptr) {
    //  if (!m_hierarchy->insert_vertex(vi.index())) {
    //    build_hierarchy();
    //  }
    //}
    return vi;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& v) {
    auto const vi = parent_t::insert_vertex(std::move(v));
    // if (m_hierarchy != nullptr) {
    //  if (!m_hierarchy->insert_vertex(vi.index())) {
    //    build_hierarchy();
    //  }
    //}
    return vi;
  }
  //----------------------------------------------------------------------------
  auto remove(vertex_handle const vh) {
    using namespace std::ranges;
    if (is_valid(vh) &&
        find(invalid_vertices(), vh) == end(invalid_vertices())) {
      invalid_vertices().push_back(vh);
    }
    copy(cells() | views::filter(
                       [this, vh](auto const ch) { return contains(ch, vh); }),
         std::back_inserter(invalid_cells()));
  }
  //----------------------------------------------------------------------------
  auto remove(cell_handle const ch) {
    invalid_cells().insert(ch);
  }
  //----------------------------------------------------------------------------
  template <typename... Handles>
  auto insert_cell(Handles const... handles) {
    static_assert(sizeof...(Handles) == num_vertices_per_simplex(),
                  "wrong number of vertices for simplex");
    (cell_index_data().push_back(handles), ...);
    for (auto& [key, prop] : cell_properties()) {
      prop->push_back();
    }
    return cell_handle{cells().size() - 1};
  }
  //----------------------------------------------------------------------------
  /// tidies up invalid vertices
 private:
  template <std::size_t... Is>
  auto reindex_cells_vertex_handles(std::index_sequence<Is...> /*seq*/) {
    auto offsets = std::vector<long>(size(vertex_position_data()));
    for (auto const v : invalid_vertices()) {
      for (std::size_t i = v.index(); i < size(offsets); ++i) {
        --offsets[i];
      }
    }
    for (auto const c : cells()) {
      auto vs = at(c);
      (
          [&] {
            auto& v = std::get<Is>(vs);
            if (is_valid(v)) {
              v -= offsets[v.index()];
            }
          }(),
          ...);
    }
  }
  //----------------------------------------------------------------------------
 public:
  auto tidy_up() {
    reindex_cells_vertex_handles(std::make_index_sequence<num_dimensions()>{});
    parent_t::tidy_up();
    for (auto const c : invalid_vertices()) {
      cell_index_data().erase(begin(cell_index_data()) + c.index());
      for (auto const& [key, prop] : cell_properties()) {
        prop->erase(c.index());
      }
    }
    invalid_cells().clear();
  }
  //----------------------------------------------------------------------------
  auto clear() {
    parent_t::clear();
    cell_index_data().clear();
  }
  //----------------------------------------------------------------------------
  auto cells() const { return cell_container{this}; }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  auto contains(cell_handle const ch, vertex_handle const vh,
                std::index_sequence<Is...> /*seq*/) const {
    auto cells_vertices = at(ch);
    return ((std::get<Is>(cells_vertices) == vh) || ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto contains(cell_handle const ch, vertex_handle const vh) const {
    return contains(ch, vh, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
#ifdef TATOOINE_HAS_CGAL_SUPPORT
  auto build_delaunay_mesh() requires(NumDimensions == 2) ||
      (NumDimensions == 3) {
    build_delaunay_mesh(std::make_index_sequence<NumDimensions>{});
  }

 private:
  template <std::size_t... Seq>
      auto build_delaunay_mesh(std::index_sequence<Seq...> /*seq*/)
          -> void requires(NumDimensions == 2) ||
      (NumDimensions == 3) {
    cell_index_data().clear();
    using kernel_t      = CGAL::Exact_predicates_inexact_constructions_kernel;
    using vertex_base_t = std::conditional_t<
        NumDimensions == 2,
        CGAL::Triangulation_vertex_base_with_info_2<vertex_handle, kernel_t>,
        CGAL::Triangulation_vertex_base_with_info_3<vertex_handle, kernel_t>>;
    using triangulation_data_t =
        std::conditional_t<NumDimensions == 2,
                           CGAL::Triangulation_data_structure_2<vertex_base_t>,
                           CGAL::Triangulation_data_structure_3<vertex_base_t>>;
    using triangulation_t = std::conditional_t<
        NumDimensions == 2,
        CGAL::Delaunay_triangulation_2<kernel_t, triangulation_data_t>,
        CGAL::Delaunay_triangulation_3<kernel_t, triangulation_data_t>>;
    using point_t = typename triangulation_t::Point;
    std::vector<std::pair<point_t, vertex_handle>> points;
    points.reserve(vertices().size());
    for (auto v : vertices()) {
      points.emplace_back(point_t{at(v)(Seq)...}, v);
    }

    triangulation_t triangulation{begin(points), end(points)};
    if constexpr (NumDimensions == 2) {
      for (auto it = triangulation.finite_faces_begin();
           it != triangulation.finite_faces_end(); ++it) {
        insert_cell(vertex_handle{it->vertex(0)->info()},
                    vertex_handle{it->vertex(Seq + 1)->info()}...);
      }
    } else if constexpr (NumDimensions == 3) {
      for (auto it = triangulation.finite_cells_begin();
           it != triangulation.finite_cells_end(); ++it) {
        insert_cell(vertex_handle{it->vertex(0)->info()},
                    vertex_handle{it->vertex(Seq + 1)->info()}...);
      }
    }
  }
#endif

#if TATOOINE_CDT_AVAILABLE
 public:
  auto build_delaunay_mesh(
      std::vector<std::pair<vertex_handle, vertex_handle>> const& constraints)
          -> void requires(NumDimensions == 2) ||
      (NumDimensions == 3) {
    build_delaunay_mesh(constraints, std::make_index_sequence<NumDimensions>{});
  }

 private:
  template <std::size_t... Seq>
  requires(NumDimensions == 2) /*|| (NumDimensions == 3)*/
      auto build_delaunay_mesh(
          std::vector<std::pair<vertex_handle, vertex_handle>> const&
              constraints,
          std::index_sequence<Seq...> /*seq*/) -> void {
    cell_index_data().clear();
    std::vector<CDT::Edge> edges;
    edges.reserve(size(constraints));
    boost::transform(constraints, std::back_inserter(edges),
                     [](auto const& c) -> CDT::Edge {
                       return {c.first.index(), c.second.index()};
                     });
    auto triangulation =
        CDT::Triangulation<Real>{CDT::FindingClosestPoint::BoostRTree};

    triangulation.insertVertices(
        vertices().begin(), vertices().end(),
        [this](auto const& v) { return this->vertex_at(v)(0); },
        [this](auto const& v) { return this->vertex_at(v)(1); });
    auto const duplicates_info = CDT::RemoveDuplicatesAndRemapEdges<Real>(
        triangulation.vertices, edges,
        [this](auto const& v) { return v.pos.x; },
        [this](auto const& v) { return v.pos.y; });

    triangulation.insertEdges(edges);
    triangulation.eraseSuperTriangle();
    // triangulation.eraseOuterTrianglesAndHoles();
    for (auto const& tri : triangulation.triangles) {
      insert_cell(vertex_handle{tri.vertices[0]},
                  vertex_handle{tri.vertices[1]},
                  vertex_handle{tri.vertices[2]});
    }
  }
#endif
  //----------------------------------------------------------------------------
 public:
  template <typename T>
  auto cell_property(std::string const& name) -> auto& {
    auto it = cell_properties().find(name);
    if (it == end(cell_properties())) {
      return insert_cell_property<T>(name);
    }
    if (typeid(T) != it->second->type()) {
      throw std::runtime_error{
          "type of property \"" + name + "\"(" +
          boost::core::demangle(it->second->type().name()) +
          ") does not match specified type " + type_name<T>() + "."};
    }
    return *dynamic_cast<cell_property_t<T>*>(cell_properties().at(name).get());
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto cell_property(std::string const& name) const -> const auto& {
    auto it = cell_properties().find(name);
    if (it == end(cell_properties())) {
      throw std::runtime_error{"property \"" + name + "\" not found"};
    }
    if (typeid(T) != it->second->type()) {
      throw std::runtime_error{
          "type of property \"" + name + "\"(" +
          boost::core::demangle(it->second->type().name()) +
          ") does not match specified type " + type_name<T>() + "."};
    }
    return *dynamic_cast<cell_property_t<T>*>(cell_properties().at(name).get());
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
    auto [it, suc] = cell_properties().insert(
        std::pair{name, std::make_unique<cell_property_t<T>>(value)});
    auto prop = dynamic_cast<cell_property_t<T>*>(it->second.get());
    prop->resize(cells().size());
    return *prop;
  }
  //----------------------------------------------------------------------------
  auto write_vtk(std::filesystem::path const& path,
                 std::string const&           title = "tatooine grid") const {
    if constexpr (SimplexDim == 2 || SimplexDim == 3) {
      write_unstructured_triangular_grid_vtk(path, title);
    }
  }
  //----------------------------------------------------------------------------
  auto write_vtp(filesystem::path const& path) const {
    auto file = std::ofstream{path, std::ios::binary};
    if (!file.is_open()) {
      throw std::runtime_error{"Could not write " + path.string()};
    }
    auto offset       = std::size_t{};
    using header_type = std::uint64_t;
    file << "<VTKFile"
         << " type=\"PolyData\""
         << " version=\"1.0\" "
            "byte_order=\"LittleEndian\""
         << " header_type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<header_type>())
         << "\">";
    file << "<PolyData>\n";
    file << "<Piece"
         << " NumberOfPoints=\"" << vertices().size() << "\""
         << " NumberOfPolys=\"" << cells().size() << "\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\"0\""
         << " NumberOfStrips=\"0\""
         << ">\n";

    // Points
    file << "<Points>";
    file << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<Real>())
         << "\" NumberOfComponents=\"" << num_dimensions() << "\"/>";
    auto const num_bytes_points =
        header_type(sizeof(Real) * num_dimensions() * vertices().size());
    offset += num_bytes_points + sizeof(header_type);

    file << "</Points>\n";

    // Polys
    file << "<Polys>\n";
    // Polys - connectivity
    using polys_connectivity_int_t = std::int32_t;
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<polys_connectivity_int_t>())
         << "\" Name=\"connectivity\"/>\n";
    auto const num_bytes_polys_connectivity = cells().size() *
                                              num_vertices_per_simplex() *
                                              sizeof(polys_connectivity_int_t);
    offset += num_bytes_polys_connectivity + sizeof(header_type);

    // Polys - offsets
    using polys_offset_int_t = polys_connectivity_int_t;
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<polys_offset_int_t>())
         << "\" Name=\"offsets\"/>\n";
    auto const num_bytes_polys_offsets =
        sizeof(polys_offset_int_t) * cells().size();
    offset += num_bytes_polys_offsets + sizeof(header_type);
    file << "</Polys>\n";
    file << "</Piece>\n";
    file << "</PolyData>\n";
    file << "<AppendedData encoding=\"raw\">_";
    // Writing vertex data to appended data section
    auto arr_size = header_type{};

    {
      auto point_data = std::vector<pos_t>(vertices().size());
      using namespace std::ranges;
      auto position = [this](auto const v) -> auto& { return at(v); };
      copy(vertices() | views::transform(position), begin(point_data));
      arr_size = num_bytes_polys_connectivity;
      arr_size = num_bytes_points;
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(point_data.data()),
                 num_bytes_points);
      std::cout << "points\n";
      for (auto const x : point_data) {
        std::cout << x << '\n';
      }
    }

    // Writing polys connectivity data to appended data section
    {
      auto connectivity_data = std::vector<polys_connectivity_int_t>(
          cells().size() * num_vertices_per_simplex());
      std::ranges::copy(cells().data_container() |
                            std::views::transform(
                                [](auto const x) -> polys_connectivity_int_t {
                                  return x.index();
                                }),
                        begin(connectivity_data));
      arr_size = num_bytes_polys_connectivity;
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 num_bytes_polys_connectivity);
      std::cout << "connectivity_data\n";
      for (auto const x : connectivity_data) {
        std::cout << x << '\n';
      }
    }

    // Writing polys offsets to appended data section
    {
      auto offsets = std::vector<polys_offset_int_t>(
          cells().size(), num_vertices_per_simplex());
      for (std::size_t i = 1; i < size(offsets); ++i) {
        offsets[i] += offsets[i - 1];
      }
      arr_size = num_bytes_polys_offsets;
      std::cout << "offsets\n";
      for (auto const x : offsets) {
        std::cout << x << '\n';
      }
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(offsets.data()),
                 num_bytes_polys_offsets);
    }
    file << "</AppendedData>";
    file << "</VTKFile>";
  }

 private:
  template <std::size_t SimplexDim_     = SimplexDim,
            enable_if<SimplexDim_ == 2> = true>
  auto write_unstructured_triangular_grid_vtk(std::filesystem::path const& path,
                                              std::string const& title) const
      -> bool {
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
        copy(vertex_position_data() | three_dimensional, begin(v3s));
        writer.write_points(v3s);

      } else if constexpr (NumDimensions == 3) {
        writer.write_points(vertex_position_data());
      }

      // auto vertices_per_cell = std::vector<std::vector<std::size_t>> {};
      // vertices_per_cell.reserve(cells().size());
      // auto cell_types =
      //    std::vector<vtk::cell_type>(cells().size(),
      //    vtk::cell_type::triangle);
      // for (auto const c : cells()) {
      //  auto const [v0, v1, v2] = at(c);
      //  vertices_per_cell.push_back(std::vector{v0.index(), v1.index(),
      //  v2.index()});
      //}
      // writer.write_cells(vertices_per_cell);
      // writer.write_cell_types(cell_types);
      //
      //// write vertex_handle data
      // writer.write_point_data(vertices().size());
      // for (auto const& [name, prop] : vertex_properties()) {
      //  if (prop->type() == typeid(vec<Real, 4>)) {
      //    auto const& casted_prop =
      //        *dynamic_cast<vertex_property_t<vec<Real, 4>>
      //        const*>(prop.get());
      //    writer.write_scalars(name, casted_prop.data());
      //  } else if (prop->type() == typeid(vec<Real, 3>)) {
      //    auto const& casted_prop =
      //        *dynamic_cast<vertex_property_t<vec<Real, 3>>
      //        const*>(prop.get());
      //    writer.write_scalars(name, casted_prop.data());
      //  } else if (prop->type() == typeid(vec<Real, 2>)) {
      //    auto const& casted_prop =
      //        *dynamic_cast<vertex_property_t<vec<Real, 2>>
      //        const*>(prop.get());
      //    writer.write_scalars(name, casted_prop.data());
      //  } else if (prop->type() == typeid(Real)) {
      //    auto const& casted_prop =
      //        *dynamic_cast<vertex_property_t<Real> const*>(prop.get());
      //    writer.write_scalars(name, casted_prop.data());
      //  }
      //}

      writer.close();
      return true;
    }
    return false;
  }
  //----------------------------------------------------------------------------
  template <std::size_t SimplexDim_     = SimplexDim,
            enable_if<SimplexDim_ == 3> = true>
  auto write_unstructured_tetrahedral_grid_vtk(
      std::filesystem::path const& path, std::string const& title) const
      -> bool {
    using boost::copy;
    using boost::adaptors::transformed;
    vtk::legacy_file_writer writer(path, vtk::dataset_type::unstructured_grid);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      writer.write_points(vertex_position_data());

      std::vector<std::vector<std::size_t>> vertices_per_cell;
      vertices_per_cell.reserve(cells().size());
      std::vector<vtk::cell_type> cell_types(cells().size(),
                                             vtk::cell_type::tetra);
      for (auto const t : cells()) {
        auto const [v0, v1, v2, v3] = at(t);
        vertices_per_cell.push_back(
            std::vector{v0.index(), v1.index(), v2.index(), v3.index()});
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
          writer.write_scalars(name, casted_prop.data());
        } else if (prop->type() == typeid(Real)) {
          auto const& casted_prop =
              *dynamic_cast<vertex_property_t<Real> const*>(prop.get());
          writer.write_scalars(name, casted_prop.data());
        }
      }

      writer.close();
      return true;
    }
    return false;
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
  template <std::size_t NumDimensions_ = NumDimensions,
            enable_if<NumDimensions_ == 2 || NumDimensions_ == 3> = true>
  auto read_vtk(std::filesystem::path const& path) {
    struct listener_t : vtk::legacy_file_listener {
      unstructured_simplicial_grid& grid;
      std::vector<int>              cells;

      explicit listener_t(unstructured_simplicial_grid& _grid) : grid{_grid} {}
      auto add_cells(std::vector<int> const& cells) -> void {
        std::size_t i = 0;
        while (i < size(cells)) {
          auto const num_vertices = cells[i++];
          if (num_vertices != num_vertices_per_simplex()) {
            throw std::runtime_error{
                "Number of vertices in file does not match number of vertices "
                "per simplex."};
          }
          for (std::size_t j = 0; j < static_cast<std::size_t>(num_vertices);
               ++j) {
            grid.cell_index_data().push_back(vertex_handle{cells[i++]});
          }
          for (auto& [key, prop] : grid.cell_properties()) {
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
              "[unstructured_simplicial_grid] need polydata or "
              "unstructured_grid "
              "when reading vtk legacy"};
        }
      }

      auto on_points(std::vector<std::array<float, 3>> const& ps)
          -> void override {
        for (const auto& p : ps) {
          if constexpr (NumDimensions == 2) {
            grid.insert_vertex(static_cast<Real>(p[0]),
                               static_cast<Real>(p[1]));
          }
          if constexpr (NumDimensions == 3) {
            grid.insert_vertex(static_cast<Real>(p[0]), static_cast<Real>(p[1]),
                               static_cast<Real>(p[2]));
          }
        }
      }
      auto on_points(std::vector<std::array<double, 3>> const& ps)
          -> void override {
        for (const auto& p : ps) {
          if constexpr (NumDimensions == 2) {
            grid.insert_vertex(static_cast<Real>(p[0]),
                               static_cast<Real>(p[1]));
          }
          if constexpr (NumDimensions == 3) {
            grid.insert_vertex(static_cast<Real>(p[0]), static_cast<Real>(p[1]),
                               static_cast<Real>(p[2]));
          }
        }
      }
      auto on_polygons(std::vector<int> const& ps) -> void override {
        add_cells(ps);
      }
      auto on_scalars(std::string const& data_name,
                      std::string const& /*lookup_table_name*/,
                      std::size_t num_comps, std::vector<double> const& scalars,
                      vtk::reader_data data) -> void override {
        if (data == vtk::reader_data::point_data) {
          if (num_comps == 1) {
            auto& prop =
                grid.template insert_vertex_property<double>(data_name);
            for (auto v = vertex_handle{0}; v < vertex_handle{prop.size()};
                 ++v) {
              prop[v] = scalars[v.index()];
            }
          } else if (num_comps == 2) {
            auto& prop =
                grid.template insert_vertex_property<vec<double, 2>>(data_name);

            for (auto v = vertex_handle{0}; v < vertex_handle{prop.size()};
                 ++v) {
              for (std::size_t j = 0; j < num_comps; ++j) {
                prop[v][j] = scalars[v.index() * num_comps + j];
              }
            }
          } else if (num_comps == 3) {
            auto& prop =
                grid.template insert_vertex_property<vec<double, 3>>(data_name);
            for (auto v = vertex_handle{0}; v < vertex_handle{prop.size()};
                 ++v) {
              for (std::size_t j = 0; j < num_comps; ++j) {
                prop[v][j] = scalars[v.index() * num_comps + j];
              }
            }
          } else if (num_comps == 4) {
            auto& prop =
                grid.template insert_vertex_property<vec<double, 4>>(data_name);
            for (auto v = vertex_handle{0}; v < vertex_handle{prop.size()};
                 ++v) {
              for (std::size_t j = 0; j < num_comps; ++j) {
                prop[v][j] = scalars[v.index() * num_comps + j];
              }
            }
          }
        } else if (data == vtk::reader_data::cell_data) {
          if (num_comps == 1) {
            auto& prop = grid.template insert_cell_property<double>(data_name);
            for (auto c = cell_handle{0}; c < cell_handle{prop.size()}; ++c) {
              prop[c] = scalars[c.index()];
            }
          } else if (num_comps == 2) {
            auto& prop =
                grid.template insert_cell_property<vec<double, 2>>(data_name);

            for (auto c = cell_handle{0}; c < cell_handle{prop.size()}; ++c) {
              for (std::size_t j = 0; j < num_comps; ++j) {
                prop[c][j] = scalars[c.index() * num_comps + j];
              }
            }
          } else if (num_comps == 3) {
            auto& prop =
                grid.template insert_cell_property<vec<double, 3>>(data_name);
            for (auto c = cell_handle{0}; c < cell_handle{prop.size()}; ++c) {
              for (std::size_t j = 0; j < num_comps; ++j) {
                prop[c][j] = scalars[c.index() * num_comps + j];
              }
            }
          } else if (num_comps == 4) {
            auto& prop =
                grid.template insert_cell_property<vec<double, 4>>(data_name);
            for (auto c = cell_handle{0}; c < cell_handle{prop.size()}; ++c) {
              for (std::size_t j = 0; j < num_comps; ++j) {
                prop[c][j] = scalars[c.index() * num_comps + j];
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
  constexpr auto is_valid(cell_handle t) const {
    return std::ranges::find(invalid_cells(), t) == end(invalid_cells());
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
      m_hierarchy   = std::make_unique<hierarchy_t>(bb.min(), bb.max(), *this);
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
//==============================================================================unstructured_simplicial_grid()->unstructured_simplicial_grid<double,
// 3>;
unstructured_simplicial_grid(std::string const&)
    ->unstructured_simplicial_grid<double, 3>;
template <typename... Dims>
unstructured_simplicial_grid(rectilinear_grid<Dims...> const& g)
    -> unstructured_simplicial_grid<typename rectilinear_grid<Dims...>::real_t,
                                    sizeof...(Dims)>;
//==============================================================================
namespace detail::unstructured_simplicial_grid {
//==============================================================================
template <typename Real, std::size_t NumDimensions, std::size_t SimplexDim>
struct cell_container {
  using grid_t =
      tatooine::unstructured_simplicial_grid<Real, NumDimensions, SimplexDim>;
  using handle_t = typename grid_t::cell_handle;
  //----------------------------------------------------------------------------
  struct iterator : iterator_facade<iterator> {
    struct sentinel_type {};
    iterator() = default;
    iterator(handle_t const ch, grid_t const* ps) : m_ch{ch}, m_ps{ps} {}
    iterator(iterator const& other) : m_ch{other.m_ch}, m_ps{other.m_ps} {}

   private:
    handle_t      m_ch{};
    grid_t const* m_ps = nullptr;

   public:
    constexpr auto increment() {
      do {
        ++m_ch;
      } while (!m_ps->is_valid(m_ch));
    }
    constexpr auto decrement() {
      do {
        --m_ch;
      } while (!m_ps->is_valid(m_ch));
    }

    [[nodiscard]] constexpr auto equal(iterator const& other) const {
      return m_ch == other.m_ch;
    }
    [[nodiscard]] auto dereference() const { return m_ch; }

    constexpr auto at_end() const {
      return m_ch.index() == m_ps->cell_index_data().size();
    }
  };
  //--------------------------------------------------------------------------
  grid_t const* m_grid;
  //--------------------------------------------------------------------------
  auto begin() const {
    iterator vi{handle_t{0}, m_grid};
    if (!m_grid->is_valid(*vi)) {
      ++vi;
    }
    return vi;
  }
  //--------------------------------------------------------------------------
  auto end() const { return iterator{handle_t{size()}, m_grid}; }
  //--------------------------------------------------------------------------
  auto size() const {
    return m_grid->cell_index_data().size() /
               m_grid->num_vertices_per_simplex() -
           m_grid->invalid_cells().size();
  }
  auto data_container() const -> auto const& { return m_grid->cell_index_data(); }
  auto data() const { return m_grid->cell_index_data().data(); }
  auto operator[](std::size_t const i) const { return m_grid->at(handle_t{i}); }
  auto operator[](std::size_t const i) { return m_grid->at(handle_t{i}); }
  auto operator[](handle_t const i) const { return m_grid->at(i); }
  auto operator[](handle_t const i) { return m_grid->at(i); }
  auto at(std::size_t const i) const { return m_grid->at(handle_t{i}); }
  auto at(std::size_t const i) { return m_grid->at(handle_t{i}); }
  auto at(handle_t const i) const { return m_grid->at(i); }
  auto at(handle_t const i) { return m_grid->at(i); }
};
//------------------------------------------------------------------------------
template <typename Real, size_t NumDimensions, std::size_t SimplexDim>
auto begin(cell_container<Real, NumDimensions, SimplexDim> cells) {
  return cells.begin();
}
//------------------------------------------------------------------------------
template <typename Real, size_t NumDimensions, std::size_t SimplexDim>
auto end(cell_container<Real, NumDimensions, SimplexDim> cells) {
  return cells.end();
}
//------------------------------------------------------------------------------
template <typename Real, size_t NumDimensions, std::size_t SimplexDim>
auto size(cell_container<Real, NumDimensions, SimplexDim> cells) {
  return cells.size();
}
//==============================================================================
}  // namespace detail::unstructured_simplicial_grid
//==============================================================================
// namespace detail {
// template <typename MeshCont>
// auto write_grid_container_to_vtk(MeshContc onst& grids, std::string const&
// path,
//                                 std::string const& title) {
//  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
//  if (writer.is_open()) {
//    std::size_t num_pts = 0;
//    std::size_t cur_first = 0;
//    for (auto const& m : grids) { num_pts += m.num_vertices(); }
//    std::vector<std::array<typename MeshCont::value_type::real_t, 3>>
//    points; std::vector<std::vector<std::size_t>> cells;
//    points.reserve(num_pts); cells.reserve(grids.size());
//
//    for (auto const& m : grids) {
//      // add points
//      for (auto const& v : m.vertices()) {
//        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
//      }
//
//      // add cells
//      for (auto t : m.cells()) {
//        cells.emplace_back();
//        cells.back().push_back(cur_first + m[t][0].index());
//        cells.back().push_back(cur_first + m[t][1].index());
//        cells.back().push_back(cur_first + m[t][2].index());
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
// auto write_vtk(std::vector<unstructured_simplicial_grid<Real, 3>> const&
// grids, std::string const& path,
//               std::string const& title = "tatooine grids") {
//  detail::write_grid_container_to_vtk(grids, path, title);
//}
//------------------------------------------------------------------------------
static constexpr inline auto constrained_delaunay_available(
    std::size_t const NumDimensions) -> bool {
  if (NumDimensions == 2) {
    return cdt_available();
  }
  return false;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
template <typename Real, std::size_t NumDimensions, std::size_t SimplexDim>
inline constexpr bool std::ranges::enable_borrowed_range<
    typename tatooine::detail::unstructured_simplicial_grid::cell_container<
        Real, NumDimensions, SimplexDim>> = true;
#include <tatooine/unstructured_tetrahedral_grid.h>
#include <tatooine/unstructured_triangular_grid.h>
//==============================================================================
#endif
