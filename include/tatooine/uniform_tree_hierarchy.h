#ifndef TATOOINE_UNIFORM_TREE_HIERARCHY_H
#define TATOOINE_UNIFORM_TREE_HIERARCHY_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/for_loop.h>
#include <vector>
#include <tatooine/math.h>

#include <set>
//==============================================================================
namespace tatooine {
//==============================================================================
/// For octrees and quadtrees
template <typename Real, size_t NumDims, typename Derived>
struct base_uniform_tree_hierarchy : aabb<Real, NumDims> {
  template <std::size_t I>
  struct dim {
    enum e : std::uint8_t { left = 0, right = 1 << I };
  };
  using real_type   = Real;
  using this_type   = base_uniform_tree_hierarchy<Real, NumDims, Derived>;
  using parent_t = aabb<Real, NumDims>;
  using parent_t::center;
  using parent_t::is_inside;
  using parent_t::max;
  using parent_t::min;
  using typename parent_t::vec_t;
  friend class std::unique_ptr<this_type>;
  static constexpr auto num_dimensions() { return NumDims; }
  static constexpr auto num_children() {
    return ipow(2, NumDims);
  }

  size_t                     m_level;
  size_t                     m_max_depth;
  std::array<std::unique_ptr<Derived>, num_children()> m_children;
  static constexpr size_t default_max_depth = 4;
  //============================================================================
  base_uniform_tree_hierarchy()                                       = default;
  base_uniform_tree_hierarchy(base_uniform_tree_hierarchy const&)     = default;
  base_uniform_tree_hierarchy(base_uniform_tree_hierarchy&&) noexcept = default;
  auto operator=(base_uniform_tree_hierarchy const&)
      -> base_uniform_tree_hierarchy& = default;
  auto operator=(base_uniform_tree_hierarchy&&) noexcept
      -> base_uniform_tree_hierarchy& = default;
  virtual ~base_uniform_tree_hierarchy() = default;
  //----------------------------------------------------------------------------
  explicit base_uniform_tree_hierarchy(size_t const max_depth = default_max_depth)
      : m_level{1}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  base_uniform_tree_hierarchy(vec_t const& min, vec_t const& max,
                         size_t const max_depth = default_max_depth)
      : parent_t{min, max}, m_level{1}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 protected:
  base_uniform_tree_hierarchy(vec_t const& min, vec_t const& max,
                         size_t const level, size_t const max_depth)
      : parent_t{min, max},
        m_level{level},
        m_max_depth{max_depth} {}
  //============================================================================
 public:
  constexpr auto is_splitted() const { return m_children.front() != nullptr; }
  constexpr auto level() const { return m_level; }
  constexpr auto is_at_max_depth() const { return level() == m_max_depth; }
  auto children() const -> auto const&{ return m_children; }
  //----------------------------------------------------------------------------
 private:
  auto as_derived() -> auto& {
    return *static_cast<Derived*>(this);
  }
  auto as_derived() const -> auto const& {
    return *static_cast<Derived const*>(this);
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Seq, typename... Is>
  static constexpr auto index(std::index_sequence<Seq...> /*seq*/,
                              Is const... is) {
    return ((is << Seq) + ...);
  }
  //----------------------------------------------------------------------------
 public:
  static constexpr auto index(integral auto const... is) {
    static_assert(sizeof...(is) == NumDims,
                  "Number of indices does not match number of dimensions.");
    return index(std::make_index_sequence<NumDims>{}, is...);
  }
  //------------------------------------------------------------------------------
  auto child_at(integral auto const... is) const -> auto const& {
    static_assert(sizeof...(is) == NumDims,
                  "Number of indices does not match number of dimensions.");
    return m_children[index(is...)];
  }
  //------------------------------------------------------------------------------
  template <size_t... Seq>
  auto split(std::index_sequence<Seq...> seq) {
    auto const c  = center();
    auto       it = [&](auto const... is) {
      auto   cur_min = min();
      auto   cur_max = max();
      size_t dim     = 0;
      for (auto const i : std::array{is...}) {
        if (i == 0) {
          cur_max[dim] = c(dim);
        } else if (i == 1) {
          cur_min[dim] = c(dim);
        }
        ++dim;
      }
      m_children[index(seq, is...)] =
          construct(cur_min, cur_max, level() + 1, m_max_depth);
    };
    for_loop(it, ((void)Seq, 2)...);
  }
  //------------------------------------------------------------------------------
  template <typename... Args>
  auto split() {
    split(std::make_index_sequence<NumDims>{});
  }
  //------------------------------------------------------------------------------
  auto distribute() {
    as_derived().distribute();
  }
  //------------------------------------------------------------------------------
  template <typename... Args>
  auto construct(vec_t const& min, vec_t const& max, size_t const level,
                 size_t const max_depth) const -> std::unique_ptr<Derived> {
    return as_derived().construct(min, max, level, max_depth);
  }
  //------------------------------------------------------------------------------
  auto split_and_distribute() {
    split();
    distribute();
  }
};
//==============================================================================
/// For octrees and quadtrees
template <typename Geometry>
struct uniform_tree_hierarchy;
//==============================================================================
template <typename Real, size_t NumDimensions,
          size_t SimplexDim>
class unstructured_simplicial_grid;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Real, size_t NumDims, size_t SimplexDim>
struct uniform_tree_hierarchy<unstructured_simplicial_grid<Real, NumDims, SimplexDim>>
    : base_uniform_tree_hierarchy<
          Real, NumDims,
          uniform_tree_hierarchy<unstructured_simplicial_grid<Real, NumDims, SimplexDim>>> {
  using mesh_t   = unstructured_simplicial_grid<Real, NumDims, SimplexDim>;
  using this_type   = uniform_tree_hierarchy<mesh_t>;
  using parent_t = base_uniform_tree_hierarchy<Real, NumDims, this_type>;
  using real_type   = typename parent_t::real_type;
  using parent_t::center;
  using parent_t::is_inside;
  using parent_t::is_simplex_inside;
  using parent_t::max;
  using parent_t::min;
  using parent_t::is_at_max_depth;
  using parent_t::is_splitted;
  using parent_t::split_and_distribute;
  using parent_t::children;

  using typename parent_t::vec_t;
  using vertex_handle = typename mesh_t::vertex_handle;
  using simplex_handle   = typename mesh_t::simplex_handle;

  mesh_t const*              m_mesh;
  std::vector<vertex_handle> m_vertex_handles;
  std::vector<simplex_handle>   m_simplex_handles;
  //============================================================================
  uniform_tree_hierarchy()                                  = default;
  uniform_tree_hierarchy(uniform_tree_hierarchy const&)     = default;
  uniform_tree_hierarchy(uniform_tree_hierarchy&&) noexcept = default;
  auto operator=(uniform_tree_hierarchy const&)
      -> uniform_tree_hierarchy&                            = default;
  auto operator=(uniform_tree_hierarchy&&) noexcept
      -> uniform_tree_hierarchy&                            = default;
  virtual ~uniform_tree_hierarchy()                         = default;
  //----------------------------------------------------------------------------
  explicit uniform_tree_hierarchy(
      mesh_t const& mesh, size_t const max_depth = parent_t::default_max_depth)
      : parent_t{vec<Real, NumDims>::zeros(), vec<Real, NumDims>::zeros(), 1,
                 max_depth},
        m_mesh{&mesh} {
    parent_t::operator=(mesh.bounding_box());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  uniform_tree_hierarchy(vec_t const& min, vec_t const& max, mesh_t const& mesh,
                         size_t const max_depth = parent_t::default_max_depth)
      : parent_t{min, max, 1, max_depth}, m_mesh{&mesh} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  uniform_tree_hierarchy(vec_t const& min, vec_t const& max, size_t const level,
                         size_t const max_depth, mesh_t const& mesh)
      : parent_t{min, max, level, max_depth}, m_mesh{&mesh} {}
  //============================================================================
 public:
  auto mesh() const -> auto const& { return *m_mesh; }
  auto constexpr holds_vertices() const { return !m_vertex_handles.empty(); }
  auto constexpr holds_simplices() const { return !m_simplex_handles.empty(); }
  //----------------------------------------------------------------------------
  auto num_vertex_handles() const { return size(m_vertex_handles); }
  auto num_simplex_handles() const { return size(m_simplex_handles); }
  //----------------------------------------------------------------------------
  auto insert_vertex(vertex_handle const v) -> bool {
    if (!is_inside(mesh().vertex_at(v))) {
      return false;
    }
    if (holds_vertices()) {
      if (is_at_max_depth()) {
        m_vertex_handles.push_back(v);
      } else {
        split_and_distribute();
        distribute_vertex(v);
      }
    } else {
      if (is_splitted()) {
        distribute_vertex(v);
      } else {
        m_vertex_handles.push_back(v);
      }
    }
    return true;
  }
  //------------------------------------------------------------------------------
 private:
  template <size_t... Is>
  auto insert_simplex(simplex_handle const c, std::index_sequence<Is...> /*seq*/)
      -> bool {
    auto const vs = mesh()[c];
    if (!is_simplex_inside(mesh()[std::get<Is>(vs)]...)) {
      return false;
    }
    if (holds_simplices()) {
      if (is_at_max_depth()) {
        m_simplex_handles.push_back(c);
      } else {
        split_and_distribute();
        distribute_simplex(c);
      }
    } else {
      if (is_splitted()) {
        distribute_simplex(c);
      } else {
        m_simplex_handles.push_back(c);
      }
    }
    return true;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto insert_simplex(simplex_handle const c) -> bool {
    return insert_simplex(
        c, std::make_index_sequence<mesh_t::num_vertices_per_simplex()>{});
  }
  //----------------------------------------------------------------------------
  auto distribute() {
    if (!m_vertex_handles.empty()) {
      distribute_vertex(m_vertex_handles.front());
      m_vertex_handles.clear();
    }
    if (!m_simplex_handles.empty()) {
      distribute_simplex(m_simplex_handles.front());
      m_simplex_handles.clear();
    }
  }
  //------------------------------------------------------------------------------
  auto construct(vec_t const& min, vec_t const& max, size_t const level,
                 size_t const max_depth) const {
    return std::unique_ptr<this_type>{
        new this_type{min, max, level, max_depth, mesh()}};
  }
  //----------------------------------------------------------------------------
  auto distribute_vertex(vertex_handle const v) {
    for (auto& child : children()) {
      child->insert_vertex(v);
    }
  }
  //----------------------------------------------------------------------------
  auto distribute_simplex(simplex_handle const c) {
    for (auto& child : children()) {
      child->insert_simplex(c);
    }
  }
  //============================================================================
  auto collect_possible_intersections(
      ray<Real, NumDims> const& r,
      std::set<simplex_handle>&               possible_collisions) const -> void {
    if (parent_t::check_intersection(r)) {
      if (is_splitted()) {
        for (auto const& child : children()) {
          child->collect_possible_intersections(r, possible_collisions);
        }
      } else {
        std::copy(begin(m_simplex_handles), end(m_simplex_handles),
                  std::inserter(possible_collisions, end(possible_collisions)));
      }
    }
  }
  //----------------------------------------------------------------------------
  auto collect_possible_intersections(ray<Real, NumDims> const& r) const {
    std::set<simplex_handle> possible_collisions;
    collect_possible_intersections(r, possible_collisions);
    return possible_collisions;
  }
  //----------------------------------------------------------------------------
  auto collect_nearby_simplices(vec<Real, NumDims> const& pos,
                            std::set<simplex_handle>& simplices) const -> void {
    if (is_inside(pos)) {
      if (is_splitted()) {
        for (auto const& child : children()) {
          child->collect_nearby_simplices(pos, simplices);
        }
      } else {
        if (!m_simplex_handles.empty()) {
          std::copy(begin(m_simplex_handles), end(m_simplex_handles),
                    std::inserter(simplices, end(simplices)));
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  auto nearby_simplices(vec<Real, NumDims> const& pos) const {
    std::set<simplex_handle> simplices;
    collect_nearby_simplices(pos, simplices);
    return simplices;
  }
};
//==============================================================================
template <typename T>
struct is_uniform_tree_hierarchy_impl : std::false_type {};
template <typename Mesh>
struct is_uniform_tree_hierarchy_impl<uniform_tree_hierarchy<Mesh>>
    : std::true_type {};
template <typename T>
constexpr auto is_uniform_tree_hierarchy() {
  return is_uniform_tree_hierarchy_impl<std::decay_t<T>>::value;
}
template <typename T>
constexpr auto is_uniform_tree_hierarchy(T&&) {
  return is_uniform_tree_hierarchy<std::decay_t<T>>();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
