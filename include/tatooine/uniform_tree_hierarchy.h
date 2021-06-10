#ifndef TATOOINE_UNIFORM_TREE_HIERARCHY_H
#define TATOOINE_UNIFORM_TREE_HIERARCHY_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/for_loop.h>
#include <tatooine/math.h>

#include <set>
//==============================================================================
namespace tatooine {
//==============================================================================
/// For octrees and quadtrees
template <typename Mesh>
struct uniform_tree_hierarchy
    : aabb<typename Mesh::real_t, Mesh::num_dimensions()> {
  template <std::size_t N>
  struct dim {
    enum e : std::uint8_t { left = 0, right = 1 << N };
  };
  using mesh_t   = Mesh;
  using real_t   = typename mesh_t::real_t;
  using this_t   = uniform_tree_hierarchy<mesh_t>;
  using parent_t = aabb<real_t, mesh_t::num_dimensions()>;
  using parent_t::center;
  using parent_t::is_inside;
  using parent_t::is_simplex_inside;
  using parent_t::max;
  using parent_t::min;
  using typename parent_t::vec_t;
  using vertex_handle = typename mesh_t::vertex_handle;
  using cell_handle   = typename mesh_t::cell_handle;
  friend class std::unique_ptr<this_t>;
  static constexpr auto num_dimensions() { return mesh_t::num_dimensions(); }
  static constexpr auto num_children() {
    return ipow(2, mesh_t::num_dimensions());
  }

  mesh_t const*              m_mesh;
  size_t                     m_level;
  size_t                     m_max_depth;
  std::vector<vertex_handle> m_vertex_handles;
  std::vector<cell_handle>   m_cell_handles;
  std::array<std::unique_ptr<uniform_tree_hierarchy>, num_children()>
                          m_children;
  static constexpr size_t default_max_depth = 2;
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
  explicit uniform_tree_hierarchy(size_t const max_depth = default_max_depth)
      : m_level{1}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit uniform_tree_hierarchy(Mesh const&  mesh,
                                  size_t const max_depth = default_max_depth)
      : parent_t{vec<real_t, num_dimensions()>::zeros(),
                 vec<real_t, num_dimensions()>::zeros()},
        m_mesh{&mesh},
        m_level{1},
        m_max_depth{max_depth} {
    parent_t::operator=(mesh.bounding_box());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  uniform_tree_hierarchy(Mesh const& mesh, vec_t const& min, vec_t const& max,
                         size_t const max_depth = default_max_depth)
      : parent_t{min, max}, m_mesh{&mesh}, m_level{1}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  uniform_tree_hierarchy(Mesh const& mesh, vec_t const& min, vec_t const& max,
                         size_t const level, size_t const max_depth)
      : parent_t{min, max},
        m_mesh{&mesh},
        m_level{level},
        m_max_depth{max_depth} {}
  //============================================================================
 public:
  auto mesh() const -> auto const& { return *m_mesh; }
  //----------------------------------------------------------------------------
  auto num_vertex_handles() const { return size(m_vertex_handles); }
  auto num_cell_handles() const { return size(m_cell_handles); }
  //----------------------------------------------------------------------------
  constexpr auto is_splitted() const { return m_children.front() != nullptr; }
  constexpr auto holds_vertices() const { return !m_vertex_handles.empty(); }
  constexpr auto holds_cells() const { return !m_cell_handles.empty(); }
  constexpr auto is_at_max_depth() const { return m_level == m_max_depth; }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Seq, typename... Is>
  static constexpr auto index(std::index_sequence<Seq...> /*seq*/,
                              Is const... is) {
    return ((is << Seq) + ...);
  }
  //----------------------------------------------------------------------------
 public:
  template <typename... Is, enable_if_integral<Is...> = true>
  static constexpr auto index(Is const... is) {
    static_assert(sizeof...(Is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return index(std::make_index_sequence<num_dimensions()>{}, is...);
  }
  //------------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto child_at(Is const... is) const -> auto const& {
    static_assert(sizeof...(Is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return m_children[index(is...)];
  }
  //------------------------------------------------------------------------------
  template <size_t... Seq>
  auto create_children(std::index_sequence<Seq...> seq) {
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
      m_children[index(seq, is...)] = std::unique_ptr<this_t>(
          new this_t{mesh(), cur_min, cur_max, m_level + 1, m_max_depth});
    };
    for_loop(it, ((void)Seq, 2)...);
  }
  //------------------------------------------------------------------------------
  auto create_children() {
    create_children(std::make_index_sequence<num_dimensions()>{});
  }
  //------------------------------------------------------------------------------
 public:
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
  auto insert_cell(cell_handle const c, std::index_sequence<Is...> /*seq*/)
      -> bool {
    auto const vs = mesh()[c];
    if (!is_simplex_inside(mesh()[std::get<Is>(vs)]...)) {
      return false;
    }
    if (holds_cells()) {
      if (is_at_max_depth()) {
        m_cell_handles.push_back(c);
      } else {
        split_and_distribute();
        distribute_cell(c);
      }
    } else {
      if (is_splitted()) {
        distribute_cell(c);
      } else {
        m_cell_handles.push_back(c);
      }
    }
    return true;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto insert_cell(cell_handle const c) -> bool {
    return insert_cell(
        c, std::make_index_sequence<mesh_t::num_vertices_per_simplex()>{});
  }
  //----------------------------------------------------------------------------
  auto split_and_distribute() {
    create_children();
    if (!m_vertex_handles.empty()) {
      distribute_vertex(m_vertex_handles.front());
      m_vertex_handles.clear();
    }
    if (!m_cell_handles.empty()) {
      distribute_cell(m_cell_handles.front());
      m_cell_handles.clear();
    }
  }
  //----------------------------------------------------------------------------
  auto distribute_vertex(vertex_handle const v) {
    for (auto& child : m_children) {
      child->insert_vertex(v);
    }
  }
  //----------------------------------------------------------------------------
  auto distribute_cell(cell_handle const c) {
    for (auto& child : m_children) {
      child->insert_cell(c);
    }
  }
  //============================================================================
  auto collect_possible_intersections(
      ray<real_t, num_dimensions()> const& r,
      std::set<cell_handle>&               possible_collisions) const -> void {
    if (parent_t::check_intersection(r)) {
      if (is_splitted()) {
        for (auto const& child : m_children) {
          child->collect_possible_intersections(r, possible_collisions);
        }
      } else {
        std::copy(begin(m_cell_handles), end(m_cell_handles),
                  std::inserter(possible_collisions, end(possible_collisions)));
      }
    }
  }
  //----------------------------------------------------------------------------
  auto collect_possible_intersections(ray<real_t, num_dimensions()> const& r) const {
    std::set<cell_handle> possible_collisions;
    collect_possible_intersections(r, possible_collisions);
    return possible_collisions;
  }
  //----------------------------------------------------------------------------
  auto collect_nearby_cells(vec<real_t, num_dimensions()> const& pos,
                            std::set<cell_handle>& cells) const -> void {
    if (is_inside(pos)) {
      if (is_splitted()) {
        for (auto const& child : m_children) {
          child->collect_nearby_cells(pos, cells);
        }
      } else {
        if (!m_cell_handles.empty()) {
          std::copy(begin(m_cell_handles), end(m_cell_handles),
                    std::inserter(cells, end(cells)));
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  auto nearby_cells(vec<real_t, num_dimensions()> const& pos) const {
    std::set<cell_handle> cells;
    collect_nearby_cells(pos, cells);
    return cells;
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
