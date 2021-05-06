#ifndef TATOOINE_CELLTREE_H
#define TATOOINE_CELLTREE_H
//==============================================================================
#include <tatooine/math.h>
#include <tatooine/ray_intersectable.h>
#include <tatooine/utility.h>
#include <tatooine/vec.h>
#include <tatooine/vtk_legacy.h>

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail {
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Celltree, typename Real, size_t NumDimensions,
          size_t NumVerticesPerSimplex>
struct celltree_parent {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Celltree, typename real_t>
struct celltree_parent<Celltree, real_t, 3, 3> : ray_intersectable<real_t> {
  using parent_t = ray_intersectable<real_t>;

  using typename parent_t::optional_intersection_t;
  using typename parent_t::ray_t;
  auto as_celltree() const -> auto const& {
    return *dynamic_cast<Celltree const*>(this);
  }
  auto check_intersection(ray_t const& r, real_t const min_t = 0) const
      -> optional_intersection_t override {
    auto const& c        = as_celltree();
    auto        cur_aabb = c.bounding_box();

    return {};
  }
  //============================================================================
  auto collect_possible_intersections(
      ray<real_t, 3> const& r, size_t const ni,
      axis_aligned_bounding_box<real_t, 3> const& cur_aabb,
      std::vector<size_t>& possible_collisions) const -> void {
    auto const& c        = as_celltree();
    if (c.node(ni).is_leaf()) {
      std::copy(begin(c.cell_handles()) + c.node(ni).type.leaf.start,
                begin(c.cell_handles()) + c.node(ni).type.leaf.start +
                    c.node(ni).type.leaf.size,
                std::back_inserter(possible_collisions));
    } else {
      auto     left_aabb  = cur_aabb;
      left_aabb.max(c.node(ni).dim) = c.node(ni).type.split.left_max;
      auto right_aabb                = cur_aabb;
      right_aabb.min(c.node(ni).dim) = c.node(ni).type.split.right_min;

      if (left_aabb.check_intersection(r)) {
        collect_possible_intersections(r, c.node(ni).left_child_index(),
                                       left_aabb, possible_collisions);
      }
      if (right_aabb.check_intersection(r)) {
        collect_possible_intersections(r, c.node(ni).right_child_index(),
                                       right_aabb, possible_collisions);
      }
    }
  }
  //----------------------------------------------------------------------------
  auto collect_possible_intersections(ray<real_t, 3> const& r) const {
     auto const& c        = as_celltree();
    // auto        cur_aabb = c.bounding_box();
    auto const cur_aabb = axis_aligned_bounding_box<real_t, 3>{c.m_min, c.m_max};
    std::vector<size_t> possible_collisions;
    if (cur_aabb.check_intersection(r)) {
      collect_possible_intersections(r, 0, cur_aabb, possible_collisions);
    }
    return possible_collisions;
  }
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
}  // namespace detail
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh>
struct celltree : detail::celltree_parent<celltree<Mesh>, typename Mesh::real_t,
                                          Mesh::num_dimensions(),
                                          Mesh::num_vertices_per_simplex()> {
  friend struct detail::celltree_parent<celltree<Mesh>, typename Mesh::real_t,
                                        Mesh::num_dimensions(),
                                        Mesh::num_vertices_per_simplex()>;
  using real_t = typename Mesh::real_t;
  static constexpr auto num_dimensions() { return Mesh::num_dimensions(); }
  static constexpr auto num_vertices_per_simplex() {
    return Mesh::num_vertices_per_simplex();
  }

  using vec_t  = vec<real_t, num_dimensions()>;
  using this_t = celltree<Mesh>;
  //============================================================================
  struct node_t {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    struct split_node_t {
      float left_max, right_min;
    };
    struct leaf_node_t {
      std::uint32_t start, size;
    };
    union type_t {
      split_node_t split;
      leaf_node_t  leaf;
    };

    std::uint8_t dim;  // 0, 1, ..., num_dimensions() - 1 for split node,
                       // num_dimensions() for leaf node
    std::size_t m_left_child_index;
    type_t      type;

    constexpr node_t() = default;
    constexpr node_t(node_t const& other) noexcept
        : dim{other.dim}, m_left_child_index{other.m_left_child_index} {
      if (is_leaf()) {
        type.leaf.start = other.type.leaf.start;
        type.leaf.size  = other.type.leaf.size;
      } else {
        type.split.left_max  = other.type.split.left_max;
        type.split.right_min = other.type.split.right_min;
      }
    }
    constexpr auto operator=(node_t const& other) noexcept -> node_t& {
      dim                = other.dim;
      m_left_child_index = other.m_left_child_index;
      if (is_leaf()) {
        type.leaf.start = other.type.leaf.start;
        type.leaf.size  = other.type.leaf.size;
      } else {
        type.split.left_max  = other.type.split.left_max;
        type.split.right_min = other.type.split.right_min;
      }
    }
    constexpr auto is_leaf() const { return dim == num_dimensions(); }
    auto           left_child_index() const {
      // assert(is_leaf());
      return m_left_child_index;
    }
    auto right_child_index() const {
      // assert(is_leaf());
      return m_left_child_index + 1;
    }
  };

  //============================================================================
 private:
  Mesh const*                   m_mesh;
  std::vector<node_t>           m_nodes;
  std::vector<std::size_t>      m_cell_handles;
  vec<real_t, num_dimensions()> m_min, m_max;

  //============================================================================
 public:
  celltree(celltree const&)     = default;
  celltree(celltree&&) noexcept = default;
  auto operator=(celltree const&) -> celltree& = default;
  auto operator=(celltree&&) noexcept -> celltree& = default;
  ~celltree()                                      = default;
  //===========================================================================
  celltree(Mesh const& mesh, vec<real_t, num_dimensions()> const& min,
           vec<real_t, num_dimensions()> const& max)
      : m_mesh{&mesh},
        m_cell_handles(mesh.cells().size()),
        m_min{min},
        m_max{max} {
    std::iota(begin(m_cell_handles), end(m_cell_handles), 0);
    auto& initial_node           = m_nodes.emplace_back();
    initial_node.dim             = num_dimensions();
    initial_node.type.leaf.start = 0;
    initial_node.type.leaf.size  = mesh.cells().size();
    split_if_necessary(0, 0, 128);
  }
  //===========================================================================
 public:
  constexpr auto mesh() const -> auto const& { return *m_mesh; }
  constexpr auto cell_handles() const -> auto const& { return m_cell_handles; }
  constexpr auto node(size_t const i) const -> auto const& {
    return m_nodes[i];
  }
  constexpr auto nodes() const -> auto const& { return m_nodes; }
  constexpr auto indices() const -> auto const& { return m_cell_handles; }

 private:
  constexpr auto mesh() -> auto& { return *m_mesh; }
  constexpr auto cell_handles() -> auto& { return m_cell_handles; }
  constexpr auto node(size_t const i) -> auto& { return m_nodes[i]; }
  constexpr auto nodes() -> auto& { return m_nodes; }
  constexpr auto indices() -> auto& { return m_cell_handles; }
  //===========================================================================
 public:
  template <size_t... Seq>
  constexpr auto min_cell_boundary(size_t const cell_idx, size_t const dim,
                                   std::index_sequence<Seq...> /*seq*/) const {
    auto const cell_vertex_handles = mesh().cell_at(cell_idx);
    return tatooine::min(mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
  }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  constexpr auto max_cell_boundary(size_t const cell_idx, size_t const dim,
                                   std::index_sequence<Seq...> /*seq*/) const {
    auto const cell_vertex_handles = mesh().cell_at(cell_idx);
    return tatooine::max(mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
  }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  constexpr auto cell_center(size_t const cell_idx, size_t const dim,
                             std::index_sequence<Seq...> seq) const {
    return (min_cell_boundary(cell_idx, dim, seq) +
            max_cell_boundary(cell_idx, dim, seq)) /
           2;
  }
  //----------------------------------------------------------------------------
  constexpr auto bounding_box() const { return mesh().bounding_box(); }
  //----------------------------------------------------------------------------
  auto cells_at(vec_t const& x) const {
    std::vector<size_t> cells;
    cells_at(x, 0, cells,
             std::make_index_sequence<num_vertices_per_simplex()>{});
    return cells;
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Seq>
  auto cells_at(vec_t const& x, size_t const cur_node_idx,
                std::vector<size_t>&        cells,
                std::index_sequence<Seq...> seq) const -> void {
    auto const& n = node(cur_node_idx);
    if (n.is_leaf()) {
      auto const vertex_handles = mesh().cell_at(n.type.leaf.start);
      auto       A              = mat<real_t, num_dimensions() + 1,
                   Mesh::num_vertices_per_simplex()>::ones();
      auto       b              = vec<real_t, num_dimensions() + 1>::ones();
      for (size_t i = 0; i < num_dimensions(); ++i) {
        ((A(Seq, i) = mesh()[std::get<Seq>(vertex_handles)](i)), ...);
      }
      for (size_t i = 0; i < num_dimensions(); ++i) {
        b(i) = x(i);
      }
      auto const       barycentric_coordinates = solve(A, b);
      auto             is_inside               = true;
      constexpr real_t eps                     = 1e-6;
      for (size_t i = 0; i < Mesh::num_vertices_per_simplex(); ++i) {
        is_inside &= barycentric_coordinates(0) >= -eps;
        is_inside &= barycentric_coordinates(0) <= 1 + eps;
      }
      if (is_inside) {
        std::copy(begin(cell_handles()) + n.type.leaf.start,
                  begin(cell_handles()) + n.type.leaf.start + n.type.leaf.size,
                  std::back_inserter(cell_handles()));
      }
    } else {
      if (x(n.dim) <= n.type.split.left_max &&
          x(n.dim) < n.type.split.right_min) {
        cells_at(x, n.left_child_index(), cells, seq);
      } else if (x(n.dim) >= n.type.split.right_min &&
                 x(n.dim) > n.type.split.left_max) {
        cells_at(x, n.right_child_index(), cells, seq);
      } else if (x(n.dim) <= n.type.split.left_max &&
                 x(n.dim) >= n.type.split.right_min) {
        // TODO choose best side
        cells_at(x, n.left_child_index(), cells, seq);
        cells_at(x, n.right_child_index(), cells, seq);
      }
    }
  }
  //===========================================================================
  /// \param ni node index
  template <size_t... Seq>
  auto split_dimension(size_t const ni,
                       std::index_sequence<Seq...> /*seq*/) const {
    assert(node(ni).is_leaf());
    auto aabb = make_array<num_dimensions()>(
        std::tuple{std::uint8_t{0}, std::numeric_limits<float>::max(),
                   -std::numeric_limits<float>::max()});
    for (size_t dim = 0; dim < num_dimensions(); ++dim) {
      std::get<0>(aabb[dim]) = dim;
    }

    for (auto cell_it = begin(cell_handles()) + node(ni).type.leaf.start;
         cell_it != begin(cell_handles()) + node(ni).type.leaf.start +
                        node(ni).type.leaf.size;
         ++cell_it) {
      auto const cell                = *cell_it;
      auto const cell_vertex_handles = mesh().cell_at(cell);
      for (auto& [dim, min, max] : aabb) {
        min = tatooine::min(min,
                            mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
        max = tatooine::max(max,
                            mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
      }
    }
    auto split_dim  = std::numeric_limits<std::uint8_t>::max();
    auto max_extent = float(0);

    for (auto const& [dim, min, max] : aabb) {
      auto const extent = max - min;
      if (extent > max_extent) {
        max_extent = extent;
        split_dim  = dim;
      }
    }
    return aabb[split_dim];
  }
  //----------------------------------------------------------------------------
  /// \param ni node index
  auto add_children(size_t const ni) {
    assert(node(ni).is_leaf());
    size_t left_child_index;
    {
      // TODO this is a critical region (race condition)
      left_child_index            = nodes().size();
      node(ni).m_left_child_index = left_child_index;
      nodes().emplace_back();
      nodes().emplace_back();
    }
    return std::pair{left_child_index, left_child_index + 1};
  }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  auto split_if_necessary(size_t const ni, size_t const level,
                          size_t const max_level) {
    if (node(ni).type.leaf.size > 1 && level <= max_level) {
      split(ni, level, max_level);
    }
  }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  auto split(size_t const ni, size_t const level, size_t const max_level) {
    split(ni, level, max_level,
          std::make_index_sequence<num_vertices_per_simplex()>{});
  }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  auto split(size_t const ni, size_t const level,
             size_t const max_level,
             std::index_sequence<Seq...> seq) -> void {
    assert(node(ni).is_leaf());
    auto const [li, ri]              = add_children(ni);
    auto const [split_dim, min, max] = split_dimension(ni, seq);
    node(ni).dim                     = split_dim;

    sort_indices(ni, seq);

    // median-based
    // auto const rmin_cell_index =
    //    node(ni).type.leaf.start + node(ni).type.leaf.size / 2;
    // auto const lmax_cell_index = rmin_cell_index - 1;

    auto          min_cost  = std::numeric_limits<real_t>::max();
    float         best_lmax = 0, best_rmin = 0;
    auto          best_lsize = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t cur_lsize  = 1;

    for (auto cell_it = begin(cell_handles()) + node(ni).type.leaf.start;
         cell_it != begin(cell_handles()) + node(ni).type.leaf.start +
                        node(ni).type.leaf.size - 1;
         ++cell_it) {
      auto const cur_lmax = max_cell_boundary(*cell_it, split_dim, seq);
      auto const cur_rmin = min_cell_boundary(*next(cell_it), split_dim, seq);
      auto const cur_cost = cur_lmax * cur_lsize -
                            cur_rmin * (node(ni).type.leaf.size - cur_lsize);
      if (cur_cost < min_cost) {
        min_cost   = cur_cost;
        best_lmax  = cur_lmax;
        best_rmin  = cur_rmin;
        best_lsize = cur_lsize;
      }
      ++cur_lsize;
    }

    // setup children
    node(li).dim = num_dimensions();
    node(ri).dim = num_dimensions();

    node(li).type.leaf.start = node(ni).type.leaf.start;
    node(ri).type.leaf.start = node(ni).type.leaf.start + best_lsize;

    node(li).type.leaf.size = best_lsize;
    node(ri).type.leaf.size = node(ni).type.leaf.size - best_lsize;

    // calculate lmax and rmin
    node(ni).type.split.left_max  = best_lmax;
    node(ni).type.split.right_min = best_rmin;

    split_if_necessary(li, level + 1, max_level);
    split_if_necessary(ri, level + 1, max_level);
  }
  //----------------------------------------------------------------------------
  /// node must be at an intermediate state. It needs to store the split
  /// dimension but also a range of indices stored in type.leaf
  /// \param ni node index
  template <size_t... Seq>
  auto sort_indices(size_t const ni, std::index_sequence<Seq...> seq) {
    auto comparator = [this, ni, seq](auto const i, auto const j) {
      return cell_center(i, node(ni).dim, seq) <
             cell_center(j, node(ni).dim, seq);
    };
    std::sort(begin(cell_handles()) + node(ni).type.leaf.start,
              begin(cell_handles()) + node(ni).type.leaf.start +
                  node(ni).type.leaf.size,
              comparator);
  }
};
template <typename T>
struct is_celltree_impl : std::false_type {};
template <typename Mesh>
struct is_celltree_impl<celltree<Mesh>> : std::true_type {};
template <typename T>
constexpr auto is_celltree() {
  return is_celltree_impl<std::decay_t<T>>::value;
}
template <typename T>
constexpr auto is_celltree(T&&) {
  return is_celltree<std::decay_t<T>>();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
