#ifndef TATOOINE_CELLTREE_H
#define TATOOINE_CELLTREE_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
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
template <typename Celltree, typename real_type>
struct celltree_parent<Celltree, real_type, 3, 3>
    : ray_intersectable<real_type, 3> {
  using parent_type = ray_intersectable<real_type, 3>;

  using typename parent_type::optional_intersection_type;
  using typename parent_type::ray_type;
  auto as_celltree() const -> auto const& {
    return *dynamic_cast<Celltree const*>(this);
  }
  auto check_intersection(ray_type const& /*r*/,
                          real_type const /*min_t*/ = 0) const
      -> optional_intersection_type override {
    auto const& c        = as_celltree();
    auto        cur_aabb = c.axis_aligned_bounding_box();

    return {};
  }
  //============================================================================
  auto collect_possible_intersections(
      ray<real_type, 3> const& r, size_t const ni,
      tatooine::axis_aligned_bounding_box<real_type, 3> const& cur_aabb,
      std::vector<size_t>& possible_collisions) const -> void {
    auto const& c = as_celltree();
    auto const& n = c.node(ni);
    if (n.is_leaf()) {
      auto const begin_it = begin(c.cell_handles()) + n.as_leaf().start;
      auto const end_it   = begin_it + n.as_leaf().size;

      std::copy(begin_it, end_it, std::back_inserter(possible_collisions));
    } else {
      {
        auto sub_aabb       = cur_aabb;
        sub_aabb.min(n.dim) = n.as_split().right_min;
        if (sub_aabb.check_intersection(r)) {
          collect_possible_intersections(r, n.right_child_index(), sub_aabb,
                                         possible_collisions);
        }
      }
      {
        auto sub_aabb       = cur_aabb;
        sub_aabb.max(n.dim) = n.as_split().left_max;
        if (sub_aabb.check_intersection(r)) {
          collect_possible_intersections(r, n.left_child_index(), sub_aabb,
                                         possible_collisions);
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  auto collect_possible_intersections(ray<real_type, 3> const& r) const {
    auto const&         c        = as_celltree();
    auto const          cur_aabb = tatooine::axis_aligned_bounding_box{c.m_min, c.m_max};
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
struct celltree
    : detail::celltree_parent<celltree<Mesh>, typename Mesh::real_type,
                              Mesh::num_dimensions(),
                              Mesh::num_vertices_per_simplex()> {
  friend struct detail::celltree_parent<
      celltree<Mesh>, typename Mesh::real_type, Mesh::num_dimensions(),
      Mesh::num_vertices_per_simplex()>;
  using real_type = typename Mesh::real_type;
  static constexpr auto num_dimensions() { return Mesh::num_dimensions(); }
  static constexpr auto num_vertices_per_simplex() {
    return Mesh::num_vertices_per_simplex();
  }

  using vec_t     = vec<real_type, num_dimensions()>;
  using this_type = celltree<Mesh>;
  //============================================================================
  struct node_type {
    using float_type = double;
    using index_type = std::uint32_t;
    // static_assert(sizeof(float_type) == sizeof(index_type));
    struct split_node_t {
      float_type left_max, right_min;
    };
    struct leaf_node_type {
      index_type start, size;
    };
    union type_t {
      split_node_t   split;
      leaf_node_type leaf;
    };

    std::uint8_t dim;  // 0, 1, ..., num_dimensions() - 1 for split node,
                       // num_dimensions() for leaf node
   private:
    std::size_t m_left_child_index;

   public:
    type_t type;

   public:
    constexpr node_type() = default;
    constexpr node_type(node_type const& other) noexcept
        : dim{other.dim}, m_left_child_index{other.m_left_child_index} {
      if (is_leaf()) {
        as_leaf().start = other.as_leaf().start;
        as_leaf().size  = other.as_leaf().size;
      } else {
        as_split().left_max  = other.as_split().left_max;
        as_split().right_min = other.as_split().right_min;
      }
    }
    constexpr auto operator=(node_type const& other) noexcept -> node_type& {
      dim                = other.dim;
      m_left_child_index = other.m_left_child_index;
      if (is_leaf()) {
        as_leaf().start = other.as_leaf().start;
        as_leaf().size  = other.as_leaf().size;
      } else {
        as_split().left_max  = other.as_split().left_max;
        as_split().right_min = other.as_split().right_min;
      }
    }
    constexpr auto is_leaf() const { return dim == num_dimensions(); }
    auto           left_child_index() const {
      assert(!is_leaf());
      return m_left_child_index;
    }
    auto right_child_index() const {
      assert(!is_leaf());
      return m_left_child_index + 1;
    }
    auto set_left_child_index(std::size_t const i) { m_left_child_index = i; }
    auto as_leaf() -> auto& {
      assert(is_leaf());
      return type.leaf;
    }
    auto as_leaf() const -> auto const& {
      assert(is_leaf());
      return type.leaf;
    }
    auto as_split() -> auto& {
      assert(!is_leaf());
      return type.split;
    }
    auto as_split() const -> auto const& {
      assert(!is_leaf());
      return type.split;
    }
  };

  //============================================================================
 private:
  Mesh const*                      m_mesh;
  std::vector<node_type>           m_nodes;
  std::vector<std::size_t>         m_cell_handles;
  vec<real_type, num_dimensions()> m_min, m_max;

  //============================================================================
 public:
  celltree(celltree const&)     = default;
  celltree(celltree&&) noexcept = default;
  auto operator=(celltree const&) -> celltree& = default;
  auto operator=(celltree&&) noexcept -> celltree& = default;
  ~celltree()                                      = default;
  //===========================================================================
  celltree(Mesh const& mesh)
      : m_mesh{&mesh}, m_cell_handles(mesh.simplices().size()) {
    auto aabb = mesh.axis_aligned_bounding_box();
    m_min     = aabb.min();
    m_max     = aabb.max();
    std::iota(begin(m_cell_handles), end(m_cell_handles), 0);
    auto& initial_node           = m_nodes.emplace_back();
    initial_node.dim             = num_dimensions();
    initial_node.as_leaf().start = 0;
    initial_node.as_leaf().size  = m_cell_handles.size();
    split_if_necessary(0, 1, 2);
  }
  //---------------------------------------------------------------------------
  celltree(Mesh const& mesh, vec<real_type, num_dimensions()> const& min,
           vec<real_type, num_dimensions()> const& max)
      : m_mesh{&mesh},
        m_cell_handles(mesh.simplices().size()),
        m_min{min},
        m_max{max} {
    std::iota(begin(m_cell_handles), end(m_cell_handles), 0);
    auto& initial_node           = m_nodes.emplace_back();
    initial_node.dim             = num_dimensions();
    initial_node.as_leaf().start = 0;
    initial_node.as_leaf().size  = m_cell_handles.size();
    split_if_necessary(0, 1, 2);
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
  constexpr auto min_cell_boundary(size_t const       cell_idx,
                                   std::uint8_t const dim,
                                   std::index_sequence<Seq...> /*seq*/) const {
    auto const cell_vertex_handles = mesh().simplex_at(cell_idx);
    return tatooine::min(mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
  }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  constexpr auto max_cell_boundary(size_t const       cell_idx,
                                   std::uint8_t const dim,
                                   std::index_sequence<Seq...> /*seq*/) const {
    auto const cell_vertex_handles = mesh().simplex_at(cell_idx);
    return tatooine::max(mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
  }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  constexpr auto cell_center(size_t const cell_idx, std::uint8_t const dim,
                             std::index_sequence<Seq...> /*seq*/) const {
    auto const cell_vertex_handles = mesh().simplex_at(cell_idx);
    auto const min =
        tatooine::min(mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
    auto const max =
        tatooine::max(mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
    return (min + max) / 2;
  }
  //----------------------------------------------------------------------------
  constexpr auto axis_aligned_bounding_box() const {
    return mesh().axis_aligned_bounding_box();
  }
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
      auto const vertex_handles = mesh().simplex_at(n.as_leaf().start);
      auto       A              = mat<real_type, num_dimensions() + 1,
                   Mesh::num_vertices_per_simplex()>::ones();
      auto       b              = vec<real_type, num_dimensions() + 1>::ones();
      for (size_t i = 0; i < num_dimensions(); ++i) {
        ((A(Seq, i) = mesh()[std::get<Seq>(vertex_handles)](i)), ...);
      }
      for (size_t i = 0; i < num_dimensions(); ++i) {
        b(i) = x(i);
      }
      auto const          barycentric_coordinates = *solve(A, b);
      auto                is_inside               = true;
      constexpr real_type eps                     = 1e-6;
      for (size_t i = 0; i < Mesh::num_vertices_per_simplex(); ++i) {
        is_inside &= barycentric_coordinates(0) >= -eps;
        is_inside &= barycentric_coordinates(0) <= 1 + eps;
      }
      if (is_inside) {
        std::copy(begin(cell_handles()) + n.as_leaf().start,
                  begin(cell_handles()) + n.as_leaf().start + n.as_leaf().size,
                  std::back_inserter(cells));
      }
    } else {
      if (x(n.dim) <= n.as_split().left_max &&
          x(n.dim) < n.as_split().right_min) {
        cells_at(x, n.left_child_index(), cells, seq);
      } else if (x(n.dim) >= n.as_split().right_min &&
                 x(n.dim) > n.as_split().left_max) {
        cells_at(x, n.right_child_index(), cells, seq);
      } else if (x(n.dim) <= n.as_split().left_max &&
                 x(n.dim) >= n.as_split().right_min) {
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
    auto aabb = make_array<num_dimensions()>(std::tuple{
        std::uint8_t{0},
        std::numeric_limits<typename node_type::float_type>::max(),
        -std::numeric_limits<typename node_type::float_type>::max()});
    for (size_t dim = 0; dim < num_dimensions(); ++dim) {
      std::get<0>(aabb[dim]) = dim;
    }
    auto const begin_it = begin(cell_handles()) + node(ni).as_leaf().start;
    auto const end_it   = begin_it + node(ni).as_leaf().size;
    for (auto cell_it = begin_it; cell_it != end_it; ++cell_it) {
      auto const cell_vertex_handles = mesh().simplex_at(*cell_it);
      for (auto& [dim, min, max] : aabb) {
        min = tatooine::min(min,
                            mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
        max = tatooine::max(max,
                            mesh()[std::get<Seq>(cell_vertex_handles)](dim)...);
      }
    }
    auto split_dim  = std::numeric_limits<std::uint8_t>::max();
    auto max_extent = typename node_type::float_type(0);

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
      left_child_index = nodes().size();
      node(ni).set_left_child_index(left_child_index);
      nodes().emplace_back();
      nodes().emplace_back();
    }
    return left_child_index;
  }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  auto split_if_necessary(size_t const ni, size_t const level,
                          size_t const max_level) {
    if (node(ni).as_leaf().size > 1 && level < max_level) {
      std::cout << "splitting node at index " << ni << '\n';
      split(ni, level, max_level);
    } else {
      std::cout << "leaf at level " << level << "[" << node(ni).as_leaf().start
                << ", " << node(ni).as_leaf().size << "]" << '\n';
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
  auto split(size_t const ni, size_t const level, size_t const max_level,
             std::index_sequence<Seq...> seq) -> void {
    assert(node(ni).is_leaf());
    auto const li = add_children(ni);
    auto const ri = li + 1;
    // std::cout <<level << ", " <<  li << ", " << ri << '\n';
    auto const [split_dim, min, max] = split_dimension(ni, seq);

    // split_with_heuristic(ni, li, ri, split_dim, min, max, seq);
    split_with_median(ni, li, ri, split_dim, seq);

    split_if_necessary(li, level + 1, max_level);
    split_if_necessary(ri, level + 1, max_level);
  }
  //----------------------------------------------------------------------------
  template <size_t... Seq>
  auto split_with_median(size_t const ni, size_t const li, size_t const ri,
                         std::uint8_t const          split_dim,
                         std::index_sequence<Seq...> seq) {
    sort_indices(ni, split_dim, seq);

    auto const cur_start = node(ni).as_leaf().start;
    auto const cur_size  = node(ni).as_leaf().size;
    auto const lstart    = cur_start;
    auto const lsize     = cur_size / 2;
    auto const rstart    = lstart + lsize;
    auto const rsize     = cur_size - lsize;
    auto const lmax      = max_cell_boundary(rstart - 1, split_dim, seq);
    auto const rmin      = min_cell_boundary(rstart, split_dim, seq);
    assert(lsize + rsize == cur_size);
    assert(rstart + rsize == cur_start + cur_size);

    node(li).dim             = num_dimensions();
    node(li).as_leaf().start = lstart;
    node(li).as_leaf().size  = lsize;

    node(ri).dim             = num_dimensions();
    node(ri).as_leaf().start = rstart;
    node(ri).as_leaf().size  = rsize;

    node(ni).dim                  = split_dim;
    node(ni).as_split().left_max  = lmax;
    node(ni).as_split().right_min = rmin;
  }
  //----------------------------------------------------------------------------
  /// TODO heuristic not working correctly
  template <size_t... Seq>
  auto split_with_heuristic(size_t const ni, size_t const li, size_t const ri,
                            std::uint8_t const split_dim, real_type const min,
                            real_type const             max,
                            std::index_sequence<Seq...> seq) {
    sort_indices(ni, split_dim, seq);

    auto min_cost   = std::numeric_limits<real_type>::max();
    auto best_lsize = std::numeric_limits<std::uint32_t>::max();
    auto cur_lsize  = std::uint32_t(1);
    typename node_type::float_type best_lmax = 0, best_rmin = 0;
    auto const start_it = begin(cell_handles()) + node(ni).as_leaf().start;
    auto const end_it   = start_it + node(ni).as_leaf().size - 1;
    for (auto cell_it = start_it; cell_it != end_it; ++cell_it) {
      auto const cur_lmax = max_cell_boundary(*cell_it, split_dim, seq);
      auto const cur_rmin = min_cell_boundary(*next(cell_it), split_dim, seq);
      auto const cur_cost =
          (cur_lmax - min) * cur_lsize -
          (max - cur_rmin) * (node(ni).as_leaf().size - cur_lsize);
      // auto const cur_cost = -2 * cur_lsize + node(ni).as_leaf().size;
      if (cur_cost < min_cost) {
        min_cost   = cur_cost;
        best_lmax  = cur_lmax;
        best_rmin  = cur_rmin;
        best_lsize = cur_lsize;
      }
      ++cur_lsize;
    }

    node(li).dim             = num_dimensions();
    node(li).as_leaf().start = node(ni).as_leaf().start;
    node(li).as_leaf().size  = best_lsize;

    node(ri).dim             = num_dimensions();
    node(ri).as_leaf().start = node(ni).as_leaf().start + best_lsize;
    node(ri).as_leaf().size  = node(ni).as_leaf().size - best_lsize;

    node(ni).dim                  = split_dim;
    node(ni).as_split().left_max  = best_lmax;
    node(ni).as_split().right_min = best_rmin;
  }
  //----------------------------------------------------------------------------
  /// \param ni node index
  template <size_t... Seq>
  auto sort_indices(size_t const ni, std::uint8_t const dim,
                    std::index_sequence<Seq...> seq) {
    auto comparator = [this, ni, dim, seq](auto const i, auto const j) {
      return cell_center(i, dim, seq) < cell_center(j, dim, seq);
    };
    auto const begin_it = begin(cell_handles()) + node(ni).as_leaf().start;
    auto const end_it   = begin_it + node(ni).as_leaf().size;
    std::sort(begin_it, end_it, comparator);
  }
  //----------------------------------------------------------------------------
 public:
  auto write_vtk(filesystem::path const& path) {
    vtk::legacy_file_writer f{path, vtk::dataset_type::polydata};
    f.write_header();
    std::vector<vec<real_type, 3>>   positions;
    std::vector<std::vector<size_t>> indices;
    auto const parent_bounding_box = tatooine::axis_aligned_bounding_box{m_min, m_max};
    write_vtk_collect_positions_and_indices(positions, indices, 0,
                                            parent_bounding_box);
    f.write_points(positions);
    f.write_lines(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  auto write_vtk_collect_positions_and_indices(
      std::vector<vec<real_type, num_dimensions()>>& positions,
      std::vector<std::vector<size_t>>& indices, size_t cur_node_idx,
      tatooine::axis_aligned_bounding_box<real_type, num_dimensions()> const&
             aabb,
      size_t cur_level = 0, size_t cur_idx = 0) -> size_t {
    if (node(cur_node_idx).is_leaf()) {
      positions.push_back(vec{aabb.min(0), aabb.min(1), aabb.min(2)});
      positions.push_back(vec{aabb.max(0), aabb.min(1), aabb.min(2)});
      positions.push_back(vec{aabb.max(0), aabb.max(1), aabb.min(2)});
      positions.push_back(vec{aabb.min(0), aabb.max(1), aabb.min(2)});
      positions.push_back(vec{aabb.min(0), aabb.min(1), aabb.max(2)});
      positions.push_back(vec{aabb.max(0), aabb.min(1), aabb.max(2)});
      positions.push_back(vec{aabb.max(0), aabb.max(1), aabb.max(2)});
      positions.push_back(vec{aabb.min(0), aabb.max(1), aabb.max(2)});
      indices.push_back(
          {cur_idx, cur_idx + 1, cur_idx + 2, cur_idx + 3, cur_idx});
      indices.push_back(
          {cur_idx + 4, cur_idx + 5, cur_idx + 6, cur_idx + 7, cur_idx + 4});
      indices.push_back({cur_idx, cur_idx + 4});
      indices.push_back({cur_idx + 1, cur_idx + 5});
      indices.push_back({cur_idx + 2, cur_idx + 6});
      indices.push_back({cur_idx + 3, cur_idx + 7});
      cur_idx += 8;
    } else {
      auto sub_aabb = aabb;
      sub_aabb.max(node(cur_node_idx).dim) =
          node(cur_node_idx).as_split().left_max;
      cur_idx = write_vtk_collect_positions_and_indices(
          positions, indices, node(cur_node_idx).left_child_index(), sub_aabb,
          cur_level + 1, cur_idx);

      sub_aabb.max(node(cur_node_idx).dim) = aabb.max(node(cur_node_idx).dim);
      sub_aabb.min(node(cur_node_idx).dim) =
          node(cur_node_idx).as_split().right_min;
      cur_idx = write_vtk_collect_positions_and_indices(
          positions, indices, node(cur_node_idx).right_child_index(), sub_aabb,
          cur_level + 1, cur_idx);
    }
    return cur_idx;
  }
};

template <typename Mesh>
celltree(Mesh const& mesh) -> celltree<Mesh>;
template <typename Mesh, typename Real, std::size_t NumDimensions>
celltree(Mesh const& mesh, vec<Real, NumDimensions> const& min,
         vec<Real, NumDimensions> const& max) -> celltree<Mesh>;
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
