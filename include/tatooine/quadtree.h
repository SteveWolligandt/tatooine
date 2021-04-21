#ifndef TATOOINE_QUADTREE_H
#define TATOOINE_QUADTREE_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/vtk_legacy.h>

#include <functional>
#include <set>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct quadtree : aabb<Real, 2> {
  enum class dim_x : std::uint8_t { left = 0, right = 1 };
  enum class dim_y : std::uint8_t { bottom = 0, top = 2 };
  using this_t   = quadtree<Real>;
  using parent_t = aabb<Real, 2>;
  using parent_t::center;
  using parent_t::is_triangle_inside;
  using parent_t::is_inside;
  using parent_t::max;
  using parent_t::min;
  using typename parent_t::vec_t;
  friend class std::unique_ptr<this_t>;

 private:
  size_t                                   m_level;
  size_t                                   m_max_depth;
  std::vector<size_t>                      m_vertex_handles;
  std::vector<size_t>                      m_triangle_handles;
  std::array<std::unique_ptr<quadtree>, 4> m_children;
  static constexpr size_t                  default_max_depth = 10;

 public:
  quadtree(vec_t const& min, vec_t const& max,
           size_t const max_depth = default_max_depth)
      : parent_t{min, max}, m_level{0}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit quadtree(size_t const max_depth = default_max_depth)
      : m_level{0}, m_max_depth{max_depth} {}
  virtual ~quadtree() = default;

 private:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  quadtree(vec_t const& min, vec_t const& max, size_t const level,
           size_t const max_depth)
      : parent_t{min, max}, m_level{level}, m_max_depth{max_depth} {}

 public:
  auto num_vertex_handles() const { return size(m_vertex_handles); }
  auto num_triangle_handles() const { return size(m_triangle_handles); }
  //------------------------------------------------------------------------------
  template <typename Mesh>
  auto insert_vertex(Mesh const& mesh, size_t const vertex_idx) -> bool {
    if (!is_inside(mesh.vertex_at(vertex_idx))) {
      return false;
    }
    if (holds_vertices()) {
      if (is_at_max_depth()) {
        m_vertex_handles.push_back(vertex_idx);
      } else {
        split_and_distribute(mesh);
        distribute_vertex(mesh, vertex_idx);
      }
    } else {
      if (is_splitted()) {
        distribute_vertex(mesh, vertex_idx);
      } else {
        m_vertex_handles.push_back(vertex_idx);
      }
    }
    return true;
  }
  //------------------------------------------------------------------------------
  template <typename TriangularMesh>
  auto insert_triangle(TriangularMesh const& mesh, size_t const triangle_idx) -> bool {
    auto [vi0, vi1, vi2] = mesh.triangle_at(triangle_idx);
    if (!is_triangle_inside(mesh[vi0], mesh[vi1], mesh[vi2])) {
      return false;
    }
    if (holds_triangles()) {
      if (is_at_max_depth()) {
        m_triangle_handles.push_back(triangle_idx);
      } else {
        split_and_distribute(mesh);
        distribute_triangle(mesh, triangle_idx);
      }
    } else {
      if (is_splitted()) {
        distribute_triangle(mesh, triangle_idx);
      } else {
        m_triangle_handles.push_back(triangle_idx);
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto nearby_triangles(vec_t const& x) const -> std::set<size_t> {
    std::set<size_t> collector;
    nearby_triangles(x, collector);
    return collector;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto nearby_triangles(vec_t const& x, std::set<size_t>& collector) const
      -> void {
    if (!is_inside(x)) {
      return;
    }
    if (is_splitted()) {
      for (auto const& child : m_children) {
        child->nearby_triangles(x, collector);
      }
    } else {
      std::copy(begin(m_triangle_handles), end(m_triangle_handles),
                std::inserter(collector, end(collector)));
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto is_splitted() const { return m_children.front() != nullptr; }
  constexpr auto holds_vertices() const { return !m_vertex_handles.empty(); }
  constexpr auto holds_triangles() const { return !m_triangle_handles.empty(); }
  constexpr auto is_at_max_depth() const { return m_level == m_max_depth; }
  //----------------------------------------------------------------------------
  static constexpr auto index(dim_x const d0, dim_y const d1) {
    return static_cast<std::uint8_t>(d0) + static_cast<std::uint8_t>(d1);
  }
  //----------------------------------------------------------------------------
  static constexpr auto left_bottom_index() {
    return index(dim_x::left, dim_y::bottom);
  }
  static constexpr auto right_bottom_index() {
    return index(dim_x::right, dim_y::bottom);
  }
  static constexpr auto left_top_index() {
    return index(dim_x::left, dim_y::top);
  }
  static constexpr auto right_top_index() {
    return index(dim_x::right, dim_y::top);
  }
  //----------------------------------------------------------------------------
  auto bottom_left() const -> auto const& {
    return m_children[left_bottom_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto bottom_left() -> auto& { return m_children[left_bottom_index()]; }
  //----------------------------------------------------------------------------
  auto bottom_right() const -> auto const& {
    return m_children[right_bottom_index];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto bottom_right() -> auto& { return m_children[right_bottom_index()]; }
  //----------------------------------------------------------------------------
  auto top_left() const -> auto const& { return m_children[left_top_index()]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto top_left() -> auto& { return m_children[left_top_index()]; }
  //----------------------------------------------------------------------------
  auto top_right() const -> auto const& {
    return m_children[right_top_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto top_right() -> auto& { return m_children[right_top_index()]; }

 private:
  //----------------------------------------------------------------------------
  auto create_bottom_left() {
    bottom_left() = std::unique_ptr<this_t>(
        new this_t{min(), center(), m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_bottom_right() {
    bottom_right() = std::unique_ptr<this_t>(
        new this_t{vec_t{center(0), min(1)}, vec_t{max(0), center(1)},
                   m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_top_left() {
    top_left() = std::unique_ptr<this_t>(new this_t{vec_t{min(0), center(1)},
                                                    vec_t{center(0), max(1)},
                                                    m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_top_right() {
    top_right() = std::unique_ptr<this_t>(
        new this_t{center(), max(), m_level + 1, m_max_depth});
  }
  auto create_children() {
    create_bottom_left();
    create_bottom_right();
    create_top_left();
    create_top_right();
  }
  //----------------------------------------------------------------------------
  template <typename Mesh>
  auto distribute_vertex(Mesh const& mesh, size_t const vertex_idx) {
    for (auto& child : m_children) {
      child->insert_vertex(mesh, vertex_idx);
    }
  }
  //----------------------------------------------------------------------------
  template <typename Mesh>
  auto distribute_triangle(Mesh const& mesh, size_t const triangle_idx) {
    for (auto& child : m_children) {
      child->insert_triangle(mesh, triangle_idx);
    }
  }
  //----------------------------------------------------------------------------
  template <typename Mesh>
  auto split_and_distribute(Mesh const& mesh) {
    create_children();
    if (!m_vertex_handles.empty()) {
      distribute_vertex(mesh, m_vertex_handles.front());
      m_vertex_handles.clear();
    }
    if (!m_triangle_handles.empty()) {
      distribute_triangle(mesh, m_triangle_handles.front());
      m_triangle_handles.clear();
    }
  }
  //----------------------------------------------------------------------------
 public:
  auto write_vtk(filesystem::path const& path) {
    vtk::legacy_file_writer f{path, vtk::dataset_type::polydata};
    f.write_header();
    std::vector<vec<Real, 2>>        positions;
    std::vector<std::vector<size_t>> indices;
    write_vtk_collect_positions_and_indices(positions, indices);
    f.write_points(positions);
    f.write_lines(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  auto write_vtk_collect_positions_and_indices(
      std::vector<vec<Real, 2>>&        positions,
      std::vector<std::vector<size_t>>& indices, size_t cur_idx = 0)
      -> size_t {
    positions.push_back(vec{min(0), min(1)});
    positions.push_back(vec{max(0), min(1)});
    positions.push_back(vec{max(0), max(1)});
    positions.push_back(vec{min(0), max(1)});
    indices.push_back(
        {cur_idx, cur_idx + 1, cur_idx + 2, cur_idx + 3, cur_idx});
    cur_idx += 4;
    if (is_splitted()) {
      for (auto& child : m_children) {
        cur_idx = child->write_vtk_collect_positions_and_indices(
            positions, indices, cur_idx);
      }
    }
    return cur_idx;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
