#ifndef TATOOINE_OCTREE_H
#define TATOOINE_OCTREE_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/has_triangle_at_method.h>
#include <tatooine/has_tetrahedron_at_method.h>
//==============================================================================
namespace tatooine{
//==============================================================================
template <typename Real>
struct octree : aabb<Real, 3> {
  enum class dim_x : std::uint8_t { left = 0, right = 1 };
  enum class dim_y : std::uint8_t { bottom = 0, top = 2 };
  enum class dim_z : std::uint8_t { front = 0, back = 4 };
  using this_t   = octree<Real>;
  using parent_t = aabb<Real, 3>;
  using parent_t::center;
  using parent_t::is_inside;
  using parent_t::is_triangle_inside;
  using parent_t::is_tetrahedron_inside;
  using parent_t::max;
  using parent_t::min;
  using typename parent_t::vec_t;
  friend class std::unique_ptr<this_t>;

  size_t                                   m_level;
  size_t                                   m_max_depth;
  std::vector<size_t>                      m_vertex_handles;
  std::vector<size_t>                      m_triangle_handles;
  std::vector<size_t>                      m_tet_handles;
  std::array<std::unique_ptr<octree>, 8>   m_children;
  static constexpr size_t                  default_max_depth = 10;
  //============================================================================
  octree()                                     = default;
  octree(octree const&)                        = default;
  octree(octree&&) noexcept                    = default;
  auto operator=(octree const&) -> octree&     = default;
  auto operator=(octree&&) noexcept -> octree& = default;
  virtual ~octree()                            = default;
  //----------------------------------------------------------------------------
  explicit octree(size_t const max_depth = default_max_depth)
      : m_level{0}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  octree(vec_t const& min, vec_t const& max,
         size_t const max_depth = default_max_depth)
      : parent_t{min, max}, m_level{0}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  octree(vec_t const& min, vec_t const& max, size_t const level,
         size_t const max_depth)
      : parent_t{min, max}, m_level{level}, m_max_depth{max_depth} {}
  //============================================================================
 public:
  auto num_vertex_handles() const { return size(m_vertex_handles); }
  auto num_triangle_handles() const { return size(m_triangle_handles); }
  auto num_tetrahedron_handles() const { return size(m_tet_handles); }
  //----------------------------------------------------------------------------
  constexpr auto is_splitted() const { return m_children.front() != nullptr; }
  constexpr auto holds_vertices() const { return !m_vertex_handles.empty(); }
  constexpr auto holds_triangles() const { return !m_triangle_handles.empty(); }
  constexpr auto holds_tetrahedrons() const { return !m_tet_handles.empty(); }
  constexpr auto is_at_max_depth() const { return m_level == m_max_depth; }
  //----------------------------------------------------------------------------
  static constexpr auto index(dim_x const d0, dim_y const d1, dim_z const d2) {
    return static_cast<std::uint8_t>(d0) + static_cast<std::uint8_t>(d1) +
           static_cast<std::uint8_t>(d2);
  }
  //----------------------------------------------------------------------------
  static constexpr auto left_bottom_front_index() {
    return index(dim_x::left, dim_y::bottom, dim_z::front);
  }
  static constexpr auto right_bottom_front_index() {
    return index(dim_x::right, dim_y::bottom, dim_z::front);
  }
  static constexpr auto left_top_front_index() {
    return index(dim_x::left, dim_y::top, dim_z::front);
  }
  static constexpr auto right_top_front_index() {
    return index(dim_x::right, dim_y::top, dim_z::front);
  }
  static constexpr auto left_bottom_back_index() {
    return index(dim_x::left, dim_y::bottom, dim_z::back);
  }
  static constexpr auto right_bottom_back_index() {
    return index(dim_x::right, dim_y::bottom, dim_z::back);
  }
  static constexpr auto left_top_back_index() {
    return index(dim_x::left, dim_y::top, dim_z::back);
  }
  static constexpr auto right_top_back_index() {
    return index(dim_x::right, dim_y::top, dim_z::back);
  }
  //----------------------------------------------------------------------------
  auto left_bottom_front() const -> auto const& {
    return m_children[left_bottom_front_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto left_bottom_front() -> auto& {
    return m_children[left_bottom_front_index()];
  }
  //----------------------------------------------------------------------------
  auto right_bottom_front() const -> auto const& {
    return m_children[right_bottom_front_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto right_bottom_front() -> auto& {
    return m_children[right_bottom_front_index()];
  }
  //----------------------------------------------------------------------------
  auto left_top_front() const -> auto const& {
    return m_children[left_top_front_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto left_top_front() -> auto& { return m_children[left_top_front_index()]; }
  //----------------------------------------------------------------------------
  auto right_top_front() const -> auto const& {
    return m_children[right_top_front_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto right_top_front() -> auto& {
    return m_children[right_top_front_index()];
  }
  //----------------------------------------------------------------------------
  auto left_bottom_back() const -> auto const& {
    return m_children[left_bottom_back_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto left_bottom_back() -> auto& { return m_children[left_bottom_back_index()]; }
  //----------------------------------------------------------------------------
  auto right_bottom_back() const -> auto const& {
    return m_children[right_bottom_back_index];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto right_bottom_back() -> auto& {
    return m_children[right_bottom_back_index()];
  }
  //----------------------------------------------------------------------------
  auto left_top_back() const -> auto const& {
    return m_children[left_top_back_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto left_top_back() -> auto& { return m_children[left_top_back_index()]; }
  //----------------------------------------------------------------------------
  auto right_top_back() const -> auto const& {
    return m_children[right_top_back_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto right_top_back() -> auto& { return m_children[right_top_back_index()]; }

 private:
  //----------------------------------------------------------------------------
  auto create_left_bottom_front() {
    left_bottom_front() = std::unique_ptr<this_t>(new this_t{
        vec_t{min(0), min(1), min(2)}, vec_t{center(0), center(1), center(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_right_bottom_front() {
    right_bottom_front() = std::unique_ptr<this_t>(new this_t{
        vec_t{center(0), min(1), min(2)}, vec_t{max(0), center(1), center(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_left_top_front() {
    left_top_front() = std::unique_ptr<this_t>(new this_t{
        vec_t{min(0), center(1), min(2)}, vec_t{center(0), max(1), center(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_right_top_front() {
    right_top_front() = std::unique_ptr<this_t>(
        new this_t{vec_t{center(0), center(1), min(2)},
                   vec_t{max(0), max(1), center(2)}, m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_left_bottom_back() {
    left_bottom_back() = std::unique_ptr<this_t>(new this_t{
        vec_t{min(0), min(1), center(2)}, vec_t{center(0), center(1), max(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_right_bottom_back() {
    right_bottom_back() = std::unique_ptr<this_t>(new this_t{
        vec_t{center(0), min(1), center(2)}, vec_t{max(0), center(1), max(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_left_top_back() {
    left_top_back() = std::unique_ptr<this_t>(new this_t{
        vec_t{min(0), center(1), center(2)}, vec_t{center(0), max(1), max(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_right_top_back() {
    right_top_back() = std::unique_ptr<this_t>(
        new this_t{vec_t{center(0), center(1), center(2)},
                   vec_t{max(0), max(1), max(2)}, m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_children() {
    create_left_bottom_front();
    create_right_bottom_front();
    create_left_top_front();
    create_right_top_front();
    create_left_bottom_back();
    create_right_bottom_back();
    create_left_top_back();
    create_right_top_back();
  }
  //------------------------------------------------------------------------------
 public:
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
  template <typename TriangularMesh,
            enable_if<has_triangle_at_method<TriangularMesh>()> = true>
  auto insert_triangle(TriangularMesh const& mesh, size_t const triangle_idx)
      -> bool {
    auto const [vi0, vi1, vi2] = mesh.triangle_at(triangle_idx);
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
  //------------------------------------------------------------------------------
  template <typename TetrahedalMesh,
            enable_if<has_tetrahedron_at_method<TetrahedalMesh>()> = true>
  auto insert_tetrahedron(TetrahedalMesh const& mesh,
                          size_t const          tetrahedron_idx) -> bool {
    auto const [vi0, vi1, vi2, vi3] = mesh.tetrahedron_at(tetrahedron_idx);
    if (!is_tetrahedron_inside(mesh[vi0], mesh[vi1], mesh[vi2], mesh[vi3])) {
      return false;
    }
    if (holds_tetrahedrons()) {
      if (is_at_max_depth()) {
        m_tet_handles.push_back(tetrahedron_idx);
      } else {
        split_and_distribute(mesh);
        distribute_tetrahedron(mesh, tetrahedron_idx);
      }
    } else {
      if (is_splitted()) {
        distribute_tetrahedron(mesh, tetrahedron_idx);
      } else {
        m_tet_handles.push_back(tetrahedron_idx);
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  template <typename Mesh>
  auto split_and_distribute(Mesh const& mesh) {
    create_children();
    if (!m_vertex_handles.empty()) {
      distribute_vertex(mesh, m_vertex_handles.front());
      m_vertex_handles.clear();
    }
    if constexpr (has_triangle_at_method<Mesh>()) {
      if (!m_triangle_handles.empty()) {
        distribute_triangle(mesh, m_triangle_handles.front());
        m_triangle_handles.clear();
      }
    }
    if constexpr (has_tetrahedron_at_method<Mesh>()) {
      if (!m_tet_handles.empty()) {
        distribute_tetrahedron(mesh, m_tet_handles.front());
        m_tet_handles.clear();
      }
    }
  }
  //----------------------------------------------------------------------------
  template <typename Mesh>
  auto distribute_vertex(Mesh const& mesh, size_t const vertex_idx) {
    for (auto& child : m_children) {
      child->insert_vertex(mesh, vertex_idx);
    }
  }
  //----------------------------------------------------------------------------
  template <typename TriangularMesh,
            enable_if<has_triangle_at_method<TriangularMesh>()> = true>
  auto distribute_triangle(TriangularMesh const& mesh,
                           size_t const          triangle_idx) {
    for (auto& child : m_children) {
      child->insert_triangle(mesh, triangle_idx);
    }
  }
  //----------------------------------------------------------------------------
  template <typename TetrahedalMesh,
            enable_if<has_tetrahedron_at_method<TetrahedalMesh>()> = true>
  auto distribute_tetrahedron(TetrahedalMesh const& mesh,
                              size_t const          tet_idx) {
    for (auto& child : m_children) {
      child->insert_tetrahedron(mesh, tet_idx);
    }
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
