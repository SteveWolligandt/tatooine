#ifndef TATOOINE_OCTREE_H
#define TATOOINE_OCTREE_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/has_tetrahedron_at_method.h>
#include <tatooine/has_face_at_method.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Mesh>
struct octree : aabb<typename Mesh::real_t, 3> {
  enum class dim_x : std::uint8_t { left = 0, right = 1 };
  enum class dim_y : std::uint8_t { bottom = 0, top = 2 };
  enum class dim_z : std::uint8_t { front = 0, back = 4 };
  using this_t   = octree<Mesh>;
  using real_t = typename Mesh::real_t;
  using parent_t = aabb<real_t, 3>;
  using parent_t::center;
  using parent_t::is_inside;
  using parent_t::is_tetrahedron_inside;
  using parent_t::is_triangle_inside;
  using parent_t::max;
  using parent_t::min;
  using typename parent_t::vec_t;
  friend class std::unique_ptr<this_t>;

  Mesh const*                            m_mesh;
  size_t                                 m_level;
  size_t                                 m_max_depth;
  std::vector<size_t>                    m_vertex_handles;
  std::vector<size_t>                    m_triangle_handles;
  std::vector<size_t>                    m_tet_handles;
  std::array<std::unique_ptr<octree>, 8> m_children;
  static constexpr size_t                default_max_depth = 6;
  //============================================================================
  octree()                  = default;
  octree(octree const&)     = default;
  octree(octree&&) noexcept = default;
  auto operator=(octree const&) -> octree& = default;
  auto operator=(octree&&) noexcept -> octree& = default;
  virtual ~octree()                            = default;
  //----------------------------------------------------------------------------
  explicit octree(size_t const max_depth = default_max_depth)
      : m_level{0}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  octree(Mesh const& mesh, vec_t const& min, vec_t const& max,
         size_t const max_depth = default_max_depth)
      : parent_t{min, max}, m_mesh{&mesh}, m_level{0}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  octree(Mesh const& mesh, vec_t const& min, vec_t const& max,
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
  auto left_bottom_back() -> auto& {
    return m_children[left_bottom_back_index()];
  }
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
    left_bottom_front() = std::unique_ptr<this_t>(new this_t{mesh(),
        vec_t{min(0), min(1), min(2)}, vec_t{center(0), center(1), center(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_right_bottom_front() {
    right_bottom_front() = std::unique_ptr<this_t>(new this_t{mesh(),
        vec_t{center(0), min(1), min(2)}, vec_t{max(0), center(1), center(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_left_top_front() {
    left_top_front() = std::unique_ptr<this_t>(new this_t{mesh(),
        vec_t{min(0), center(1), min(2)}, vec_t{center(0), max(1), center(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_right_top_front() {
    right_top_front() = std::unique_ptr<this_t>(
        new this_t{mesh(), vec_t{center(0), center(1), min(2)},
                   vec_t{max(0), max(1), center(2)}, m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_left_bottom_back() {
    left_bottom_back() = std::unique_ptr<this_t>(new this_t{mesh(), 
        vec_t{min(0), min(1), center(2)}, vec_t{center(0), center(1), max(2)},
        m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_right_bottom_back() {
    right_bottom_back() = std::unique_ptr<this_t>(
        new this_t{mesh(), vec_t{center(0), min(1), center(2)},
                   vec_t{max(0), center(1), max(2)}, m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_left_top_back() {
    left_top_back() = std::unique_ptr<this_t>(
        new this_t{mesh(), vec_t{min(0), center(1), center(2)},
                   vec_t{center(0), max(1), max(2)}, m_level + 1, m_max_depth});
  }
  //----------------------------------------------------------------------------
  auto create_right_top_back() {
    right_top_back() = std::unique_ptr<this_t>(
        new this_t{mesh(), vec_t{center(0), center(1), center(2)},
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
  auto insert_vertex(size_t const vertex_idx) -> bool {
    if (!is_inside(mesh().vertex_at(vertex_idx))) {
      return false;
    }
    if (holds_vertices()) {
      if (is_at_max_depth()) {
        m_vertex_handles.push_back(vertex_idx);
      } else {
        split_and_distribute();
        distribute_vertex(vertex_idx);
      }
    } else {
      if (is_splitted()) {
        distribute_vertex(vertex_idx);
      } else {
        m_vertex_handles.push_back(vertex_idx);
      }
    }
    return true;
  }
  //------------------------------------------------------------------------------
  //template <typename TriangularMesh                             = Mesh,
  //          enable_if<has_face_at_method<TriangularMesh>()> = true>
  auto insert_triangle(size_t const triangle_idx) -> bool {
    auto const [vi0, vi1, vi2] = mesh().cell_at(triangle_idx);
    if (!is_triangle_inside(mesh()[vi0], mesh()[vi1], mesh()[vi2])) {
      return false;
    }
    if (holds_triangles()) {
      if (is_at_max_depth()) {
        m_triangle_handles.push_back(triangle_idx);
      } else {
        split_and_distribute();
        distribute_triangle(triangle_idx);
      }
    } else {
      if (is_splitted()) {
        distribute_triangle(triangle_idx);
      } else {
        m_triangle_handles.push_back(triangle_idx);
      }
    }
    return true;
  }
  //------------------------------------------------------------------------------
  template <typename TetrahedalMesh                                = Mesh,
            enable_if<has_tetrahedron_at_method<TetrahedalMesh>()> = true>
  auto insert_tetrahedron(size_t const tetrahedron_idx) -> bool {
    auto const [vi0, vi1, vi2, vi3] = mesh().tetrahedron_at(tetrahedron_idx);
    if (!is_tetrahedron_inside(mesh()[vi0], mesh()[vi1], mesh()[vi2],
                               mesh()[vi3])) {
      return false;
    }
    if (holds_tetrahedrons()) {
      if (is_at_max_depth()) {
        m_tet_handles.push_back(tetrahedron_idx);
      } else {
        split_and_distribute();
        distribute_tetrahedron(tetrahedron_idx);
      }
    } else {
      if (is_splitted()) {
        distribute_tetrahedron(tetrahedron_idx);
      } else {
        m_tet_handles.push_back(tetrahedron_idx);
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto split_and_distribute() {
    create_children();
    if (!m_vertex_handles.empty()) {
      distribute_vertex(m_vertex_handles.front());
      m_vertex_handles.clear();
    }
    if constexpr (has_face_at_method<Mesh>()) {
      if (!m_triangle_handles.empty()) {
        distribute_triangle(m_triangle_handles.front());
        m_triangle_handles.clear();
      }
    }
    if constexpr (has_tetrahedron_at_method<Mesh>()) {
      if (!m_tet_handles.empty()) {
        distribute_tetrahedron(m_tet_handles.front());
        m_tet_handles.clear();
      }
    }
  }
  //----------------------------------------------------------------------------
  auto distribute_vertex(size_t const vertex_idx) {
    for (auto& child : m_children) {
      child->insert_vertex(vertex_idx);
    }
  }
  //----------------------------------------------------------------------------
  //template <typename TriangularMesh                             = Mesh,
  //          enable_if<has_face_at_method<TriangularMesh>()> = true>
  auto distribute_triangle(size_t const triangle_idx) {
    for (auto& child : m_children) {
      child->insert_triangle(triangle_idx);
    }
  }
  //----------------------------------------------------------------------------
  template <typename TetrahedalMesh                                = Mesh,
            enable_if<has_tetrahedron_at_method<TetrahedalMesh>()> = true>
  auto distribute_tetrahedron(size_t const tet_idx) {
    for (auto& child : m_children) {
      child->insert_tetrahedron(tet_idx);
    }
  }
  //----------------------------------------------------------------------------
  auto nearby_tetrahedrons(vec_t const& x) const -> std::set<size_t> {
    std::set<size_t> collector;
    nearby_tetrahedrons(x, collector);
    return collector;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto nearby_tetrahedrons(vec_t const& x, std::set<size_t>& collector) const
      -> void {
    if (!is_inside(x)) {
      return;
    }
    if (is_splitted()) {
      for (auto const& child : m_children) {
        child->nearby_tetrahedrons(x, collector);
      }
    } else {
      std::copy(begin(m_tet_handles), end(m_tet_handles),
                std::inserter(collector, end(collector)));
    }
  }
  //============================================================================
  auto collect_possible_intersections(
      ray<real_t, 3> const& r, std::set<size_t>& possible_collisions) const
      -> void {
    if (parent_t::check_intersection(r)) {
      if (is_splitted()) {
        for (auto const& child : m_children) {
          child->collect_possible_intersections(r, possible_collisions);
        }
      } else {
        std::copy(begin(m_triangle_handles), end(m_triangle_handles),
                  std::inserter(possible_collisions, end(possible_collisions)));
      }
    }
  }
  //----------------------------------------------------------------------------
  auto collect_possible_intersections(ray<real_t, 3> const& r) const {
    std::set<size_t> possible_collisions;
    collect_possible_intersections(r, possible_collisions);
    return possible_collisions;
  }
  //----------------------------------------------------------------------------
  //auto check_intersection(ray<real_t, 3> const& r, real_t const min_t = 0) const
  //    -> std::optional<intersection<real_t, 3>> override {
  //  std::set<size_t> possible_collisions;
  //  collect_possible_intersections(r, possible_collisions);
  //
  //}
  //----------------------------------------------------------------------------
 public:
  auto write_vtk(filesystem::path const& path) {
    vtk::legacy_file_writer f{path, vtk::dataset_type::polydata};
    f.write_header();
    std::vector<vec<real_t, 3>>        positions;
    std::vector<std::vector<size_t>> indices;
    write_vtk_collect_positions_and_indices(positions, indices);
    f.write_points(positions);
    f.write_lines(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  auto write_vtk_collect_positions_and_indices(
      std::vector<vec<real_t, 3>>&        positions,
      std::vector<std::vector<size_t>>& indices, size_t cur_idx = 0)
      -> size_t {
    positions.push_back(vec{min(0), min(1), min(2)});
    positions.push_back(vec{max(0), min(1), min(2)});
    positions.push_back(vec{max(0), max(1), min(2)});
    positions.push_back(vec{min(0), max(1), min(2)});
    positions.push_back(vec{min(0), min(1), max(2)});
    positions.push_back(vec{max(0), min(1), max(2)});
    positions.push_back(vec{max(0), max(1), max(2)});
    positions.push_back(vec{min(0), max(1), max(2)});
    indices.push_back(
        {cur_idx, cur_idx + 1, cur_idx + 2, cur_idx + 3, cur_idx});
    indices.push_back(
        {cur_idx + 4, cur_idx + 5, cur_idx + 6, cur_idx + 7, cur_idx + 4});
    indices.push_back({cur_idx, cur_idx + 4});
    indices.push_back({cur_idx + 1, cur_idx + 5});
    indices.push_back({cur_idx + 2, cur_idx + 6});
    indices.push_back({cur_idx + 3, cur_idx + 7});
    cur_idx += 8;
    if (is_splitted()) {
      for (auto& child : m_children) {
        cur_idx = child->write_vtk_collect_positions_and_indices(
            positions, indices, cur_idx);
      }
    }
    return cur_idx;
  }
};
template <typename T>
struct is_octree_impl : std::false_type {};
template <typename Mesh>
struct is_octree_impl<octree<Mesh>> : std::true_type {};
template <typename T>
constexpr auto is_octree() {
  return is_octree_impl<std::decay_t<T>>::value;
}
template <typename T>
constexpr auto is_octree(T&&) {
  return is_octree<std::decay_t<T>>();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
