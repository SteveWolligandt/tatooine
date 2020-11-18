#ifndef TATOOINE_QUADTREE_H
#define TATOOINE_QUADTREE_H
//==============================================================================
#include <functional>
#include <tatooine/axis_aligned_bounding_box.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct quadtree : aabb<Real, 2> {
  enum class dim0 : std::uint8_t { left = 0, right = 1 };
  enum class dim1 : std::uint8_t { bottom = 0, top = 2 };
  using this_t   = quadtree<Real>;
  using parent_t = aabb<Real, 2>;
  using parent_t::center;
  using parent_t::is_inside;
  using parent_t::is_triangle_inside;
  using parent_t::max;
  using parent_t::min;
  using typename parent_t::vec_t;
  friend class std::unique_ptr<this_t>;

 private:
  size_t                                   m_level;
  size_t                                   m_max_depth;
  std::vector<size_t>                      m_vertex_indices;
  std::vector<size_t>                      m_triangle_indices;
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
  auto num_vertex_indices() const { return size(m_vertex_indices); }
  auto num_triangle_indices() const { return size(m_triangle_indices); }
  //------------------------------------------------------------------------------
  template <typename Mesh>
  auto insert_vertex(Mesh const& mesh, size_t const vertex_idx) -> bool {
    if (!is_inside(mesh.vertex_at(vertex_idx))) {
      return false;
    }
    if (holds_vertices()) {
      if (is_at_max_depth()) {
        m_vertex_indices.push_back(vertex_idx);
      } else {
        split_and_distribute(mesh);
        distribute_vertex(mesh, vertex_idx);
      }
    } else {
      if (is_splitted()) {
        distribute_vertex(mesh, vertex_idx);
      } else {
        m_vertex_indices.push_back(vertex_idx);
      }
    }
    return true;
  }
  //------------------------------------------------------------------------------
  template <typename TriangularMesh>
  auto insert_face(TriangularMesh const& mesh, size_t const triangle_idx)
      -> bool {
    auto [vi0, vi1, vi2] = mesh.triangle_at(triangle_idx);
    if (!is_triangle_inside(mesh[vi0], mesh[vi1], mesh[vi2])) {
      return false;
    }
    if (holds_triangles()) {
      if (is_at_max_depth()) {
        m_triangle_indices.push_back(triangle_idx);
      } else {
        split_and_distribute(mesh);
        distribute_triangle(mesh, triangle_idx);
      }
    } else {
      if (is_splitted()) {
        distribute_triangle(mesh, triangle_idx);
      } else {
        m_triangle_indices.push_back(triangle_idx);
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto nearby_triangles(vec_t const& x) const -> auto const& {
    if (auto tris = nearby_triangles_ptr(x); tris != nullptr) {
      return *tris;
    }
    throw std::runtime_error{"[quadtree::triangle_candidatas] out of domain"};
  }
  //----------------------------------------------------------------------------
  auto nearby_triangles_ptr(vec_t const& x) const
      -> std::vector<size_t> const* {
    if (!is_inside(x)) {
      return nullptr;
    }
    if (is_splitted()) {
      for (auto const& child : m_children) {
        if (auto tris = child->nearby_triangles_ptr(x); tris != nullptr) {
          return tris;
        }
      }
    }
    return &m_triangle_indices;
  }
  //----------------------------------------------------------------------------
  constexpr auto is_splitted() const { return m_children.front() != nullptr; }
  constexpr auto holds_vertices() const { return !m_vertex_indices.empty(); }
  constexpr auto holds_triangles() const { return !m_triangle_indices.empty(); }
  constexpr auto is_at_max_depth() const { return m_level == m_max_depth; }
  //----------------------------------------------------------------------------
  static constexpr auto index(dim0 const d0, dim1 const d1) {
    return static_cast<std::uint8_t>(d0) + static_cast<std::uint8_t>(d1);
  }
  //----------------------------------------------------------------------------
  static constexpr auto bottom_left_index() {
    return index(dim0::left, dim1::bottom);
  }
  static constexpr auto bottom_right_index() {
    return index(dim0::right, dim1::bottom);
  }
  static constexpr auto top_left_index() {
    return index(dim0::left, dim1::top);
  }
  static constexpr auto top_right_index() {
    return index(dim0::right, dim1::top);
  }
  //----------------------------------------------------------------------------
  auto bottom_left() const -> auto const& {
    return m_children[bottom_left_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto bottom_left() -> auto& { return m_children[bottom_left_index()]; }
  //----------------------------------------------------------------------------
  auto bottom_right() const -> auto const& {
    return m_children[bottom_right_index];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto bottom_right() -> auto& { return m_children[bottom_right_index()]; }
  //----------------------------------------------------------------------------
  auto top_left() const -> auto const& { return m_children[top_left_index()]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto top_left() -> auto& { return m_children[top_left_index()]; }
  //----------------------------------------------------------------------------
  auto top_right() const -> auto const& {
    return m_children[top_right_index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto top_right() -> auto& { return m_children[top_right_index()]; }

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
      child->insert_face(mesh, triangle_idx);
    }
  }
  //----------------------------------------------------------------------------
  template <typename Mesh>
  auto split_and_distribute(Mesh const& mesh) {
    create_children();
    if (!m_vertex_indices.empty()) {
      distribute_vertex(mesh, m_vertex_indices.front());
      m_vertex_indices.clear();
    }
    if (!m_triangle_indices.empty()) {
      distribute_triangle(mesh, m_triangle_indices.front());
      m_triangle_indices.clear();
    }
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
