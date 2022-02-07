#ifndef TATOOINE_KDTREE_H
#define TATOOINE_KDTREE_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/vtk_legacy.h>

#include <functional>
#include <set>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Mesh>
struct kdtree : aabb<typename Mesh::real_t, Mesh::num_dimensions()> {
  static constexpr auto num_dimensions() { return Mesh::num_dimensions(); }
  using real_t   = typename Mesh::real_t;
  using this_t   = kdtree<Mesh>;
  using parent_type = aabb<real_t, num_dimensions()>;
  using parent_type::center;
  using parent_type::is_inside;
  using parent_type::max;
  using parent_type::min;
  using typename parent_type::vec_t;
  using vertex_handle = typename Mesh::vertex_handle;
  friend class std::unique_ptr<this_t>;
  using parent_type::is_simplex_inside;

 private:
  Mesh const*                            m_mesh;
  size_t                                 m_level;
  size_t                                 m_max_depth;
  std::vector<vertex_handle>             m_vertex_handles;
  std::vector<size_t>                    m_triangle_handles;
  std::array<std::unique_ptr<kdtree>, 2> m_children;
  static constexpr size_t                default_max_depth = 64;

 public:
  explicit kdtree(Mesh const& mesh, size_t const max_depth = default_max_depth)
      : m_mesh{&mesh}, m_level{0}, m_max_depth{max_depth} {
    auto min = vec_t::ones() * std::numeric_limits<real_t>::infinity();
    auto max = -vec_t::ones() * std::numeric_limits<real_t>::infinity();
    for (auto v : mesh.vertices()) {
      for (size_t i = 0; i < num_dimensions(); ++i) {
        min(i) = std::min(min(i), mesh[v](i));
        max(i) = std::max(max(i), mesh[v](i));
      }
    }
    this->min() = min;
    this->max() = max;

    m_vertex_handles.resize(mesh.vertices().size());
    std::iota(begin(m_vertex_handles), end(m_vertex_handles), vertex_handle{0});
    split_if_necessary();
  }
  virtual ~kdtree() = default;

 private:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  kdtree(Mesh const& mesh, vec_t const& min, vec_t const& max,
         size_t const level, size_t const max_depth)
      : parent_type{min, max},
        m_mesh{&mesh},
        m_level{level},
        m_max_depth{max_depth} {}

 public:
  auto mesh() const -> auto const& { return *m_mesh; }
  //------------------------------------------------------------------------------
  auto num_vertex_handles() const { return size(m_vertex_handles); }
  auto num_triangle_handles() const { return size(m_triangle_handles); }
  //------------------------------------------------------------------------------
  //auto insert_vertex(size_t const vertex_idx) -> bool {
  //  if (!is_inside(mesh().vertex_at(vertex_idx))) {
  //    return false;
  //  }
  //  if (is_splitted()) {
  //    if ()
  //  } else {
  //    m_vertex_handles.push_back(vertex_idx);
  //    split_if_necessary();
  //  }
  //  return true;
  //}
  //----------------------------------------------------------------------------
  auto split_if_necessary() -> void{
    if (num_vertex_handles() > 1 && !is_at_max_depth()) {
      split();
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto is_splitted() const { return m_children.front() != nullptr; }
  constexpr auto holds_vertices() const { return !m_vertex_handles.empty(); }
  constexpr auto is_at_max_depth() const { return m_level == m_max_depth; }

 private:
  //----------------------------------------------------------------------------
  auto distribute_vertices(size_t const split_index, real_t const split_pos) {
    for (auto const v : m_vertex_handles) {
      if (mesh()[v](split_index) <= split_pos) {
        m_children[0]->m_vertex_handles.push_back(v);
      }
      if (mesh()[v](split_index) >= split_pos) {
        m_children[1]->m_vertex_handles.push_back(v);
      }
    }
    m_vertex_handles.clear();
  }
  //----------------------------------------------------------------------------
  auto split() {
    auto min0            = this->min();
    auto max0            = this->max();
    auto min1            = this->min();
    auto max1            = this->max();
    auto split_index     = std::numeric_limits<size_t>::max();
    auto split_pos       = std::numeric_limits<real_t>::max();
    auto max_space_range = real_t(0);
    auto dim_positions   = std::vector<real_t>{};

    for (size_t i = 0; i < num_dimensions(); ++i) {
      for (auto const v : m_vertex_handles) {
        dim_positions.push_back(mesh()[v](i));
      }
      std::sort(begin(dim_positions), end(dim_positions));
      if (auto const space = dim_positions.back() - dim_positions.front();
          space > max_space_range) {
        max_space_range = space;
        split_index     = i;

        size_t const i0 = size(dim_positions) / 2;
        split_pos       = (dim_positions[i0] + dim_positions[i0 - 1]) / 2;
      }
      dim_positions.clear();
    }
    assert(split_index != std::numeric_limits<size_t>::max());
    max0(split_index) = split_pos;
    min1(split_index) = split_pos;

    m_children[0] = std::unique_ptr<this_t>(
        new this_t{mesh(), min0, max0, m_level + 1, m_max_depth});
    m_children[1] = std::unique_ptr<this_t>(
        new this_t{mesh(), min1, max1, m_level + 1, m_max_depth});
    distribute_vertices(split_index, split_pos);
    m_children[0]->split_if_necessary();
    m_children[1]->split_if_necessary();
  }
  //----------------------------------------------------------------------------
 public:
  auto write_vtk(filesystem::path const& path) {
    vtk::legacy_file_writer f{path, vtk::dataset_type::polydata};
    f.write_header();
    std::vector<vec<real_t, num_dimensions()>> positions;
    std::vector<std::vector<size_t>>           indices;
    write_vtk_collect_positions_and_indices(positions, indices);
    f.write_points(positions);
    f.write_lines(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  auto write_vtk_collect_positions_and_indices(
      std::vector<vec<real_t, num_dimensions()>>& positions,
      std::vector<std::vector<size_t>>& indices, size_t cur_idx = 0) -> size_t {
    if constexpr (num_dimensions() == 2) {
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
    } else if constexpr (num_dimensions() == 3) {
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
  }
};
template <typename T>
struct is_kdtree_impl : std::false_type {};
template <typename Mesh>
struct is_kdtree_impl<kdtree<Mesh>> : std::true_type {};
template <typename T>
constexpr auto is_kdtree() {
  return is_kdtree_impl<std::decay_t<T>>::value;
}
template <typename T>
constexpr auto is_kdtree(T&&) {
  return is_kdtree<std::decay_t<T>>();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
