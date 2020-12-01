#ifndef TATOOINE_OCTREE_H
#define TATOOINE_OCTREE_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
//==============================================================================
namespace tatooine{
//==============================================================================
template <typename Real>
struct octree : aabb<Real, 3> {
  enum class dim0 : std::uint8_t { left = 0, right = 1 };
  enum class dim1 : std::uint8_t { bottom = 0, top = 2 };
  enum class dim2 : std::uint8_t { front = 0, back = 4 };
  using this_t   = octree<Real>;
  using parent_t = aabb<Real, 3>;
  using parent_t::center;
  using parent_t::is_inside;
  using parent_t::is_triangle_inside;
  using parent_t::max;
  using parent_t::min;
  using typename parent_t::vec_t;
  friend class std::unique_ptr<this_t>;

  size_t                                   m_level;
  size_t                                   m_max_depth;
  std::vector<size_t>                      m_vertex_indices;
  std::vector<size_t>                      m_triangle_indices;
  std::array<std::unique_ptr<octree>, 8> m_children;
  static constexpr size_t                  default_max_depth = 10;
  //============================================================================
  octree()                                     = default;
  octree(octree const&)                        = default;
  octree(octree&&) noexcept                    = default;
  auto operator=(octree const&) -> octree&     = default;
  auto operator=(octree&&) noexcept -> octree& = default;
  virtual ~octree()                            = default;
  //----------------------------------------------------------------------------
  octree(vec_t const& min, vec_t const& max,
         size_t const max_depth = default_max_depth)
      : parent_t{min, max}, m_level{0}, m_max_depth{max_depth} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit octree(size_t const max_depth = default_max_depth)
      : m_level{0}, m_max_depth{max_depth} {}
  //----------------------------------------------------------------------------
  template <typename Mesh>
  auto insert_vertex(Mesh const& /*mesh*/, size_t const /*vertex_idx*/)
      -> bool {
    return true;
  }
  //----------------------------------------------------------------------------
  template <typename Mesh>
  auto insert_face(Mesh const& /*mesh*/, size_t const /*triangle_idx*/)
      -> bool {
    return true;
  }
};
//==============================================================================
}
//==============================================================================
#endif
