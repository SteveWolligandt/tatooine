#ifndef TATOOINE_QUADTREE_H
#define TATOOINE_QUADTREE_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
struct quadtree : boundingbox<T, 2> {
 private:
  size_t              m_level;
  size_t              m_max_depth;
  std::vector<size_t> m_vertex_indices;
  std::unique_ptr<quadtree> m_bottom_left;
  std::unique_ptr<quadtree> m_bottom_right;
  std::unique_ptr<quadtree> m_top_left;
  std::unique_ptr<quadtree> m_top_right;

 public:
  quadtree(size_t const max_depth = std::numeric_limits<size_t>::max()): m_level{0}, m_max_depth{max_depth}{}
 private:
 public:
  template <typename PointSet>
  void insert_vertex(PointSet const& ps, size_t const vertex_idx) {
    
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
