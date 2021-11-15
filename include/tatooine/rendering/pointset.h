#ifndef TATOOINE_RENDERING_POINTSET_H
#define TATOOINE_RENDERING_POINTSET_H
//==============================================================================
#include <tatooine/pointset.h>
#include <tatooine/gl/indexeddata.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename... VertexAttributes>
struct pointset {
 private:
  gl::indexeddata<VertexAttributes...> m_geometry;
 public:
  pointset(std::size_t const num_vertices){
    m_geometry.vertexbuffer().resize(num_vertices);
    m_geometry.indexbuffer().resize(num_vertices);
    {
      auto map = m_geometry.indexbuffer().wmap();
      for (std::size_t i = 0; i < num_vertices; ++i) {
        map[i] = i;
      }
    }
  }
  //----------------------------------------------------------------------------
  auto bind() { m_geometry.bind(); }
  auto draw() { m_geometry.draw_points(); }
  //----------------------------------------------------------------------------
  auto geometry() const -> auto const& { return m_geometry; }
  auto geometry()       -> auto&       { return m_geometry; }
  //----------------------------------------------------------------------------
  auto vertexbuffer() const -> auto const& { return m_geometry.vertexbuffer(); }
  auto vertexbuffer()       -> auto&       { return m_geometry.vertexbuffer(); }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
