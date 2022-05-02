#ifndef TATOOINE_RENDERING_LINE_LOOP_H
#define TATOOINE_RENDERING_LINE_LOOP_H
//==============================================================================
#include <tatooine/gl/indexeddata.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename... VertexAttributes>
struct line_loop {
 private:
   gl::indexeddata<VertexAttributes...> m_geometry;

 public:
  explicit line_loop(std::size_t const size) {
    m_geometry.vertexbuffer().resize(size);
    m_geometry.indexbuffer().resize(size);

    {
      auto map = m_geometry.indexbuffer().wmap();
      for (std::size_t i = 0; i < size; ++i) {
        map[i] = i;
      }
    }
  }
  //----------------------------------------------------------------------------
  auto bind() { m_geometry.bind(); }
  auto draw() { m_geometry.draw_line_loop(); }
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
