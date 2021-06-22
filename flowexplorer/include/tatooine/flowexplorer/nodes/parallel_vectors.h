#ifndef TATOOINE_FLOWEXPLORER_NODES_PARALLEL_VECTORS_H
#define TATOOINE_FLOWEXPLORER_NODES_PARALLEL_VECTORS_H
//==============================================================================
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/line.h>
#include <tatooine/gl/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct parallel_vectors : renderable<parallel_vectors> {
  std::array<GLfloat, 4>       m_line_color;
  int                          m_line_width;
  ui::input_pin&               v_pin;
  ui::input_pin&               w_pin;
  ui::input_pin&               grid_pin;
  std::vector<line<real_t, 3>> m_lines;
  gl::indexeddata<vec3f> m_geometry;
  //============================================================================
  parallel_vectors(flowexplorer::scene& s);
  ~parallel_vectors() = default;
  //============================================================================
  auto calculate() -> void;
  auto render(mat4f const& P, mat4f const& V) -> void override;
  auto on_pin_connected(ui::input_pin&, ui::output_pin&) -> void override;
  auto draw_properties() -> bool override;
  auto is_transparent() const -> bool override { return m_line_color[3] < 255; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::parallel_vectors,
    TATOOINE_REFLECTION_INSERT_METHOD(line_width, m_line_width),
    TATOOINE_REFLECTION_INSERT_METHOD(line_color, m_line_color))
#endif
