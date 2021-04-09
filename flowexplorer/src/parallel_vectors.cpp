#include <tatooine/flowexplorer/scene.h>
//
#include <tatooine/flowexplorer/nodes/parallel_vectors.h>
#include <tatooine/parallel_vectors.h>
#include <tatooine/flowexplorer/line_shader.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
parallel_vectors::parallel_vectors(flowexplorer::scene& s)
    : renderable<parallel_vectors>{"Parallel Vectors", s},
      v_pin{insert_input_pin<parent::vectorfield<real_t, 3>>("V")},
      w_pin{insert_input_pin<parent::vectorfield<real_t, 3>>("W")},
      grid_pin{insert_input_pin<non_uniform_grid<real_t, 3>>("grid")} {}
//------------------------------------------------------------------------------
auto parallel_vectors::render(mat4f const& P, mat4f const& V) -> void {
  line_shader::get().bind();
  line_shader::get().set_projection_matrix(P);
  line_shader::get().set_modelview_matrix(V);
  m_geometry.draw_lines();
}
//------------------------------------------------------------------------------
auto parallel_vectors::on_pin_connected(ui::input_pin&, ui::output_pin&)
    -> void {
  if (v_pin.is_linked() && w_pin.is_linked() && grid_pin.is_linked()) {
    m_lines = tatooine::parallel_vectors(
        v_pin.get_linked_as<parent::vectorfield<real_t, 3>>(),
        w_pin.get_linked_as<parent::vectorfield<real_t, 3>>(),
        grid_pin.get_linked_as<non_uniform_grid<real_t, 3>>(), 0);

    m_geometry.clear();
    size_t i = 0;
    for (auto const& l : m_lines) {
      for (auto const& v : l.vertices()) {
        m_geometry.vertexbuffer().push_back(vec3f{v});
      }
      for (auto const& v : l.vertices()) {
        m_geometry.indexbuffer().push_back(i);
        ++i;
        m_geometry.indexbuffer().push_back(i);
      }
      ++i;
    }
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
