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
      v_pin{insert_input_pin<polymorphic::vectorfield<real_t, 3>>("V")},
      w_pin{insert_input_pin<polymorphic::vectorfield<real_t, 3>>("W")},
      grid_pin{insert_input_pin<non_uniform_grid<real_t, 3>>("grid")} {}
//------------------------------------------------------------------------------
auto parallel_vectors::render(mat4f const& P, mat4f const& V) -> void {
  auto &shader = line_shader::get();
  shader.bind();
  shader.set_projection_matrix(P);
  shader.set_modelview_matrix(V);
  shader.set_color(m_line_color[0],
                   m_line_color[1],
                   m_line_color[2],
                   m_line_color[3]);
  gl::line_width(m_line_width);
  m_geometry.draw_lines();
}
//------------------------------------------------------------------------------
auto parallel_vectors::on_pin_connected(ui::input_pin&, ui::output_pin&)
    -> void {
  if (v_pin.is_linked() && w_pin.is_linked() && grid_pin.is_linked()) {
    calculate();
  }
}
//------------------------------------------------------------------------------
auto parallel_vectors::calculate() -> void {
  m_lines = tatooine::parallel_vectors(
      v_pin.get_linked_as<polymorphic::vectorfield<real_t, 3>>(),
      w_pin.get_linked_as<polymorphic::vectorfield<real_t, 3>>(),
      grid_pin.get_linked_as<non_uniform_grid<real_t, 3>>(), 0);
  write_vtk(m_lines, "pv.vtk");

  m_geometry.clear();
  size_t i = 0;
  for (auto const& l : m_lines) {
    for (auto const& v : l.vertices()) {
      m_geometry.vertexbuffer().push_back(vec3f{v(0), v(1), v(2)});
    }
    m_geometry.indexbuffer().push_back(i);
    m_geometry.indexbuffer().push_back(++i);
    for (size_t j = 1; j < l.num_vertices() - 1; ++j) {
      m_geometry.indexbuffer().push_back(i);
      m_geometry.indexbuffer().push_back(++i);
    }
    ++i;
  }
}
//------------------------------------------------------------------------------
auto parallel_vectors::draw_properties() -> bool {
  auto const changed = renderable<parallel_vectors>::draw_properties();
  ImGui::TextUnformatted(std::to_string(size(m_lines)).c_str());
  return changed;
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
