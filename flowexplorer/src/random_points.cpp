#include <tatooine/flowexplorer/scene.h>
//
#include <tatooine/geometry/sphere.h>
#include <tatooine/flowexplorer/nodes/random_points.h>
#include <tatooine/flowexplorer/point_shader.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
random_points::random_points(flowexplorer::scene& s)
    : renderable<random_points>{"Random Points", s, *this},
      m_input{insert_input_pin<geometry::sphere<real_t, 3>>("domain")} {}
auto random_points::render(mat4f const& P, mat4f const& V) -> void {
  point_shader::get().bind();
  point_shader::get().set_modelview_matrix(V);
  point_shader::get().set_projection_matrix(P);
  yavin::gl::point_size(5);
  m_points_gpu.draw_points();
}
//------------------------------------------------------------------------------
auto random_points::on_property_changed() -> void { update_points(); }
//------------------------------------------------------------------------------
auto random_points::on_pin_connected(ui::input_pin& /*this_pin*/,
                                     ui::output_pin& /*other_pin*/) -> void {
  update_points();
}
//------------------------------------------------------------------------------
auto random_points::update_points() -> void {
  m_points2d.clear();
  m_points3d.clear();
  m_points_gpu.clear();
  if (m_input.linked_type() == typeid(geometry::sphere<real_t, 3>)) {
    auto const& sphere = m_input.get_linked_as<geometry::sphere<real_t, 3>>();
    m_points3d = sphere.random_points(m_num_points);
    for (int i = 0; i < m_num_points; ++i) {
      auto const& x = m_points3d[i];
      m_points_gpu.vertexbuffer().push_back(vec3f{static_cast<float>(x(0)),
                                                  static_cast<float>(x(1)),
                                                  static_cast<float>(x(2))});
      m_points_gpu.indexbuffer().push_back(i);
    }
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
