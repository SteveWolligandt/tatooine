#ifndef TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_EVALUATOR_H
#define TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_EVALUATOR_H
//==============================================================================
#include <tatooine/flowexplorer/nodes/autonomous_particles_flowmap.h>
#include <tatooine/flowexplorer/nodes/position.h>
#include <tatooine/flowexplorer/point_shader.h>
#include <tatooine/flowexplorer/renderable.h>
#include <yavin/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct autonomous_particles_flowmap_evaluator
    : renderable<autonomous_particles_flowmap_evaluator> {
  yavin::indexeddata<vec3f>     m_gpu_data;
  point_shader                  m_shader;
  position<2>*                  m_x0      = nullptr;
  autonomous_particles_flowmap* m_flowmap = nullptr;
  std::array<GLfloat, 4>        m_color;
  int                           m_point_size = 1;
  bool                          m_is_evaluatable = false;
  //----------------------------------------------------------------------------
  autonomous_particles_flowmap_evaluator(flowexplorer::scene& s)
      : renderable<autonomous_particles_flowmap_evaluator>{
            "Autonomous Particles Flowmap Evaluator", s},
        m_color{0.0f, 0.0f, 0.0f, 1.0f}
  {
    this->template insert_input_pin<autonomous_particles_flowmap>("flowmap");
    this->template insert_input_pin<position<2>>("x0");
    m_gpu_data.vertexbuffer().resize(1);
    m_gpu_data.indexbuffer().push_back(0);
  }
  //----------------------------------------------------------------------------
  virtual ~autonomous_particles_flowmap_evaluator() = default;
  //============================================================================
  auto render(mat4f const& projection_matrix, mat4f const& view_matrix)
      -> void override {
    if (m_is_evaluatable) {
      m_shader.bind();
      m_shader.set_modelview_matrix(view_matrix);
      m_shader.set_projection_matrix(projection_matrix);
      m_shader.set_color(m_color[0], m_color[1], m_color[2], m_color[3]);
      yavin::gl::point_size(m_point_size);
      m_gpu_data.draw_points();
    }
  }
  //----------------------------------------------------------------------------
  bool is_transparent() const override { return m_color[3] < 1; }
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool override {
    bool changed = false;
    changed |= ImGui::SliderInt("point size", &m_point_size, 1, 50);
    changed |= ImGui::ColorEdit4("color", m_color.data());
    return changed;
  }
  //----------------------------------------------------------------------------
  virtual auto on_pin_connected(ui::pin& /*this_pin*/, ui::pin& other_pin)
      -> void override {
    if (other_pin.type() == typeid(position<2>)) {
      m_x0 = dynamic_cast<position<2>*>(&other_pin.node());
    } else if ((other_pin.type() == typeid(autonomous_particles_flowmap))) {
      m_flowmap =
          dynamic_cast<autonomous_particles_flowmap*>(&other_pin.node());
    }
    if (m_x0 != nullptr && m_flowmap != nullptr) {
      evaluate();
    }
  }
  auto on_property_changed() -> void override {
    if (m_x0 != nullptr && m_flowmap != nullptr) {
      evaluate();
    }
  }
  auto evaluate() -> void {
    if (m_flowmap->mesh_available()) {
      auto& flowmap_prop =
          m_flowmap->mesh().vertex_property<vec<double, 2>>("flowmap");
      auto flowmap_sampler_autonomous_particles =
          m_flowmap->mesh().vertex_property_sampler(flowmap_prop);
      try {
        auto const x1 = flowmap_sampler_autonomous_particles(*m_x0);
        m_gpu_data.vertexbuffer()[0] = vec3f{x1(0), x1(1), 0.0f};
        m_is_evaluatable = true;
      } catch (std::runtime_error&) {
        m_is_evaluatable = false;
      }
    }
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::autonomous_particles_flowmap_evaluator,
    TATOOINE_REFLECTION_INSERT_METHOD(point_size, m_point_size),
    TATOOINE_REFLECTION_INSERT_METHOD(color, m_color))
#endif
