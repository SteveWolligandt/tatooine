#include <tatooine/flowexplorer/scene.h>

#include <tatooine/flowexplorer/nodes/autonomous_particles_flowmap_evaluator.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
std::vector<std::string> const autonomous_particles_flowmap_evaluator::items{
    "piecewise linear", "inverse distance weighting", "moving least squares"};
//----------------------------------------------------------------------------
autonomous_particles_flowmap_evaluator::autonomous_particles_flowmap_evaluator(flowexplorer::scene& s)
      : renderable<autonomous_particles_flowmap_evaluator>{
            "Autonomous Particles Flowmap Evaluator", s},
        m_color{0.0f, 0.0f, 0.0f, 1.0f}
  {
  this->template insert_input_pin<autonomous_particles_flowmap>("flowmap");
  this->template insert_input_pin<position<2>>("x0");
  m_gpu_data.vertexbuffer().resize(1);
  m_gpu_data.indexbuffer().push_back(0);
}
//============================================================================
auto autonomous_particles_flowmap_evaluator::render(
    mat4f const& projection_matrix, mat4f const& view_matrix) -> void {
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
auto autonomous_particles_flowmap_evaluator::is_transparent() const -> bool {
  return m_color[3] < 1;
}
//----------------------------------------------------------------------------
auto autonomous_particles_flowmap_evaluator::draw_properties() -> bool {
  bool changed = false;

  if (ImGui::BeginCombo("##combo", items[m_current_item].c_str())) {
    size_t i = 0;
    for (auto const& item : items) {
      bool const is_selected = (m_current_item == i);
      if (ImGui::Selectable(item.c_str(), is_selected)) {
        m_current_item = i;
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
      ++i;
    }
    ImGui::EndCombo();
  }
  changed |= ImGui::SliderInt("point size", &m_point_size, 1, 50);
  changed |= ImGui::ColorEdit4("color", m_color.data());
  return changed;
}
//----------------------------------------------------------------------------
auto autonomous_particles_flowmap_evaluator::on_pin_connected(
    ui::input_pin& /*this_pin*/, ui::output_pin& other_pin) -> void {
  if (other_pin.type() == typeid(position<2>)) {
    m_x0 = dynamic_cast<position<2>*>(&other_pin.node());
  } else if ((other_pin.type() == typeid(autonomous_particles_flowmap))) {
    m_flowmap = dynamic_cast<autonomous_particles_flowmap*>(&other_pin.node());
  }
  if (m_x0 != nullptr && m_flowmap != nullptr) {
    evaluate();
  }
}
//----------------------------------------------------------------------------
auto autonomous_particles_flowmap_evaluator::on_property_changed() -> void {
  if (m_x0 != nullptr && m_flowmap != nullptr) {
    evaluate();
  }
}
//----------------------------------------------------------------------------
auto autonomous_particles_flowmap_evaluator::evaluate() -> void {
  if (m_flowmap->data_available()) {
    if (m_current_item == 0) {
      auto& flowmap_prop = m_flowmap->mesh().vertex_property<vec2>("flowmap");
      auto  flowmap_sampler_autonomous_particles =
          m_flowmap->mesh().sampler(flowmap_prop);
      try {
        auto const x1 = flowmap_sampler_autonomous_particles(*m_x0);
        m_gpu_data.vertexbuffer()[0] = vec3f{x1(0), x1(1), 0.0f};
        m_is_evaluatable             = true;
      } catch (std::runtime_error&) {
        m_is_evaluatable = false;
      }
    } else if (m_current_item == 1) {
      auto& flowmap_prop = m_flowmap->mesh().vertex_property<vec2>("flowmap");
      auto  flowmap_sampler_autonomous_particles =
          m_flowmap->mesh().inverse_distance_weighting_sampler(flowmap_prop);
      try {
        auto const x1 = flowmap_sampler_autonomous_particles(*m_x0);
        m_gpu_data.vertexbuffer()[0] = vec3f{x1(0), x1(1), 0.0f};
        m_is_evaluatable             = true;
      } catch (std::runtime_error&) {
        m_is_evaluatable = false;
      }
    } else if (m_current_item == 2) {
      auto& flowmap_prop = m_flowmap->mesh().vertex_property<vec2>("flowmap");
      auto  flowmap_sampler_autonomous_particles =
          m_flowmap->mesh().moving_least_squares_sampler(flowmap_prop, 0.1);
      try {
        auto const x1 = flowmap_sampler_autonomous_particles(*m_x0);
        m_gpu_data.vertexbuffer()[0] = vec3f{x1(0), x1(1), 0.0f};
        m_is_evaluatable             = true;
      } catch (std::runtime_error&) {
        m_is_evaluatable = false;
      }
    }
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
