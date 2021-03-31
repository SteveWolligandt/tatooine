#include <tatooine/flowexplorer/scene.h>
//
#include <tatooine/flowexplorer/nodes/scalar.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes{
//==============================================================================
scalar::scalar(flowexplorer::scene& s)
    : ui::node<scalar>{"Scalar", s, m_value} {}
//==============================================================================
auto scalar::update(std::chrono::duration<double> const& dt) -> void {
  if (m_vary) {
    m_internal_value += dt.count() * m_speed;
    if (m_variation_type == 0) {
      m_value = m_internal_value;
    } else if (m_variation_type == 1) {
      m_value = std::sin(m_internal_value);
    } else if (m_variation_type == 2) {
      m_value = std::cos(m_internal_value);
    }

    for (auto l : self_pin().links()) {
      l->input().node().on_property_changed();
    }
  }
}
//------------------------------------------------------------------------------
auto scalar::draw_properties() -> bool {
  bool changed = false;

  if (ImGui::DragDouble("value", &m_value, 0.01)) {
    if (m_variation_type == 0) {
      m_internal_value = m_value;
    } else if (m_variation_type == 1) {
      m_internal_value = std::clamp<real_t>(m_internal_value, -1,1);
      m_internal_value = std::asin(m_value);
    } else if (m_variation_type == 2) {
      m_internal_value = std::clamp<real_t>(m_internal_value, -1,1);
      m_internal_value = std::acos(m_value);
    }
    changed = true;
  }
  changed |= ImGui::Checkbox("vary", &m_vary);
  ImGui::SameLine();
  changed |= ImGui::DragDouble("speed", &m_speed, 0.01);
  if (ImGui::RadioButton("linear", &m_variation_type, 0)) {
    m_internal_value = m_value;
    changed = true;
  }
  ImGui::SameLine();
  changed |= ImGui::RadioButton("sine", &m_variation_type, 1);
  ImGui::SameLine();
  changed |= ImGui::RadioButton("cosine", &m_variation_type, 2);

  return changed;
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
