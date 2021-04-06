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
    m_value += dt.count() * m_speed;
    notify_property_changed();
  }
}
//------------------------------------------------------------------------------
auto scalar::draw_properties() -> bool {
  bool changed = false;

  changed |= ImGui::DragDouble("value", &m_value, 0.01);
  changed |= ImGui::Checkbox("vary", &m_vary);
  ImGui::SameLine();
  changed |= ImGui::DragDouble("speed", &m_speed, 0.01);

  return changed;
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
