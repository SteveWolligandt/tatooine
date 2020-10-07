#include <tatooine/flowexplorer/nodes/autonomous_particle.h>
#include <tatooine/flowexplorer/window.h>
#include <tatooine/rendering/yavin_interop.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
autonomous_particle::autonomous_particle(flowexplorer::window& w)
    : renderable{w, "Autonomous Particle"} {
  this->template insert_input_pin<vectorfield_t>("2D Vector Field");
  this->template insert_input_pin<position<2>>("x0");

  S()  = mat_t::eye() * m_radius;
  t1() = m_t0;
}
//============================================================================
void autonomous_particle::render(mat<float, 4, 4> const& projection_matrix,
                                 mat<float, 4, 4> const& view_matrix) {
  m_line_shader.bind();
  m_line_shader.set_color(m_ellipses_color[0], m_ellipses_color[1],
                          m_ellipses_color[2], m_ellipses_color[3]);
  m_line_shader.set_projection_matrix(projection_matrix);
  m_line_shader.set_modelview_matrix(view_matrix);
  
  m_ellipses.draw_lines();
}
//----------------------------------------------------------------------------
void autonomous_particle::integrate() {
  if (m_integration_going_on) {
    //m_needs_another_update = true;
    return;
  }
  m_integration_going_on = true;
  //this->window().do_async([&node = *this] {
  auto& node = *this;
    node.x1() = *node.m_x0;
    node.m_ellipses.clear();
    auto const particles = node.integrate(node.m_taustep, node.m_max_t);
    size_t     i         = 0;
    geometry::sphere<double, 2> ellipse{1.0};
    auto                        discretized_ellipse = discretize(ellipse, 100);
    for (auto const& particle : particles) {
      for (auto const& x : discretized_ellipse.vertices()) {
        std::lock_guard lock{node.m_ellipses.mutex()};
        auto y = particle.S() * x + particle.x1();
        node.m_ellipses.vertexbuffer().push_back(
            gpu_vec3{static_cast<float>(y(0)), static_cast<float>(y(1)),
                     static_cast<float>(particle.t1())});
        node.m_ellipses.indexbuffer().push_back(i++);
        node.m_ellipses.indexbuffer().push_back(i);
      }
      auto y =
          particle.S() * discretized_ellipse.front_vertex() + particle.x1();
      node.m_ellipses.vertexbuffer().push_back(
          gpu_vec3{static_cast<float>(y(0)), static_cast<float>(y(1)),
                   static_cast<float>(particle.t1())});
      ++i;
    }
    node.m_integration_going_on = false;
  //});
}
//----------------------------------------------------------------------------
void autonomous_particle::draw_ui() {
  if (ImGui::DragDouble("t0", &m_t0, 0.001)) {
    t1() = m_t0;
  }
  if (ImGui::DragDouble("radius", &m_radius, 0.001, 0.001, 10.0)) {
    S() = mat_t::eye() * m_radius;
  }
  ImGui::DragDouble("tau step", &m_taustep, 0.01, 0.01, 1000.0);
  if (ImGui::DragDouble("end time", &m_max_t, 0.01, 0.01, 1000.0)) {
    if (m_x0 != nullptr && &this->phi().vectorfield() != nullptr) {
      integrate();
    }
  }
  ImGui::ColorEdit4("ellipses color", m_ellipses_color.data());
  if (ImGui::Button("integrate")) {
    if (m_x0 != nullptr && &this->phi().vectorfield() != nullptr) {
      integrate();
    }
  }
}
//----------------------------------------------------------------------------
auto autonomous_particle::is_transparent() const -> bool {
  return false;
}
//----------------------------------------------------------------------------
void autonomous_particle::on_pin_connected(ui::pin& this_pin,
                                           ui::pin& other_pin) {
  if (other_pin.type() == typeid(position<2>)) {
    m_x0 = dynamic_cast<vec<double, 2>*>(&other_pin.node());
    x0() = *m_x0;
    x1() = *m_x0;
  } else if ((other_pin.type() == typeid(vectorfield_t))) {
    this->phi().set_vectorfield(
        dynamic_cast<vectorfield_t*>(&other_pin.node()));
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
