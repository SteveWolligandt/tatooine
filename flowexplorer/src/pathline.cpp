#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/window.h>
//------------------------------------------------------------------------------
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/flowexplorer/nodes/pathline.h>
#include <tatooine/flowexplorer/nodes/random_points.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
pathline::pathline(flowexplorer::scene& s)
    : renderable<pathline>{"Path Line", s},
      m_t0_pin{insert_input_pin<real_t>("t0")},
      m_backward_tau_pin{insert_input_pin<real_t>("tau -")},
      m_forward_tau_pin{insert_input_pin<real_t>("tau +")},
      m_v_pin{insert_input_pin<vectorfield2_t, vectorfield3_t>("Vector Field")},
      m_x0_pin{
          insert_input_pin<vec2, vec3, std::vector<vec2>, std::vector<vec3>>(
              "x0")},
      m_neg2_pin{insert_output_pin("negative end", m_x_neg2)},
      m_pos2_pin{insert_output_pin("positive end", m_x_pos2)},
      m_neg3_pin{insert_output_pin("negative end", m_x_neg3)},
      m_pos3_pin{insert_output_pin("positive end", m_x_pos3)} {
  insert_input_pin_property_link(m_t0_pin, m_t0);
  insert_input_pin_property_link(m_forward_tau_pin, m_ftau);
  insert_input_pin_property_link(m_backward_tau_pin, m_btau);
  m_neg2_pin.deactivate();
  m_pos2_pin.deactivate();
  m_neg3_pin.deactivate();
  m_pos3_pin.deactivate();
}
//============================================================================
auto pathline::render(mat4f const& P, mat4f const& V) -> void {
  auto & shader = line_shader::get();
  shader.bind();
  shader.set_color(m_line_color[0], m_line_color[1], m_line_color[2],
                     m_line_color[3]);
  shader.set_projection_matrix(P);
  shader.set_modelview_matrix(V);
  rendering::gl::line_width(m_line_width);
  m_gpu_data.draw_lines();
}
//----------------------------------------------------------------------------
auto pathline::draw_properties() -> bool {
  bool changed = false;
  changed |= ImGui::DragDouble("t0", &m_t0, 0.1, 0, 100);
  changed |= ImGui::DragDouble("backward tau", &m_btau, 0.1, -100, 0);
  changed |= ImGui::DragDouble("forward tau", &m_ftau, 0.1, 0, 100);
  changed |= ImGui::SliderInt("line width", &m_line_width, 1, 50);
  changed |= ImGui::ColorEdit4("line color", m_line_color.data());
  if (m_x0_pin.is_linked() && m_v_pin.is_linked() &&
      ((m_x0_pin.linked_type() == typeid(vec2) &&
        m_v_pin.linked_type() == typeid(vectorfield3_t)) ||
       (m_x0_pin.linked_type() == typeid(vec3) &&
        m_v_pin.linked_type() == typeid(vectorfield2_t)))) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(255, 0, 0)));
    ImGui::Text("Position and vector fields number of dimensions differ.");
    ImGui::PopStyleColor();
  }
  if (m_x0_pin.is_linked() && m_v_pin.is_linked() &&
      ((m_x0_pin.linked_type() == typeid(vec2) &&
        m_v_pin.linked_type() == typeid(vectorfield2_t)) ||
       (m_x0_pin.linked_type() == typeid(vec3) &&
        m_v_pin.linked_type() == typeid(vectorfield3_t)))) {
    if (ImGui::Button("write vtk")) {
      m_cpu_data.write_vtk("pathline.vtk");
    }
  }
  return changed;
}
//----------------------------------------------------------------------------
auto pathline::integrate_lines() -> void {
  if (m_integration_going_on) {
    return;
  }
  auto worker = [node = this] {
    size_t index          = 0;
    bool   insert_segment = false;
    bool   forward        = true;
    vec2*  cur_end_point2 = nullptr;
    vec3*  cur_end_point3 = nullptr;
    auto   callback       = [&cur_end_point2, &cur_end_point3, node, &index,
                     &insert_segment,
                     &forward](auto const& y, auto const t, auto const& dy) {
      constexpr auto  N = std::decay_t<decltype(y)>::num_components();
      std::lock_guard lock{node->m_gpu_data.mutex()};
      if constexpr (N == 2) {
        node->m_gpu_data.vertexbuffer().push_back(
            vec3f{static_cast<GLfloat>(y(0)), static_cast<GLfloat>(y(1)), 0.0f},
            vec3f{static_cast<GLfloat>(dy(0)), static_cast<GLfloat>(dy(1)),
                  0.0f},
            static_cast<GLfloat>(t));
        if (forward) {
          node->m_cpu_data.push_back(vec3{static_cast<GLfloat>(y(0)),
                                          static_cast<GLfloat>(y(1)), 0.0f});
        } else {
          node->m_cpu_data.push_front(vec3{static_cast<GLfloat>(y(0)),
                                           static_cast<GLfloat>(y(1)), 0.0f});
        }
        *cur_end_point2 = y;
      } else if constexpr (N == 3) {
        node->m_gpu_data.vertexbuffer().push_back(
            vec3f{static_cast<GLfloat>(y(0)), static_cast<GLfloat>(y(1)),
                  static_cast<GLfloat>(y(2))},
            vec3f{static_cast<GLfloat>(dy(0)), static_cast<GLfloat>(dy(1)),
                  static_cast<GLfloat>(dy(2))},
            static_cast<GLfloat>(t));
        if (forward) {
          node->m_cpu_data.push_back(y);
        }else {
          node->m_cpu_data.push_front(y);
        }
        *cur_end_point3 = y;
      }
      if (insert_segment) {
        node->m_gpu_data.indexbuffer().push_back(index - 1);
        node->m_gpu_data.indexbuffer().push_back(index);
      } else {
        insert_segment = true;
      }
      ++index;
    };
    node->m_gpu_data.clear();
    node->m_cpu_data.clear();
    insert_segment = false;

    size_t const N = [&] { return node->num_dimensions(); }();

    if (N == 2) {
      integrator2_t  integrator;
      decltype(auto) v         = node->m_v_pin.get_linked_as<vectorfield2_t>();
      auto           integrate = [&](auto const& x0) {
        cur_end_point2 = &node->m_x_neg2;
        forward = false;
        integrator.solve(v, x0, node->m_t0, node->m_btau, callback);
        insert_segment = false;

        cur_end_point2 = &node->m_x_pos2;
        forward = true;
        integrator.solve(v, x0, node->m_t0, node->m_ftau, callback);
        insert_segment = false;
      };
      if (node->m_x0_pin.linked_type() == typeid(vec2)) {
        decltype(auto) x0 = node->m_x0_pin.get_linked_as<vec2>();
        integrate(x0);
      } else if (node->m_x0_pin.linked_type() == typeid(std::vector<vec2>)) {
        decltype(auto) x0s = node->m_x0_pin.get_linked_as<std::vector<vec2>>();
        for (auto const& x0 : x0s) {
          integrate(x0);
        }
      }
    } else if (N == 3) {
      integrator3_t  integrator;
      decltype(auto) v         = node->m_v_pin.get_linked_as<vectorfield3_t>();
      auto           integrate = [&](auto const& x0) {
        cur_end_point3 = &node->m_x_neg3;
        forward = false;
        integrator.solve(v, x0, node->m_t0, node->m_btau, callback);
        insert_segment = false;

        cur_end_point3 = &node->m_x_pos3;
        forward = true;
        integrator.solve(v, x0, node->m_t0, node->m_ftau, callback);
        insert_segment = false;
      };
      if (node->m_x0_pin.linked_type() == typeid(vec3)) {
        decltype(auto) x0 = node->m_x0_pin.get_linked_as<vec3>();
        integrate(x0);
      } else if (node->m_x0_pin.linked_type() == typeid(std::vector<vec3>)) {
        decltype(auto) x0s = node->m_x0_pin.get_linked_as<std::vector<vec3>>();
        for (auto const& x0 : x0s) {
          integrate(x0);
        }
      }
    }
    node->m_integration_going_on = false;
  };
  worker();
  // this->scene().window().do_async(worker);
}
//----------------------------------------------------------------------------
auto pathline::on_pin_connected(ui::input_pin& /*this_pin*/,
                                ui::output_pin& /*other_pin*/) -> void {
  if (all_pins_linked() && link_configuration_is_valid()) {
    if (num_dimensions() == 2) {
      if (m_x0_pin.linked_type() == typeid(vec2)) {
        m_neg2_pin.activate();
        m_pos2_pin.activate();
      } else {
        m_neg2_pin.deactivate();
        m_pos2_pin.deactivate();
      }
      m_neg3_pin.deactivate();
      m_pos3_pin.deactivate();
    } else if (num_dimensions() == 3 &&
               m_x0_pin.linked_type() == typeid(vec3)) {
      m_neg2_pin.deactivate();
      m_pos2_pin.deactivate();
      if (m_x0_pin.linked_type() == typeid(vec3)) {
        m_neg3_pin.activate();
        m_pos3_pin.activate();
      } else {
        m_neg3_pin.activate();
        m_pos3_pin.activate();
      }
    }
  }
  on_property_changed();
}
//----------------------------------------------------------------------------
auto pathline::on_pin_disconnected(ui::input_pin& /*this_pin*/) -> void {
  m_gpu_data.clear();
  m_cpu_data.clear();
  if (m_neg2_pin.is_active()) {
    m_neg2_pin.deactivate();
  }

  if (m_pos2_pin.is_active()) {
    m_pos2_pin.deactivate();
  }

  if (m_neg3_pin.is_active()) {
    m_neg3_pin.deactivate();
  }

  if (m_pos3_pin.is_active()) {
    m_pos3_pin.deactivate();
  }
}
//----------------------------------------------------------------------------
auto pathline::on_property_changed() -> void {
  if (all_pins_linked() && link_configuration_is_valid()) {
    integrate_lines();
  }
}
//----------------------------------------------------------------------------
auto pathline::all_pins_linked() const -> bool {
  return m_x0_pin.is_linked() && m_v_pin.is_linked();
}
//----------------------------------------------------------------------------
auto pathline::link_configuration_is_valid() const -> bool {
  return ((m_x0_pin.linked_type() == typeid(vec2) ||
           m_x0_pin.linked_type() == typeid(std::vector<vec2>)) &&
          m_v_pin.linked_type() == typeid(vectorfield2_t)) ||
         ((m_x0_pin.linked_type() == typeid(vec3) ||
           m_x0_pin.linked_type() == typeid(std::vector<vec3>)) &&
          m_v_pin.linked_type() == typeid(vectorfield3_t));
}
//----------------------------------------------------------------------------
auto pathline::num_dimensions() const -> size_t {
  if (((m_x0_pin.linked_type() == typeid(vec2) ||
        m_x0_pin.linked_type() == typeid(std::vector<vec2>)) &&
       m_v_pin.linked_type() == typeid(vectorfield2_t))) {
    return 2;
  } else if (((m_x0_pin.linked_type() == typeid(vec3) ||
               m_x0_pin.linked_type() == typeid(std::vector<vec3>)) &&
              m_v_pin.linked_type() == typeid(vectorfield3_t))) {
    return 3;
  }
  return 0;
}
//----------------------------------------------------------------------------
auto pathline::is_transparent() const -> bool { return m_line_color[3] < 1; }
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
