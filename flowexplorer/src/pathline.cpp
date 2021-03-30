#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/window.h>
//------------------------------------------------------------------------------
#include <tatooine/flowexplorer/nodes/pathline.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
pathline::pathline(flowexplorer::scene& s)
    : renderable<pathline>{"Path Line", s},
      m_v_pin{insert_input_pin<vectorfield2_t, vectorfield3_t>("Vector Field")},
      m_x0_pin{insert_input_pin<vec2, vec3>("x0")},
      m_neg2_pin{insert_output_pin("negative end", m_x_neg2)},
      m_pos2_pin{insert_output_pin("positive end", m_x_pos2)},
      m_neg3_pin{insert_output_pin("negative end", m_x_neg3)},
      m_pos3_pin{insert_output_pin("positive end", m_x_pos3)} {
  m_neg2_pin.deactivate();
  m_pos2_pin.deactivate();
  m_neg3_pin.deactivate();
  m_pos3_pin.deactivate();
}
//============================================================================
auto pathline::render(mat4f const& projection_matrix, mat4f const& view_matrix)
    -> void {
  m_shader.bind();
  m_shader.set_color(m_line_color[0], m_line_color[1], m_line_color[2],
                     m_line_color[3]);
  m_shader.set_projection_matrix(projection_matrix);
  m_shader.set_modelview_matrix(view_matrix);
  yavin::gl::line_width(m_line_width);
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
  if (m_x0_pin.is_linked() && m_v_pin.is_linked()) {
    if ((m_x0_pin.linked_type() == typeid(vec2) &&
         m_v_pin.linked_type() == typeid(vectorfield3_t)) ||
        (m_x0_pin.linked_type() == typeid(vec3) ||
         m_v_pin.linked_type() == typeid(vectorfield2_t))) {
      ImGui::Text("Position and vector fields number of dimensions differ.");
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
    vec2*  cur_end_point2 = nullptr;
    vec3*  cur_end_point3 = nullptr;
    auto   callback       = [&cur_end_point2, &cur_end_point3, node, &index,
                     &insert_segment](auto const& y, auto const t,
                                      auto const& dy) {
      constexpr auto  N = std::decay_t<decltype(y)>::num_dimensions();
      std::lock_guard lock{node->m_gpu_data.mutex()};
      if constexpr (N == 2) {
        node->m_gpu_data.vertexbuffer().push_back(
            vec3f{static_cast<GLfloat>(y(0)), static_cast<GLfloat>(y(1)), 0.0f},
            vec3f{static_cast<GLfloat>(dy(0)), static_cast<GLfloat>(dy(1)),
                  0.0f},
            static_cast<GLfloat>(t));
        *cur_end_point2 = y;
      } else if constexpr (N == 3) {
        node->m_gpu_data.vertexbuffer().push_back(
            vec3f{static_cast<GLfloat>(y(0)), static_cast<GLfloat>(y(1)),
                  static_cast<GLfloat>(y(2))},
            vec3f{static_cast<GLfloat>(dy(0)), static_cast<GLfloat>(dy(1)),
                  static_cast<GLfloat>(dy(2))},
            static_cast<GLfloat>(t));
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
    insert_segment = false;

    size_t const N = [&] { return node->num_dimensions(); }();

    if (N == 2) {
      integrator2_t  integrator;
      decltype(auto) v  = node->m_v_pin.linked_object_as<vectorfield2_t>();
      decltype(auto) x0 = node->m_x0_pin.linked_object_as<vec2>();

      cur_end_point2 = &node->m_x_neg2;
      integrator.solve(v, x0, node->m_t0, node->m_btau, callback);
      insert_segment = false;

      cur_end_point2 = &node->m_x_pos2;
      integrator.solve(v, x0, node->m_t0, node->m_ftau, callback);
      node->m_integration_going_on = false;

    } else if (N == 3) {
      integrator3_t  integrator;
      decltype(auto) v  = node->m_v_pin.linked_object_as<vectorfield3_t>();
      decltype(auto) x0 = node->m_x0_pin.linked_object_as<vec3>();

      cur_end_point3 = &node->m_x_neg3;
      integrator.solve(v, x0, node->m_t0, node->m_btau, callback);
      insert_segment = false;

      cur_end_point3 = &node->m_x_pos3;
      integrator.solve(v, x0, node->m_t0, node->m_ftau, callback);
      node->m_integration_going_on = false;
    }
  };
  worker();
  // this->scene().window().do_async(worker);
}
//----------------------------------------------------------------------------
auto pathline::on_pin_connected(ui::input_pin& /*this_pin*/,
                                ui::output_pin& /*other_pin*/) -> void {
  if (all_pins_linked() && link_configuration_is_valid()) {
    if (num_dimensions() == 2) {
      m_neg2_pin.activate();
      m_pos2_pin.activate();
      m_neg3_pin.deactivate();
      m_pos3_pin.deactivate();
    } else if (num_dimensions() == 3) {
      m_neg2_pin.deactivate();
      m_pos2_pin.deactivate();
      m_neg3_pin.activate();
      m_pos3_pin.activate();
    }
  }
  on_property_changed();
}
//----------------------------------------------------------------------------
auto pathline::on_pin_disconnected(ui::input_pin& /*this_pin*/) -> void {
  m_gpu_data.clear();
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
  return (m_x0_pin.linked_type() == typeid(vec2) &&
          m_v_pin.linked_type() == typeid(vectorfield2_t)) ||
         (m_x0_pin.linked_type() == typeid(vec3) &&
          m_v_pin.linked_type() == typeid(vectorfield3_t));
}
//----------------------------------------------------------------------------
auto pathline::num_dimensions() const -> size_t {
  if ((m_x0_pin.linked_type() == typeid(vec2) &&
       m_v_pin.linked_type() == typeid(vectorfield2_t))) {
    return 2;
  } else if ((m_x0_pin.linked_type() == typeid(vec3) &&
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
