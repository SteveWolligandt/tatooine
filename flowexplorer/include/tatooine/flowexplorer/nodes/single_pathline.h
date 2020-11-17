#ifndef TATOOINE_FLOWEXPLORER_NODES_SINGLE_PATHLINE_H
#define TATOOINE_FLOWEXPLORER_NODES_SINGLE_PATHLINE_H
//==============================================================================
#include <tatooine/flowexplorer/nodes/axis_aligned_bounding_box.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/flowexplorer/nodes/position.h>
#include <tatooine/gpu/line_renderer.h>
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/ode/vclibs/rungekutta43.h>

#include <mutex>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <size_t N>
struct single_pathline : renderable<single_pathline<N>> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using vectorfield_t = parent::vectorfield<double, N>;
  using integrator_t  = ode::vclibs::rungekutta43<double, N>;
  //----------------------------------------------------------------------------

  vectorfield_t const*                                    m_v  = nullptr;
  position<N> const*                                      m_x0 = nullptr;
  integrator_t                                            m_integrator;
  line_shader                                             m_shader;
  yavin::indexeddata<vec<float, 3>, vec<float, 3>, float> m_gpu_data;

  double                 m_t0   = 0;
  double                 m_btau = -5, m_ftau = 5;
  std::array<GLfloat, 4> m_line_color{0.0f, 0.0f, 0.0f, 1.0f};
  int                    m_line_width           = 1;
  bool                   m_integration_going_on = false;
  //----------------------------------------------------------------------------
  single_pathline(flowexplorer::scene& s)
      : renderable<single_pathline<N>>{"Path Line", s} {
    this->template insert_input_pin<vectorfield_t>("Vector Field");
    this->template insert_input_pin<position<N>>("x0");
  }
  //----------------------------------------------------------------------------
  void render(mat<float, 4, 4> const& projection_matrix,
              mat<float, 4, 4> const& view_matrix) override {
    m_shader.bind();
    m_shader.set_color(m_line_color[0], m_line_color[1], m_line_color[2],
                       m_line_color[3]);
    m_shader.set_projection_matrix(projection_matrix);
    m_shader.set_modelview_matrix(view_matrix);
    yavin::gl::line_width(m_line_width);
    m_gpu_data.draw_lines();
  }
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool override {
    bool changed = false;
    changed |= ImGui::DragDouble("t0", &m_t0, 0.1, 0, 100);
    changed |= ImGui::DragDouble("backward tau", &m_btau, 0.1, -100, 0);
    changed |= ImGui::DragDouble("forward tau", &m_ftau, 0.1, 0, 100);
    changed |= ImGui::SliderInt("line width", &m_line_width, 1, 50);
    changed |= ImGui::ColorEdit4("line color", m_line_color.data());
    return changed;
  }
  //----------------------------------------------------------------------------
  void integrate_lines() {
    if (m_integration_going_on) {
      return;
    }
    auto worker = [node = this] {
      size_t index          = 0;
      bool   insert_segment = false;
      auto   callback       = [node, &index, &insert_segment](
                          auto const& y, auto const t, auto const& dy) {
        std::lock_guard lock{node->m_gpu_data.mutex()};
        if constexpr (N == 3) {
          node->m_gpu_data.vertexbuffer().push_back(
              vec<GLfloat, 3>{static_cast<GLfloat>(y(0)),
                              static_cast<GLfloat>(y(1)),
                              static_cast<GLfloat>(y(2))},
              vec<GLfloat, 3>{static_cast<GLfloat>(dy(0)),
                              static_cast<GLfloat>(dy(1)),
                              static_cast<GLfloat>(dy(2))},
              static_cast<GLfloat>(t));
        } else if constexpr (N == 2) {
        node->m_gpu_data.vertexbuffer().push_back(
            vec<GLfloat, 3>{static_cast<GLfloat>(y(0)),
                            static_cast<GLfloat>(y(1)),
                            0.0f},
            vec<GLfloat, 3>{static_cast<GLfloat>(dy(0)),
                            static_cast<GLfloat>(dy(1)),
                            0.0f},
            static_cast<GLfloat>(t));
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
      insert_segment  = false;
      node->m_integrator.solve(*node->m_v, *node->m_x0, node->m_t0, node->m_btau, callback);
      insert_segment = false;
      node->m_integrator.solve(*node->m_v, *node->m_x0, node->m_t0, node->m_ftau, callback);
      node->m_integration_going_on = false;
    };
    worker();
    //this->scene().window().do_async(worker);
  }
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) override {
    if (other_pin.type() == typeid(position<N>)) {
      m_x0 = dynamic_cast<position<N>*>(&other_pin.node());
    } else if ((other_pin.type() == typeid(vectorfield_t))) {
      m_v = dynamic_cast<vectorfield_t*>(&other_pin.node());
    }
    if (m_x0 != nullptr && m_v != nullptr) {
      integrate_lines();
    }
  }
  //----------------------------------------------------------------------------
  void on_pin_disconnected(ui::pin& this_pin) override { m_gpu_data.clear(); }
  //----------------------------------------------------------------------------
  void on_property_changed() override {
    if (m_x0 != nullptr && m_v != nullptr) {
      integrate_lines();
    }
  }
  //----------------------------------------------------------------------------
  bool is_transparent() const override {
    return m_line_color[3] < 1;
  }
};
using single_pathlines2d = single_pathline<2>;
using single_pathlines3d = single_pathline<3>;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::single_pathlines2d,
    TATOOINE_REFLECTION_INSERT_METHOD(t0, m_t0),
    TATOOINE_REFLECTION_INSERT_METHOD(backward_tau, m_btau),
    TATOOINE_REFLECTION_INSERT_METHOD(forward_tau, m_ftau),
    TATOOINE_REFLECTION_INSERT_METHOD(line_width, m_line_width),
    TATOOINE_REFLECTION_INSERT_METHOD(line_color, m_line_color))
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::single_pathlines3d,
    TATOOINE_REFLECTION_INSERT_METHOD(t0, m_t0),
    TATOOINE_REFLECTION_INSERT_METHOD(backward_tau, m_btau),
    TATOOINE_REFLECTION_INSERT_METHOD(forward_tau, m_ftau),
    TATOOINE_REFLECTION_INSERT_METHOD(line_width, m_line_width),
    TATOOINE_REFLECTION_INSERT_METHOD(line_color, m_line_color))
#endif
