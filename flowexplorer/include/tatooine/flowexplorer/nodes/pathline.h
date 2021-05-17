#ifndef TATOOINE_FLOWEXPLORER_NODES_PATHLINE_H
#define TATOOINE_FLOWEXPLORER_NODES_PATHLINE_H
//==============================================================================
#include <tatooine/flowexplorer/nodes/axis_aligned_bounding_box.h>
#include <tatooine/flowexplorer/nodes/position.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/gpu/line_renderer.h>
#include <tatooine/ode/vclibs/rungekutta43.h>

#include <mutex>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct pathline : renderable<pathline> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using vectorfield2_t = parent::vectorfield<real_t, 2>;
  using vectorfield3_t = parent::vectorfield<real_t, 3>;
  using integrator2_t  = ode::vclibs::rungekutta43<real_t, 2>;
  using integrator3_t  = ode::vclibs::rungekutta43<real_t, 3>;
  //----------------------------------------------------------------------------
  ui::input_pin&                          m_t0_pin;
  ui::input_pin&                          m_forward_tau_pin;
  ui::input_pin&                          m_backward_tau_pin;
  ui::input_pin&                          m_v_pin;
  ui::input_pin&                          m_x0_pin;
  ui::output_pin&                         m_neg2_pin;
  ui::output_pin&                         m_pos2_pin;
  ui::output_pin&                         m_neg3_pin;
  ui::output_pin&                         m_pos3_pin;
  vec2                                    m_x_neg2, m_x_pos2;
  vec3                                    m_x_neg3, m_x_pos3;
  line<real_t, 3>                         m_cpu_data;
  yavin::indexeddata<vec3f, vec3f, float> m_gpu_data;

  real_t                 m_t0   = 0;
  real_t                 m_btau = -5, m_ftau = 5;
  std::array<GLfloat, 4> m_line_color{0.0f, 0.0f, 0.0f, 1.0f};
  int                    m_line_width           = 1;
  bool                   m_integration_going_on = false;
  //============================================================================
  pathline(flowexplorer::scene& s);
  //============================================================================
  auto render(mat4f const& projection_matrix, mat4f const& view_matrix)
      -> void override;
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool override;
  //----------------------------------------------------------------------------
  auto integrate_lines() -> void;
  //----------------------------------------------------------------------------
  auto on_pin_connected(ui::input_pin& this_pin,
                        ui::output_pin& other_pin) -> void override;
  //----------------------------------------------------------------------------
  auto on_pin_disconnected(ui::input_pin& this_pin) -> void override;
  //----------------------------------------------------------------------------
  auto on_property_changed() -> void override;
  //----------------------------------------------------------------------------
  auto all_pins_linked() const -> bool;
  //----------------------------------------------------------------------------
  auto link_configuration_is_valid() const -> bool;
  //----------------------------------------------------------------------------
  auto num_dimensions() const -> size_t;
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool override;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::pathline,
    TATOOINE_REFLECTION_INSERT_METHOD(t0, m_t0),
    TATOOINE_REFLECTION_INSERT_METHOD(backward_tau, m_btau),
    TATOOINE_REFLECTION_INSERT_METHOD(forward_tau, m_ftau),
    TATOOINE_REFLECTION_INSERT_METHOD(line_width, m_line_width),
    TATOOINE_REFLECTION_INSERT_METHOD(line_color, m_line_color))
#endif
