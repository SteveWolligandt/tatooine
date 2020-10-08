#ifndef TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLE_H
#define TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLE_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/flowexplorer/nodes/position.h>
#include <tatooine/flowexplorer/renderable.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct autonomous_particle
    : tatooine::autonomous_particle<numerical_flowmap_field_pointer<
          double, 2, ode::vclibs::rungekutta43, interpolation::cubic>>,
      renderable {
  using this_t = autonomous_particle;
  using parent_t =
      tatooine::autonomous_particle<numerical_flowmap_field_pointer<
          double, 2, ode::vclibs::rungekutta43, interpolation::cubic>>;
  using gpu_vec3      = vec<GLfloat, 3>;
  using vbo_t         = yavin::vertexbuffer<gpu_vec3>;
  using vectorfield_t = parent::vectorfield<double, 2>;
  using parent_t::integrate;
  //============================================================================
  double          m_taustep = 0.1;
  double          m_max_t   = 0.0;
  double          m_radius  = 0.03;
  vec<double, 2>* m_x0      = nullptr;
  double          m_t0      = 0;

  line_shader m_line_shader;

  yavin::indexeddata<gpu_vec3> m_initial_circle;
  yavin::indexeddata<gpu_vec3> m_advected_ellipses;
  yavin::indexeddata<gpu_vec3> m_initial_ellipses_back_calculation;
  std::array<GLfloat, 4>       m_ellipses_color{0.0f, 0.0f, 0.0f, 1.0f};
  bool                         m_integration_going_on = false;
  bool                         m_needs_another_update = false;
  bool                         m_stop_thread          = false;
  // phong_shader                      m_phong_shader;
  // int                               m_integral_curve_width = 1;
  // std::array<GLfloat, 4> m_integral_curve_color{0.0f, 0.0f, 0.0f, 1.0f};
  ////============================================================================
  // autonomous_particle(autonomous_particle const&)     = default;
  // autonomous_particle(autonomous_particle&&) noexcept = default;
  ////============================================================================
  // auto operator=(autonomous_particle const&)
  //  -> autonomous_particle& = default;
  // auto operator=(autonomous_particle&&) noexcept
  //  -> autonomous_particle& = default;
  //============================================================================
  autonomous_particle(flowexplorer::window& w);
  //============================================================================
  void render(mat<float, 4, 4> const& projection_matrix,
              mat<float, 4, 4> const& view_matrix) final;
  //----------------------------------------------------------------------------
 private:
  void integrate() ;

 public:
  //----------------------------------------------------------------------------
  void draw_ui() final ;
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool final ;
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) final ;
  //----------------------------------------------------------------------------
  void update(const std::chrono::duration<double>& dt) {
    if (m_needs_another_update && !m_integration_going_on) {
      integrate();
    }
  }
  void update_initial_circle();
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
