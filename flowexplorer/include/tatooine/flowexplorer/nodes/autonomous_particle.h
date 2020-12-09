#ifndef TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLE_H
#define TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLE_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/flowexplorer/nodes/position.h>
#include <tatooine/flowexplorer/point_shader.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/numerical_flowmap.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct autonomous_particle
    : tatooine::autonomous_particle<numerical_flowmap_field_pointer<
          double, 2, ode::vclibs::rungekutta43, interpolation::cubic>>,
      renderable<autonomous_particle> {
  using this_t = autonomous_particle;
  using parent_t =
      tatooine::autonomous_particle<numerical_flowmap_field_pointer<
          double, 2, ode::vclibs::rungekutta43, interpolation::cubic>>;
  using gpu_vec3      = vec<GLfloat, 3>;
  using vbo_t         = yavin::vertexbuffer<gpu_vec3>;
  using vectorfield_t = parent::vectorfield<double, 2>;
  using parent_t::advect;
  //============================================================================
  double          m_taustep = 0.1;
  double          m_max_t   = 0.0;
  double          m_radius  = 0.03;
  vec<double, 2>* m_x0      = nullptr;
  double          m_t0      = 0;

  line_shader  m_line_shader;
  point_shader m_point_shader;

  yavin::indexeddata<gpu_vec3> m_initial_circle;
  yavin::indexeddata<gpu_vec3> m_advected_ellipses;
  yavin::indexeddata<gpu_vec3> m_initial_ellipses_back_calculation;
  std::array<GLfloat, 4>       m_ellipses_color{0.0f, 0.0f, 0.0f, 1.0f};
  bool                         m_integration_going_on = false;
  bool                         m_needs_another_update = false;
  bool                         m_stop_thread          = false;
  int                          m_num_splits           = 3;
  std::vector<vec_t>           m_points_on_initial_circle;
  std::vector<vec_t>           m_advected_points_on_initial_circle;
  yavin::indexeddata<gpu_vec3> m_gpu_advected_points_on_initial_circle;
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
  autonomous_particle(flowexplorer::scene& s);
  //============================================================================
  void render(mat<float, 4, 4> const& projection_matrix,
              mat<float, 4, 4> const& view_matrix) final;
  //----------------------------------------------------------------------------
 private:
  void advect();

 public:
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool final;
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool final;
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::input_pin&  this_pin,
                        ui::output_pin& other_pin) final;
  //----------------------------------------------------------------------------
  void update(std::chrono::duration<double> const& /*dt*/) {
    if (m_needs_another_update && !m_integration_going_on) {
      advect();
    }
  }
  void update_initial_circle();
  void generate_random_points_in_initial_circle(size_t const n);
  void advect_random_points_in_initial_circle();
  void upload_advected_random_points_in_initial_circle();
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::autonomous_particle,
    TATOOINE_REFLECTION_INSERT_METHOD(t0, m_t0),
    TATOOINE_REFLECTION_INSERT_METHOD(radius, m_radius),
    TATOOINE_REFLECTION_INSERT_METHOD(tau_step, m_taustep),
    TATOOINE_REFLECTION_INSERT_METHOD(end_time, m_max_t),
    TATOOINE_REFLECTION_INSERT_METHOD(ellipses_color, m_ellipses_color),
    TATOOINE_REFLECTION_INSERT_METHOD(num_splits, m_num_splits))
#endif
