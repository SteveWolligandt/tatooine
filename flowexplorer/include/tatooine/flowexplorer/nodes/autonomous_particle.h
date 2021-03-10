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
struct autonomous_particle : tatooine::autonomous_particle<real_t, 2>,
                             renderable<autonomous_particle> {
  using this_t   = autonomous_particle;
  using parent_t      = tatooine::autonomous_particle<real_t, 2>;
  using gpu_vec3      = vec<GLfloat, 3>;
  using vbo_t         = yavin::vertexbuffer<gpu_vec3>;
  using vectorfield_t = parent::vectorfield<real_t, 2>;
  using parent_t::advect;
  //============================================================================
  real_t          m_taustep = 0.1;
  real_t          m_max_t   = 0.0;
  real_t          m_radius  = 0.03;
  vec<real_t, 2>* m_x0      = nullptr;
  real_t          m_t0      = 0;

  line_shader  m_line_shader;
  point_shader m_point_shader;

  vectorfield_t* m_v;
  yavin::indexeddata<gpu_vec3> m_initial_circle;
  yavin::indexeddata<gpu_vec3> m_advected_ellipses;
  yavin::indexeddata<gpu_vec3> m_initial_ellipses_back_calculation;
  yavin::indexeddata<gpu_vec3> m_pathlines;
  std::array<GLfloat, 4>       m_ellipses_color{0.0f, 0.0f, 0.0f, 1.0f};
  bool                         m_currently_advecting  = false;
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
  void render(mat4f const& projection_matrix,
              mat4f const& view_matrix) final;
  //----------------------------------------------------------------------------
 private:
  void advect();

 public:
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool final;
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool final;
  //----------------------------------------------------------------------------
  auto on_pin_connected(ui::input_pin& this_pin, ui::output_pin& other_pin)
      -> void final;
  //----------------------------------------------------------------------------
  auto update(std::chrono::duration<real_t> const& /*dt*/) -> void override {
    if (m_needs_another_update && !m_currently_advecting) {
      advect();
    }
  }
  auto update_initial_circle() -> void;
  auto on_property_changed() -> void override;
  auto generate_points_in_initial_circle(size_t const n) -> void;
  auto advect_points_in_initial_circle() -> void;
  auto upload_advected_points_in_initial_circle() -> void;
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
