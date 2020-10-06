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
  double          m_max_t   = 10;
  double          m_radius  = 0.1;
  vec<double, 2>* m_x0      = nullptr;
  double          m_t0      = 0;

  line_shader                  m_line_shader;

  yavin::indexeddata<gpu_vec3> m_ellipses;
  std::array<GLfloat, 4>       m_ellipses_color{0.0f, 0.0f, 0.0f, 1.0f};
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
  autonomous_particle(flowexplorer::window& w)
      : renderable{w, "Autonomous Particle"} {
    this->template insert_input_pin<vectorfield_t>("2D Vector Field");
    this->template insert_input_pin<position<2>>("x0");

    S() = mat_t::eye() * m_radius;
    t1() = m_t0;
  }
  //============================================================================
  void render(mat<float, 4, 4> const& projection_matrix,
              mat<float, 4, 4> const& view_matrix) final {
    m_line_shader.bind();
    m_line_shader.set_color(m_ellipses_color[0], m_ellipses_color[1],
                            m_ellipses_color[2], m_ellipses_color[3]);
    m_line_shader.set_projection_matrix(projection_matrix);
    m_line_shader.set_modelview_matrix(view_matrix);
    m_ellipses.draw_lines();
  }
  //----------------------------------------------------------------------------
 private:
  void integrate() {
    std::cerr << *m_x0 << '\n';
    m_ellipses.clear();
    auto const [particles, ellipses] = integrate(m_taustep, m_max_t);
    size_t i = 0;
    for (auto const& ellipse : ellipses) {
      for (auto const& x : ellipse.vertices()) {
        m_ellipses.vertexbuffer().push_back(gpu_vec3{static_cast<float>(x(0)),
                                                     static_cast<float>(x(1)),
                                                     static_cast<float>(x(2))});
        m_ellipses.indexbuffer().push_back(i++);
        m_ellipses.indexbuffer().push_back(i);
      }
      m_ellipses.vertexbuffer().push_back(gpu_vec3{
          static_cast<float>(ellipse.front_vertex()(0)),
          static_cast<float>(ellipse.front_vertex()(1)),
          static_cast<float>(ellipse.front_vertex()(2))});
      ++i;
    }
  }

 public:
  //----------------------------------------------------------------------------
  void draw_ui() final {
    if(ImGui::DragDouble("t0", &m_t0, 0.001)) {
      t1() = m_t0;
    }
    if(ImGui::DragDouble("radius", &m_radius, 0.001, 0.001, 10.0)) {
      S() = mat_t::eye() * m_radius;
    }
    ImGui::DragDouble("tau step", &m_taustep, 0.1, 0.0, 1000.0);
    ImGui::DragDouble("end time", &m_max_t, 0.1, 0.0, 1000.0);
    ImGui::ColorEdit4("ellipses color", m_ellipses_color.data());
    if (ImGui::Button("integrate")) {
      if (m_x0 != nullptr && &this->phi().vectorfield() != nullptr) {
        std::cerr << "integrating...";
        integrate();
        std::cerr << "done!\n";
      }
    }
  }
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool final {
    return false;
  }
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) final {
    if (other_pin.type() == typeid(position<2>)) {
      m_x0 = dynamic_cast<vec<double, 2>*>(&other_pin.node());
      x0() = *m_x0;
      x1() = *m_x0;
    } else if ((other_pin.type() == typeid(vectorfield_t))) {
      this->phi().set_vectorfield(
          dynamic_cast<vectorfield_t*>(&other_pin.node()));
    }
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
