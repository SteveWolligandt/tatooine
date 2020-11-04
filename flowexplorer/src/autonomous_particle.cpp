#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/window.h>
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#include <tatooine/flowexplorer/nodes/autonomous_particle.h>
#include <tatooine/random.h>
#include <tatooine/rendering/yavin_interop.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
autonomous_particle::autonomous_particle(flowexplorer::scene& s)
    : renderable<autonomous_particle>{"Autonomous Particle", s} {
  this->template insert_input_pin<vectorfield_t>("2D Vector Field");
  this->template insert_input_pin<position<2>>("x0");

  S()  = mat_t::eye() * m_radius;
  t1() = m_t0;
}
//============================================================================
void autonomous_particle::render(mat<float, 4, 4> const& projection_matrix,
                                 mat<float, 4, 4> const& view_matrix) {
  m_line_shader.set_projection_matrix(projection_matrix);
  m_line_shader.set_modelview_matrix(view_matrix);
  m_line_shader.bind();
  m_line_shader.set_color(m_ellipses_color[0], m_ellipses_color[1],
                          m_ellipses_color[2], m_ellipses_color[3]);

  m_line_shader.set_color(0.7, 0.7, 0.7, 1);
  m_initial_circle.draw_lines();
  m_line_shader.set_color(0, 0, 0, 1);
  yavin::gl::line_width(2);
  m_advected_ellipses.draw_lines();
  yavin::gl::line_width(1);
  m_line_shader.set_color(0.2, 0.8, 0.2, 1);
  m_initial_ellipses_back_calculation.draw_lines();
  m_line_shader.set_color(0.5, 0.5, 0.5, 1);
  m_gpu_advected_points_on_initial_circle.draw_lines();
}
//----------------------------------------------------------------------------
void autonomous_particle::update_initial_circle() {
  size_t                      i = 0;
  geometry::sphere<double, 2> ellipse{1.0};
  auto                        discretized_ellipse = discretize(ellipse, 100);
  std::lock_guard             lock{m_initial_circle.mutex()};
  m_initial_circle.clear();
  for (auto const& x : discretized_ellipse.vertices()) {
    if (m_stop_thread) {
      break;
    }
    auto y = mat_t::eye() * m_radius * x + *m_x0;
    m_initial_circle.vertexbuffer().push_back(
        gpu_vec3{static_cast<float>(y(0)), static_cast<float>(y(1)),
                 static_cast<float>(m_t0)});
    m_initial_circle.indexbuffer().push_back(i++);
    m_initial_circle.indexbuffer().push_back(i);
  }
  auto y = mat_t::eye() * m_radius * discretized_ellipse.front_vertex() + *m_x0;
  m_initial_circle.vertexbuffer().push_back(gpu_vec3{static_cast<float>(y(0)),
                                                     static_cast<float>(y(1)),
                                                     static_cast<float>(m_t0)});
  generate_random_points_in_initial_circle(3000);
  advect_random_points_in_initial_circle();
  upload_advected_random_points_in_initial_circle();
}

void autonomous_particle::advect() {
  if (m_integration_going_on) {
    m_stop_thread          = true;
    m_needs_another_update = true;
    return;
  }
  m_integration_going_on = true;
  m_stop_thread          = false;
  m_needs_another_update = false;

  auto run = [node = this] {
    node->advect_random_points_in_initial_circle();
    node->upload_advected_random_points_in_initial_circle();
    node->x0() = *node->m_x0;
    node->x1() = *node->m_x0;

    auto const particles = [&node] {
      switch (node->m_num_splits) {
        case 2:
          return node->advect_with_2_splits(node->m_taustep, node->m_max_t,
                                            node->m_stop_thread);
        case 3:
          return node->advect_with_3_splits(node->m_taustep, node->m_max_t,
                                            node->m_stop_thread);
        case 5:
          return node->advect_with_5_splits(node->m_taustep, node->m_max_t,
                                            node->m_stop_thread);
        case 7:
          return node->advect_with_7_splits(node->m_taustep, node->m_max_t,
                                            node->m_stop_thread);
      }
      return std::vector<parent_t>{};
    }();
    geometry::sphere<double, 2> ellipse{1.0};
    auto                        discretized_ellipse = discretize(ellipse, 100);
    size_t                      i                   = 0;
    {
      {
        std::lock_guard lock{node->m_advected_ellipses.mutex()};
        node->m_advected_ellipses.clear();
      }
      i = 0;
      for (auto const& particle : particles) {
        if (node->m_stop_thread) {
          break;
        }
        std::lock_guard lock{node->m_advected_ellipses.mutex()};
        for (auto const& x : discretized_ellipse.vertices()) {
          if (node->m_stop_thread) {
            break;
          }
          auto y = particle.S() * x + particle.x1();
          node->m_advected_ellipses.vertexbuffer().push_back(
              gpu_vec3{static_cast<float>(y(0)), static_cast<float>(y(1)),
                       static_cast<float>(particle.t1())});
          node->m_advected_ellipses.indexbuffer().push_back(i++);
          node->m_advected_ellipses.indexbuffer().push_back(i);
        }
        auto y =
            particle.S() * discretized_ellipse.front_vertex() + particle.x1();
        node->m_advected_ellipses.vertexbuffer().push_back(
            gpu_vec3{static_cast<float>(y(0)), static_cast<float>(y(1)),
                     static_cast<float>(particle.t1())});
        ++i;
      }
    }

    // back_advected ellipses
    {
      {
        std::lock_guard
        lock{node->m_initial_ellipses_back_calculation.mutex()};
        node->m_initial_ellipses_back_calculation.clear();
      }
      i = 0;
      for (auto const& particle : particles) {
        if (node->m_stop_thread) {
          break;
        }
        std::lock_guard
        lock{node->m_initial_ellipses_back_calculation.mutex()}; for (auto
        const& x : discretized_ellipse.vertices()) {
          if (node->m_stop_thread) {
            break;
          }
          auto sqrS = inv(particle.nabla_phi1()) * particle.S() * particle.S()
          *
                      inv(transposed(particle.nabla_phi1()));
          auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
          eig_vals = {std::sqrt(eig_vals(0)), std::sqrt(eig_vals(1))};
          auto S   = eig_vecs * diag(eig_vals) * transposed(eig_vecs);
          auto y   = S * x + particle.x0();
          node->m_initial_ellipses_back_calculation.vertexbuffer().push_back(
              gpu_vec3{static_cast<float>(y(0)), static_cast<float>(y(1)),
                       static_cast<float>(particle.t1())});
          node->m_initial_ellipses_back_calculation.indexbuffer().push_back(
              i++);
          node->m_initial_ellipses_back_calculation.indexbuffer().push_back(i);
        }
        auto sqrS = inv(particle.nabla_phi1()) * particle.S() * particle.S() *
                    inv(transposed(particle.nabla_phi1()));
        auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
        eig_vals = {std::sqrt(eig_vals(0)), std::sqrt(eig_vals(1))};
        auto S   = eig_vecs * diag(eig_vals) * transposed(eig_vecs);
        auto y   = S * discretized_ellipse.front_vertex() + particle.x0();
        node->m_initial_ellipses_back_calculation.vertexbuffer().push_back(
            gpu_vec3{static_cast<float>(y(0)), static_cast<float>(y(1)),
                     static_cast<float>(particle.t1())});
        ++i;
      }
    }
    node->m_integration_going_on = false;
  };

  this->scene().window().do_async(run);
  //run();
}
//----------------------------------------------------------------------------
auto autonomous_particle::draw_properties() -> bool {
  bool do_advect = false;
  if (ImGui::DragDouble("t0", &m_t0, 0.001)) {
    do_advect = true;
    t1()      = m_t0;
  }
  if (ImGui::DragDouble("radius", &m_radius, 0.001, 0.001, 10.0)) {
    do_advect = true;
    S()       = mat_t::eye() * m_radius;
    update_initial_circle();
  }
  ImGui::DragDouble("tau step", &m_taustep, 0.01, 0.01, 1000.0);
  if (ImGui::Button("<")) {
    do_advect = true;
    m_max_t -= 0.01;
    //advect_random_points_in_initial_circle();
    //upload_advected_random_points_in_initial_circle();
  }
  ImGui::SameLine();
  if (ImGui::DragDouble("end time", &m_max_t, 0.01, 0.0, 1000.0)) {
    do_advect = true;
    //advect_random_points_in_initial_circle();
    //upload_advected_random_points_in_initial_circle();
  }
  ImGui::SameLine();
  if (ImGui::Button(">")) {
    do_advect = true;
    m_max_t += 0.01;
    //advect_random_points_in_initial_circle();
    //upload_advected_random_points_in_initial_circle();
  }

  ImGui::ColorEdit4("ellipses color", m_ellipses_color.data());
  do_advect |= ImGui::Button("advect");
  if (m_integration_going_on && !m_stop_thread) {
    ImGui::SameLine();
    if (ImGui::Button("terminate")) {
      m_stop_thread = true;
    }
  }
  ImGui::TextUnformatted("number of splits:");
  ImGui::SameLine();
  if (ImGui::Button("< ")) {
    switch (m_num_splits) {
      case 3:
        m_num_splits = 2;
        do_advect    = true;
        break;
      case 5:
        m_num_splits = 3;
        do_advect    = true;
        break;
      case 7:
        m_num_splits = 5;
        do_advect    = true;
        break;
    }
  }
  ImGui::SameLine();
  ImGui::TextUnformatted(std::to_string(m_num_splits).c_str());
  ImGui::SameLine();
  if (ImGui::Button(" >")) {
    switch (m_num_splits) {
      case 2:
        m_num_splits = 3;
        do_advect    = true;
        break;
      case 3:
        m_num_splits = 5;
        do_advect    = true;
        break;
      case 5:
        m_num_splits = 7;
        do_advect    = true;
        break;
    }
  }
  if (do_advect && m_x0 != nullptr && &this->phi().vectorfield() != nullptr) {
    advect();
  }
  return false;
}
//----------------------------------------------------------------------------
auto autonomous_particle::is_transparent() const -> bool { return false; }
//----------------------------------------------------------------------------
void autonomous_particle::on_pin_connected(ui::pin& this_pin,
                                           ui::pin& other_pin) {
  if (other_pin.type() == typeid(position<2>)) {
    m_x0 = dynamic_cast<vec<double, 2>*>(&other_pin.node());
    x0() = *m_x0;
    x1() = *m_x0;
    if (&this->phi().vectorfield() != nullptr) {
      update_initial_circle();
    }
  } else if ((other_pin.type() == typeid(vectorfield_t))) {
    this->phi().set_vectorfield(
        dynamic_cast<vectorfield_t*>(&other_pin.node()));
    if (m_x0 != nullptr) {
      update_initial_circle();
    }
  }
}
//------------------------------------------------------------------------------
void autonomous_particle::generate_random_points_in_initial_circle(
    size_t const n) {
  m_points_on_initial_circle.clear();
  random_uniform<real_t> rand;
  for (size_t i = 0; i < n; ++i) {
    real_t alpha = M_PI * 2 * real_t(i) / (n - 1);
    m_points_on_initial_circle.emplace_back(std::cos(alpha) * m_radius,
                                            std::sin(alpha) * m_radius);
    m_points_on_initial_circle.back() += *m_x0;
  }
}
//------------------------------------------------------------------------------
void autonomous_particle::advect_random_points_in_initial_circle() {
  m_advected_points_on_initial_circle.clear();
  for (auto const& x : m_points_on_initial_circle) {
    m_advected_points_on_initial_circle.push_back(
        phi()(x, m_t0, m_max_t - m_t0));
  }
}
//------------------------------------------------------------------------------
void autonomous_particle::upload_advected_random_points_in_initial_circle() {
  m_gpu_advected_points_on_initial_circle.indexbuffer().resize(
      size(m_advected_points_on_initial_circle) * 2);
  m_gpu_advected_points_on_initial_circle.vertexbuffer().resize(
      size(m_advected_points_on_initial_circle));
  size_t i = 0;
  for (auto const& x : m_advected_points_on_initial_circle) {
    m_gpu_advected_points_on_initial_circle.vertexbuffer()[i] = {
        static_cast<float>(x(0)), static_cast<float>(x(1)), 0.0f};
    m_gpu_advected_points_on_initial_circle.indexbuffer()[i * 2] = i;
    m_gpu_advected_points_on_initial_circle.indexbuffer()[i * 2 + 1] =
        (i + 1) % size(m_advected_points_on_initial_circle);
    ++i;
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
