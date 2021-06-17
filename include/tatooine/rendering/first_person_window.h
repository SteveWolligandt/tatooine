#ifndef TATOOINE_RENDERING_FIRST_PERSON_WINDOW_H
#define TATOOINE_RENDERING_FIRST_PERSON_WINDOW_H
//==============================================================================
#include <tatooine/holder.h>
#include <tatooine/ray.h>
#include <tatooine/rendering/camera_controller.h>
#include <yavin/glwrapper.h>
#include <yavin/window.h>

#include <chrono>
#include <cmath>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
struct first_person_window : yavin::window {
  using parent_t = yavin::window;
  size_t                    m_width, m_height;
  struct camera_controller<float>                    m_cam;
  std::chrono::time_point<std::chrono::system_clock> m_time =
      std::chrono::system_clock::now();
  //============================================================================
  first_person_window(size_t width = 800, size_t height = 600)
      : yavin::window{"tatooine first person window", width, height},
        m_width{width},
        m_height{height},
        m_cam{width, height},
        m_time{std::chrono::system_clock::now()} {
    yavin::enable_depth_test();
    this->add_listener(m_cam);
  }
  //============================================================================
  auto width() const { return m_width; }
  //----------------------------------------------------------------------------
  auto height() const { return m_height; }
  //----------------------------------------------------------------------------
  auto camera_controller() -> auto& { return m_cam; }
  //----------------------------------------------------------------------------
  auto camera_controller() const -> auto const& { return m_cam; }
  //----------------------------------------------------------------------------
  template <typename Event>
  auto render_loop(Event&& event) {
    m_time = std::chrono::system_clock::now();
    while (!should_close()) {
      refresh();
      yavin::gl::viewport(0, 0, m_width, m_height);
      auto const before = std::chrono::system_clock::now();
      update(std::forward<Event>(event),
             std::chrono::system_clock::now() - m_time);
      m_time = before;
      render_imgui();
      swap_buffers();
    }
  }
  //----------------------------------------------------------------------------
  template <typename F>
  auto update(F&& f, std::chrono::duration<double> const& dt) {
    m_cam.update(dt);
    f(dt);
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() const {
    return m_cam.projection_matrix();
  }
  //----------------------------------------------------------------------------
  auto view_matrix() const {
    return m_cam.view_matrix();
  }
  //----------------------------------------------------------------------------
  auto on_resize(int w, int h) -> void override {
    parent_t::on_resize(w, h);
    m_width  = w;
    m_height = h;
    m_cam.on_resize(w, h);
  }
  //----------------------------------------------------------------------------
  auto on_key_pressed(yavin::key k) -> void override {
    parent_t::on_key_pressed(k);
    if (k == yavin::KEY_F2) {
      camera_controller().use_orthographic_camera();
      camera_controller().use_orthographic_controller();
    } else if (k == yavin::KEY_F3) {
      camera_controller().use_perspective_camera();
      camera_controller().use_fps_controller();
    } else if (k == yavin::KEY_F4) {
      camera_controller().look_at({0, 0, 0}, {0, 0, -1});
    }
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
