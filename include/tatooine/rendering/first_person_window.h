#ifndef TATOOINE_RENDERING_FIRST_PERSON_WINDOW_H
#define TATOOINE_RENDERING_FIRST_PERSON_WINDOW_H
//==============================================================================
#include <tatooine/gl/glwrapper.h>
#include <tatooine/gl/window.h>
#include <tatooine/holder.h>
#include <tatooine/ray.h>
#include <tatooine/rendering/camera_controller.h>

#include <chrono>
#include <cmath>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
struct first_person_window : gl::window {
  using parent_type = gl::window;
  std::size_t                                        m_width, m_height;
  struct camera_controller<float>                    m_camera_controller;
  std::chrono::time_point<std::chrono::system_clock> m_time =
      std::chrono::system_clock::now();
  //============================================================================
  first_person_window(std::size_t width = 800, std::size_t height = 600)
      : gl::window{"tatooine first person window", width, height},
        m_width{width},
        m_height{height},
        m_camera_controller{width, height},
        m_time{std::chrono::system_clock::now()} {
    gl::enable_depth_test();
    this->add_listener(m_camera_controller);
    m_camera_controller.on_resize(width, height);
  }
  virtual ~first_person_window() = default;
  //============================================================================
  auto width() const { return m_width; }
  //----------------------------------------------------------------------------
  auto height() const { return m_height; }
  //----------------------------------------------------------------------------
  auto camera_controller() -> auto& { return m_camera_controller; }
  //----------------------------------------------------------------------------
  auto camera_controller() const -> auto const& { return m_camera_controller; }
  //----------------------------------------------------------------------------
  template <typename Event>
  auto render_loop(Event&& event) {
    m_time = std::chrono::system_clock::now();
    while (!should_close()) {
      refresh();
      m_camera_controller.active_camera().set_gl_viewport();
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
    m_camera_controller.update(dt);
    f(dt);
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() const {
    return m_camera_controller.projection_matrix();
  }
  //----------------------------------------------------------------------------
  auto view_matrix() const { return m_camera_controller.view_matrix(); }
  //----------------------------------------------------------------------------
  auto on_resize(int w, int h) -> void override {
    parent_type::on_resize(w, h);
    m_width  = w;
    m_height = h;
    m_camera_controller.on_resize(w, h);
  }
  //----------------------------------------------------------------------------
  auto on_key_pressed(gl::key k) -> void override {
    parent_type::on_key_pressed(k);
    if (k == gl::KEY_F2) {
      camera_controller().use_orthographic_camera();
      camera_controller().use_orthographic_controller();
    } else if (k == gl::KEY_F3) {
      camera_controller().use_perspective_camera();
      camera_controller().use_fps_controller();
    } else if (k == gl::KEY_F4) {
      camera_controller().look_at({0, 0, 0}, {0, 0, -1});
    }
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
