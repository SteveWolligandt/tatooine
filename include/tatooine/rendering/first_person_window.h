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
  camera_controller<float>                           m_cam;
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
  }
  //============================================================================
  auto width() const {
    return m_width;
  }
  //----------------------------------------------------------------------------
  auto height() const {
    return m_height;
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void render_loop(Event&& event) {
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
  void update(F&& f, std::chrono::duration<double> const& dt) {
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
  void on_resize(int w, int h) override {
    parent_t::on_resize(w, h);
    m_width  = w;
    m_height = h;
    m_cam.on_resize(w, h);
  }
  //----------------------------------------------------------------------------
  void on_key_pressed(yavin::key k) override {
    parent_t::on_key_pressed(k);
    m_cam.on_key_pressed(k);
  }
  //----------------------------------------------------------------------------
  void on_key_released(yavin::key k) override {
    parent_t::on_key_pressed(k);
    m_cam.on_key_released(k);
  }
  void on_button_pressed(yavin::button b) override {
    parent_t::on_button_pressed(b);
    m_cam.on_button_pressed(b);
  }
  void on_button_released(yavin::button b) override {
    parent_t::on_button_released(b);
    m_cam.on_button_released(b);
  }
  void on_mouse_motion(int x, int y) override {
    parent_t::on_mouse_motion(x, y);
    m_cam.on_mouse_motion(x, y);
  }
  auto camera_controller() -> auto& {
    return m_cam;
  }
  auto camera_controller() const -> auto const& {
    return m_cam;
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
