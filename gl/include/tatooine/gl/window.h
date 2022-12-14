#ifndef TATOOINE_GL_WINDOW_H
#define TATOOINE_GL_WINDOW_H
//==============================================================================
#include <tatooine/gl/glfw/window.h>
#include <tatooine/gl/context.h>
#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/imgui.h>
#include <tatooine/gl/window_listener.h>
#include <tatooine/gl/window_notifier.h>

#include <array>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <thread>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class window : public window_notifier, public window_listener {
 public:
  //============================================================================
  // members
  //============================================================================
  std::unique_ptr<glfw::window>                   m_glfw_window;
  std::unique_ptr<struct imgui_render_backend>    m_imgui_render_backend;
  std::list<std::thread>                          m_async_tasks;
  std::vector<std::list<std::thread>::iterator>   m_joinable_async_tasks;
  std::mutex                                      m_async_tasks_mutex;
  //============================================================================
  auto imgui_render_backend() const -> auto const & {
    return *m_imgui_render_backend;
  }
  auto imgui_render_backend() -> auto & { return *m_imgui_render_backend; }
  //============================================================================
  // ctors / dtor
  //============================================================================
 public:
  window(const std::string &title, size_t width, size_t height);
  ~window();

  //============================================================================
  // methods
  //============================================================================
 public:
  void make_current();
  void release();
  void refresh();
  void render_imgui();
  void swap_buffers();
  void check_events();
  void on_key_pressed(key /*k*/) override;
  void on_key_released(key /*k*/) override;
  void on_button_pressed(button /*b*/) override;
  void on_button_released(button /*b*/) override;
  void on_wheel_up() override;
  void on_wheel_down() override;
  void on_wheel_left() override;
  void on_wheel_right() override;
  void on_cursor_moved(double /*x*/, double /*y*/) override;
  void on_resize(int /*width*/, int /*height*/) override;
  auto get() -> auto& {return *m_glfw_window;}
  auto get() const -> auto const& {return *m_glfw_window;}
  //----------------------------------------------------------------------------
  auto should_close() const { return m_glfw_window->should_close(); }
  auto primary_screen_resolution() const {
    auto                monitor = glfwGetPrimaryMonitor();
    std::pair<int, int> res;
    glfwGetMonitorPhysicalSize(monitor, &res.first, &res.second);
    return res;
  }
  //----------------------------------------------------------------------------
  template <typename F>
  void do_async(F &&f) {
    auto it = [this] {
      auto lock = std::lock_guard{m_async_tasks_mutex};
      m_async_tasks.emplace_back();
      return prev(end(m_async_tasks));
    }();

    *it = std::thread{[this, it, f = std::forward<F>(f)] {
      auto ctx = context{*this};
      ctx.make_current();
      f();
      std::lock_guard task_lock{m_async_tasks_mutex};
      m_joinable_async_tasks.push_back(it);
    }};
  }

 private:
  void setup(const std::string &title, size_t width, size_t height);
  void init_imgui(size_t width, size_t height);
  void deinit_imgui();
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
