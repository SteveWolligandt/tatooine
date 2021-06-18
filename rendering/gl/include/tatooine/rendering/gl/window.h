#ifndef TATOOINE_RENDERING_GL_WINDOW_H
#define TATOOINE_RENDERING_GL_WINDOW_H
//==============================================================================
#include <yavin/glfw/window.h>
#include <yavin/context.h>
#include <yavin/glincludes.h>
#include <yavin/imgui.h>
#include <yavin/window_listener.h>
#include <yavin/window_notifier.h>

#include <array>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <thread>
//==============================================================================
namespace tatooine::rendering::gl {
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
  static constexpr size_t                         max_num_shared_contexts = 64;
  std::vector<std::unique_ptr<context>>           m_shared_contexts;
  std::array<std::mutex, max_num_shared_contexts> m_shared_context_mutexes;
  std::array<bool, max_num_shared_contexts>       m_shared_context_in_use;
  std::array<bool, max_num_shared_contexts> m_threads_finished_successfully;
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
      std::lock_guard lock{m_async_tasks_mutex};
      m_async_tasks.emplace_back();
      return prev(end(m_async_tasks));
    }();

    *it = std::thread{[win = this, it, g = f] {
      size_t shared_context_id = max_num_shared_contexts;
      for (size_t i = 0; i < size(win->m_shared_context_in_use); ++i) {
        std::lock_guard context_lock{win->m_shared_context_mutexes[i]};
        if (!win->m_shared_context_in_use[i]) {
          shared_context_id = i;
          break;
        }
      }
      if (shared_context_id == max_num_shared_contexts) {
        shared_context_id = 0;
      }
      std::optional<std::exception> caught;
      {
        std::lock_guard context_lock{
            win->m_shared_context_mutexes[shared_context_id]};
        assert(!win->m_shared_context_in_use[shared_context_id]);
        size_t       num_tries     = 0;
        size_t const max_num_tries = 100;
        bool made_current = false;
        win->m_shared_context_in_use[shared_context_id] = true;
        while (!made_current) {
          try {
            win->m_shared_contexts[shared_context_id]->make_current();
            made_current = true;
          } catch (std::runtime_error &e) {
            ++num_tries;
            if (num_tries <= max_num_tries) {
              std::cerr << "failed (" << num_tries << " / " << max_num_tries
                        << ")" << '\n';
              std::this_thread::sleep_for(std::chrono::milliseconds{100});
            } else {
              throw e;
            }
          }
        }
        if (made_current) {
          try {
            g();
          } catch (std::exception &e) {
            caught = e;
          }
          win->m_shared_contexts[shared_context_id]->release();
          win->m_shared_context_in_use[shared_context_id] = false;
        }
        if (caught) {
          throw caught;
        }
      }
      std::lock_guard task_lock{win->m_async_tasks_mutex};
      win->m_joinable_async_tasks.push_back(it);
    }};
  }

 private:
  void setup(const std::string &title, size_t width, size_t height);
  void init_imgui(size_t width, size_t height);
  void deinit_imgui();
};
//==============================================================================
}  // namespace tatooine::rendering::gl
//==============================================================================
#endif
