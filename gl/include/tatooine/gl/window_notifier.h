#ifndef TATOOINE_GL_WINDOW_NOTIFIER_H
#define TATOOINE_GL_WINDOW_NOTIFIER_H
//==============================================================================
#include <vector>
#include <tatooine/holder.h>
#include "window_listener.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
template <typename Event>
struct resize_event : holder<Event>, window_listener {
  using holder<Event>::holder;
  void on_resize(int width, int height) override {
    this->get()(width, height);
  }
};
// copy when having rvalue
template <typename T>
resize_event(T &&) -> resize_event<T>;
// keep reference when having lvalue
template <typename T>
resize_event(T const&) -> resize_event<T const&>;
//==============================================================================
template <typename Event>
struct key_pressed_event : holder<Event>, window_listener {
  using holder<Event>::holder;
  void on_key_pressed(key k) override {
    this->get()(k);
  }
};
// copy when having rvalue
template <typename T>
key_pressed_event(T &&) -> key_pressed_event<T>;
// keep reference when having lvalue
template <typename T>
key_pressed_event(T const&) -> key_pressed_event<T const&>;
//==============================================================================
template <typename Event>
struct key_released_event : holder<Event>, window_listener {
  using holder<Event>::holder;
  void on_key_pressed(key k) override {
    this->get()(k);
  }
};
// copy when having rvalue
template <typename T>
key_released_event(T &&) -> key_released_event<T>;
// keep reference when having lvalue
template <typename T>
key_released_event(T const&) -> key_released_event<T const &>;
//==============================================================================
template <typename Event>
struct button_pressed_event : holder<Event>, window_listener {
  using holder<Event>::holder;
  void on_button_pressed(button b) override {
    this->get()(b);
  }
};
// copy when having rvalue
template <typename T>
button_pressed_event(T &&) -> button_pressed_event<T>;
// keep reference when having lvalue
template <typename T>
button_pressed_event(T const &) -> button_pressed_event<T const &>;
//==============================================================================
template <typename Event>
struct button_released_event : holder<Event>, window_listener {
  using holder<Event>::holder;
  void on_button_released(button b) override {
    this->get()(b);
  }
};
// copy when having rvalue
template <typename T>
button_released_event(T &&) -> button_released_event<T>;
// keep reference when having lvalue
template <typename T>
button_released_event(T const &) -> button_released_event<T const &>;
//==============================================================================
template <typename Event>
struct wheel_up_event : holder<Event>, window_listener {
  using holder<Event>::holder;
  void on_wheel_up() override {
    this->get()();
  }
};
// copy when having rvalue
template <typename T>
wheel_up_event(T &&) -> wheel_up_event<T>;
// keep reference when having lvalue
template <typename T>
wheel_up_event(T const &) -> wheel_up_event<T const &>;
//==============================================================================
template <typename Event>
struct wheel_down_event : holder<Event>, window_listener {
  using holder<Event>::holder;
  void on_wheel_down() override {
    this->get()();
  }
};
// copy when having rvalue
template <typename T>
wheel_down_event(T &&) -> wheel_down_event<T>;
// keep reference when having lvalue
template <typename T>
wheel_down_event(T const &) -> wheel_down_event<T const &>;
//==============================================================================
template <typename Event>
struct cursor_moved_event : holder<Event>, window_listener {
  using holder<Event>::holder;
  void on_cursor_moved(double x, double y) override {
    this->get()(x, y);
  }
};
// copy when having rvalue
template <typename T>
cursor_moved_event(T &&) -> cursor_moved_event<T>;
// keep reference when having lvalue
template <typename T>
cursor_moved_event(T const &) -> cursor_moved_event<T const &>;
//==============================================================================
struct window_notifier {
  std::vector<window_listener *> m_window_listeners;
  std::vector<std::unique_ptr<base_holder>> m_events;
  //----------------------------------------------------------------------------
  void notify_key_pressed(key k);
  void notify_key_released(key k);
  void notify_button_pressed(button b);
  void notify_button_released(button b);
  void notify_wheel_up();
  void notify_wheel_down();
  void notify_wheel_left();
  void notify_wheel_right();
  void notify_cursor_moved(double x, double y);
  void notify_resize(int width, int height);
  void notify_close();
  //----------------------------------------------------------------------------
  void add_listener(window_listener &l);
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_resize_event(Event&& event) {
    m_events.push_back(std::unique_ptr<base_holder>{
        new resize_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<gl::window_listener*>(m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_key_pressed_event(Event&& event) {
    m_events.push_back(std::unique_ptr<base_holder>{
        new key_pressed_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<gl::window_listener*>(m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_key_released_event(Event&& event) {
    m_events.push_back(std::unique_ptr<base_holder>{
        new key_released_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<gl::window_listener*>(m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_button_pressed_event(Event&& event) {
    m_events.push_back(std::unique_ptr<base_holder>{
        new button_pressed_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<gl::window_listener*>(m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_button_released_event(Event&& event) {
    m_events.push_back(std::unique_ptr<base_holder>{
        new button_released_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<gl::window_listener*>(m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_cursor_moved_event(Event&& event) {
    m_events.push_back(std::unique_ptr<base_holder>{
        new cursor_moved_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<gl::window_listener*>(m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_wheel_up_event(Event&& event) {
    m_events.push_back(std::unique_ptr<base_holder>{
        new wheel_up_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<gl::window_listener*>(m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_wheel_down_event(Event&& event) {
    m_events.push_back(std::unique_ptr<base_holder>{
        new wheel_down_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<gl::window_listener*>(m_events.back().get()));
  }
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
