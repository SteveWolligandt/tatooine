#ifndef TATOOINE_FIRST_PERSON_WINDOW_H
#define TATOOINE_FIRST_PERSON_WINDOW_H
//==============================================================================
#include <yavin/perspectivecamera.h>
#include <yavin/vec.h>
#include <yavin/glwrapper.h>
#include <yavin/window.h>
#include <tatooine/holder.h>
#include <tatooine/ray.h>

#include <chrono>
#include <cmath>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Event>
struct key_pressed_event : holder<Event>, yavin::window_listener {
  using holder<Event>::holder;
  void on_key_pressed(yavin::key k) override {this->get()(k);}
};
// copy when having rvalue
template <typename T>
key_pressed_event(T &&)->key_pressed_event<T>;
// keep reference when having lvalue
template <typename T>
key_pressed_event(const T&)->key_pressed_event<const T&>;
//==============================================================================
template <typename Event>
struct key_released_event : holder<Event>, yavin::window_listener {
  using holder<Event>::holder;
  void on_key_pressed(yavin::key k) override {this->get()(k);}
};
// copy when having rvalue
template <typename T>
key_released_event(T &&)->key_released_event<T>;
// keep reference when having lvalue
template <typename T>
key_released_event(const T&)->key_released_event<const T&>;
//==============================================================================
template <typename Event>
struct button_pressed_event : holder<Event>, yavin::window_listener {
  using holder<Event>::holder;
  void on_button_pressed(yavin::button b) override {this->get()(b);}
};
// copy when having rvalue
template <typename T>
button_pressed_event(T &&)->button_pressed_event<T>;
// keep reference when having lvalue
template <typename T>
button_pressed_event(const T&)->button_pressed_event<const T&>;
//==============================================================================
template <typename Event>
struct button_released_event : holder<Event>, yavin::window_listener {
  using holder<Event>::holder;
  void on_button_released(yavin::button b) override {this->get()(b);}
};
// copy when having rvalue
template <typename T>
button_released_event(T &&)->button_released_event<T>;
// keep reference when having lvalue
template <typename T>
button_released_event(const T&)->button_released_event<const T&>;
//==============================================================================
template <typename Event>
struct mouse_motion_event : holder<Event>, yavin::window_listener {
  using holder<Event>::holder;
  void on_mouse_motion(int x, int y) override {this->get()(x, y);}
};
// copy when having rvalue
template <typename T>
mouse_motion_event(T &&)->mouse_motion_event<T>;
// keep reference when having lvalue
template <typename T>
mouse_motion_event(const T&)->mouse_motion_event<const T&>;
//==============================================================================
struct first_person_window : yavin::window, yavin::window_listener {
  bool         m_run;
  GLsizei m_width, m_height;
  yavin::vec3                                        m_eye, m_look_dir, m_up;
  float                                              m_theta = 0, m_phi = M_PI/2;
  yavin::perspectivecamera m_cam;
  std::chrono::time_point<std::chrono::system_clock> m_time =
      std::chrono::system_clock::now();
  int  m_mouse_pos_x, m_mouse_pos_y;
  bool m_middle_button_down = false;
  bool m_w_down = false;
  bool m_s_down = false;
  bool m_a_down = false;
  bool m_d_down = false;
  bool m_q_down = false;
  bool m_e_down = false;
  std::vector<std::unique_ptr<base_holder>> m_events;
  //============================================================================
  first_person_window(GLsizei width = 800, GLsizei height = 600)
      : yavin::window{"tatooine first person window", width, height},
        m_run{true},
        m_width{width},
        m_height{height},
        m_eye{0, 0, 0},
        m_look_dir{0, 0, -1},
        m_up{0, 1, 0},
        m_cam{60,    (float)(width) / (float)(height), 0.01f, 1000.0f, width,
              height},
        m_time{std::chrono::system_clock::now()} {
    add_listener(*this);
    yavin::enable_depth_test();
  }
  //============================================================================
  template <typename Event>
  void render_loop(Event&&event) {
    m_time = std::chrono::system_clock::now();
    while (m_run) {
      refresh();
      yavin::gl::viewport(m_cam);
      update(std::forward<Event>(event), std::chrono::system_clock::now() - m_time);
      render_imgui();
      swap_buffers();
    }
  }
  //----------------------------------------------------------------------------
  template <typename F>
  void update(F&& f, const std::chrono::duration<double>& dt) {
    auto ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dt).count());

    m_time = std::chrono::system_clock::now();
    if (m_w_down) { m_eye += m_look_dir / ms; }
    if (m_s_down) { m_eye -= m_look_dir / ms; }
    if (m_q_down) { m_eye(1) += 1 / ms; }
    if (m_e_down) { m_eye(1) -= 1 / ms; }
    if (m_a_down) {
      const auto right = cross(yavin::vec3{0, 1, 0}, -m_look_dir);
      m_eye -= right / ms;
    }
    if (m_d_down) {
      const auto right = cross(yavin::vec3{0, 1, 0}, -m_look_dir);
      m_eye += right / ms;
    }
    m_cam.look_at(m_eye, m_eye + m_look_dir + 0.1);
    f(dt);
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() {
    return m_cam.projection_matrix();
  }
  //----------------------------------------------------------------------------
  auto view_matrix() {
    return *inverse(look_at_matrix(m_eye, m_eye + m_look_dir));
  }
  //============================================================================
  void on_key_pressed(yavin::key k) override {
    if (k == yavin::KEY_W) {
      m_w_down = true;
    } else if (k == yavin::KEY_S) {
      m_s_down = true;
    } else if (k == yavin::KEY_A) {
      m_a_down = true;
    } else if (k == yavin::KEY_D) {
      m_d_down = true;
    } else if (k == yavin::KEY_Q) {
      m_q_down = true;
    } else if (k == yavin::KEY_E) {
      m_e_down = true;
    } else if (k == yavin::KEY_ESCAPE) {
      m_run = false;
    }
  }
  //----------------------------------------------------------------------------
  void on_key_released(yavin::key k) override {
    if (k == yavin::KEY_W) {
      m_w_down = false;
    } else if (k == yavin::KEY_S) {
      m_s_down = false;
    } else if (k == yavin::KEY_A) {
      m_a_down = false;
    } else if (k == yavin::KEY_D) {
      m_d_down = false;
    } else if (k == yavin::KEY_Q) {
      m_q_down = false;
    } else if (k == yavin::KEY_E) {
      m_e_down = false;
    }
  }
  //----------------------------------------------------------------------------
  void on_button_pressed(yavin::button b) override {
    if (b == yavin::BUTTON_MIDDLE) { m_middle_button_down = true; }
  }
  //----------------------------------------------------------------------------
  void on_button_released(yavin::button b) override {
    if (b == yavin::BUTTON_MIDDLE) { m_middle_button_down = false; }
  }
  //----------------------------------------------------------------------------
  void on_mouse_motion(int x, int y) override {
    if (m_middle_button_down) {
      int offset_x = x - m_mouse_pos_x;
      int offset_y = y - m_mouse_pos_y;
      m_theta -= offset_x * 0.01f;
      m_phi = std::min<float>(M_PI - 0.3f, std::max(0.3f, m_phi + offset_y * 0.01f));
      m_look_dir = yavin::vec3{
          std::sin(m_phi) * std::sin(m_theta),
          std::cos(m_phi),
          std::sin(m_phi) * std::cos(m_theta),
      };
    }
    m_mouse_pos_x = x;
    m_mouse_pos_y = y;
  }
  //----------------------------------------------------------------------------
  void on_resize(int w, int h) override {
    m_width  = w;
    m_height = h;
    m_cam.set_projection(60, (float)(h) / (float)(w), 0.01f, 1000.0f, w, h);
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_key_pressed_event(Event&& event) {
    m_events.push_back(
        std::unique_ptr<base_holder>{new key_pressed_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<yavin::window_listener*>(
        m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_key_released_event(Event&& event) {
    m_events.push_back(
        std::unique_ptr<base_holder>{new key_released_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<yavin::window_listener*>(
        m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_button_pressed_event(Event&& event) {
    m_events.push_back(
        std::unique_ptr<base_holder>{new button_pressed_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<yavin::window_listener*>(
        m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_button_released_event(Event&& event) {
    m_events.push_back(
        std::unique_ptr<base_holder>{new button_released_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<yavin::window_listener*>(
        m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void add_mouse_motion_event(Event&& event) {
    m_events.push_back(
        std::unique_ptr<base_holder>{new mouse_motion_event{std::forward<Event>(event)}});
    add_listener(*dynamic_cast<yavin::window_listener*>(
        m_events.back().get()));
  }
  //----------------------------------------------------------------------------
  auto cast_ray(float x, float y) const {
    y = m_cam.viewport_height() - y;
    auto  look_dir          = normalize(m_look_dir);
    auto  right             = normalize(cross(look_dir, m_up));
    auto  up                = normalize(cross(right, look_dir));
    float plane_half_width =
        std::tan(m_cam.fovy() * 0.5 * M_PI / 180) * m_cam.near();
    float plane_half_height = float(m_cam.viewport_height()) /
                              float(m_cam.viewport_width()) * plane_half_width;
    auto bottom_left = m_eye + look_dir * m_cam.near() -
                       right * plane_half_width - up * plane_half_height;
    auto plane_base_x =
        right * 2.0f * plane_half_width / float(m_cam.viewport_width());
    auto plane_base_y =
        up * 2.0f * plane_half_height / float(m_cam.viewport_height());

    auto planepos = bottom_left + plane_base_x * x + plane_base_y * y;

    vec<double, 3> origin{m_eye(0), m_eye(1), m_eye(2)};
    vec<double, 3> dir = normalize(vec<double, 3>{planepos(0), planepos(1), planepos(2)} - origin);

    std::cerr << x << ", " << y << '\n';
    std::cerr << "eye:         " << m_eye << '\n';
    std::cerr << "look at:     " << m_look_dir << '\n';
    std::cerr << "look dir:    " << look_dir << '\n';
    std::cerr << "right:       " << right << '\n';
    std::cerr << "up:          " << up << '\n';
    std::cerr << "near:        " << m_cam.near() << '\n';
    std::cerr << "fovy:        " << m_cam.fovy() << '\n';
    std::cerr << "width:       " << m_cam.viewport_width() << '\n';
    std::cerr << "height:      " << m_cam.viewport_height() << '\n';
    std::cerr << "half width:  " << plane_half_width << '\n';
    std::cerr << "half height: " << plane_half_height << '\n';
    std::cerr << "bottom left: " << bottom_left << '\n';
    return ray{std::move(origin), std::move(dir)};
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
