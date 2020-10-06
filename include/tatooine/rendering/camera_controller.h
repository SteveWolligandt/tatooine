#ifndef TATOOINE_RENDERING_CAMERA_CONTROLLER_H
#define TATOOINE_RENDERING_CAMERA_CONTROLLER_H
//==============================================================================
#include <tatooine/rendering/perspective_camera.h>
#include <tatooine/rendering/orthographic_camera.h>
//==============================================================================
namespace tatooine::rendering{
//==============================================================================
template <typename Real>
struct camera_controller;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Real>
struct fps_camera_controller;
//==============================================================================
template <typename Real>
struct camera_controller_interface {
 private:
  camera_controller<Real>* controller;
  //----------------------------------------------------------------------------
 public:
  auto perspective_camera() -> auto& {
    return controller->perspective_camera();
  }
  auto perspective_camera() const -> auto const& {
    return controller->perspective_camera();
  }
  auto orthographic_camera() -> auto& {
    return controller->orthographic_camera();
  }
  auto orthographic_camera() const -> auto const& {
    return controller->orthographic_camera();
  }
  //----------------------------------------------------------------------------
  virtual void on_key_pressed(yavin::key k) {}
  virtual void on_key_released(yavin::key k) {}
  virtual void on_button_pressed(yavin::button b) {}
  virtual void on_button_released(yavin::button b) {}
  virtual void on_mouse_motion(int x, int y) {}
  virtual void on_resize(int w, int h) {}
}
//==============================================================================
template <typename Real>
struct camera_controller : yavin::window_listener {
  friend struct camera_controller_interface<Real>;
  perspective_camera<Real>                           m_pcam;
  orthographic_camera<Real>                          m_ocam;
  camera<Real>*                                      m_active_cam;
  std::unique_ptr<camera_controller_interface<Real>> m_controller;
  //============================================================================
  camera_controller(size_t const res_x, size_t const res_y)
      : m_pcam{{Real(0), Real(0), Real(0)},
               {Real(0), Real(0), Real(-1)},
               {Real(0), Real(1), Real(0)},
               60, 0.01, 10000,
               res_x, res_y},
        m_ocam{{Real(0), Real(0), Real(0)},
               {Real(0), Real(0), Real(-1)},
               {Real(0), Real(1), Real(0)},
               -1, 1,
               -1, 1,
               0, 10000,
               res_x, res_y},
  m_active_cam{&m_pcam} {
    use_fps_controller();
  }
  //============================================================================
  void use_perspective_camera() {
    m_active_cam = &m_pcam;
  }
  void use_orthographic_camera() {
    m_active_cam = &m_ocam;
  }
  void use_fps_controller() {
    m_controller = std::make_unique<fps_camera_controller<Real>>(this);
  }
  auto perspective_camera() -> auto& {
    return m_pcam;
  }
  auto perspective_camera() const -> auto const& {
    return m_pcam;
  }
  auto orthographic_camera() -> auto& {
    return m_ocam;
  }
  auto orthographic_camera() const -> auto const& {
    return m_ocam;
  }
  auto projection_matrix() const {
    return m_active_cam->projection_matrix();
  }
  auto transform_matrix() const -> mat4 {
    return look_at_matrix(m_active_cam->eye(), m_active_cam->lookat(),
                          m_active_cam->up());
  }
  auto view_matrix() const {
    return inv(transform_matrix());
  }
  auto set_eye(Real const x, Real const y, Real const z) {
    m_pcam.set_eye(x, y, z);
    m_ocam.set_eye(x, y, z);
  }
  auto set_eye(vec<Real, 3> const& eye) {
    m_pcam.set_eye(eye);
    m_ocam.set_eye(eye);
  }
  auto set_lookat(Real const x, Real const y, Real const z) {
    m_pcam.set_lookat(x, y, z);
    m_ocam.set_lookat(x, y, z);
  }
  auto set_lookat(vec<Real, 3> const& lookat) {
    m_pcam.set_lookat(lookat);
    m_ocam.set_lookat(lookat);
  }
  //------------------------------------------------------------------------------
  void on_key_pressed(yavin::key k) override {
    if (m_controller) {
      m_controller->on_key_pressed(k);
    }
  }
  void on_key_released(yavin::key k) override {
    if (m_controller) {
      m_controller->on_key_released(k);
    }
  }
  void on_button_pressed(yavin::button b) override {
    if (m_controller) {
      m_controller->on_button_pressed(b);
    }
  }
  void on_button_released(yavin::button b) override {
    if (m_controller) {
      m_controller->on_button_released(b);
    }
  }
  void on_mouse_motion(int x, int y) override {
    if (m_controller) {
      m_controller->on_mouse_motion(x, y);
    }
  }
  void on_resize(int w, int h) override {
    if (m_controller) {
      m_controller->on_resize(w, h);
    }
  }
};
//==============================================================================
template <typename Real>
struct fps_camera_controller : camera_controller_interface<Real> {
  size_t                    m_width, m_height;
  vec<float, 3>             m_eye, m_look_dir, m_up;
  float                     m_theta = M_PI, m_phi = M_PI / 2;
  perspective_camera<float> m_cam;
  std::chrono::time_point<std::chrono::system_clock> m_time =
      std::chrono::system_clock::now();
  int  m_mouse_pos_x, m_mouse_pos_y;
  bool m_middle_button_down = false;
  bool m_w_down             = false;
  bool m_s_down             = false;
  bool m_a_down             = false;
  bool m_d_down             = false;
  bool m_q_down             = false;
  bool m_e_down             = false;

  void on_key_pressed(yavin::key k) override {
    parent_t::on_key_pressed(k);
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
    }
  }
  //----------------------------------------------------------------------------
  void on_key_released(yavin::key k) override {
    parent_t::on_key_released(k);
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
    parent_t::on_button_pressed(b);
    if (b == yavin::BUTTON_MIDDLE) {
      m_middle_button_down = true;
    }
  }
  //----------------------------------------------------------------------------
  void on_button_released(yavin::button b) override {
    parent_t::on_button_released(b);
    if (b == yavin::BUTTON_MIDDLE) {
      m_middle_button_down = false;
    }
  }
  //----------------------------------------------------------------------------
  void on_mouse_motion(int x, int y) override {
    parent_t::on_mouse_motion(x, y);
    if (m_middle_button_down) {
      int offset_x = x - m_mouse_pos_x;
      int offset_y = y - m_mouse_pos_y;
      m_theta -= offset_x * 0.01f;
      m_phi      = std::min<float>(M_PI - 0.3f,
                              std::max(0.3f, m_phi + offset_y * 0.01f));
      m_look_dir = vec{
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
    parent_t::on_resize(w, h);
    m_width  = w;
    m_height = h;
    m_cam.set_resolution(w, h);
  }
  //----------------------------------------------------------------------------
  void update(std::chrono::duration<double> const& dt) {
    auto ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dt).count());

    m_time = std::chrono::system_clock::now();
    if (m_w_down) {
      m_eye += m_look_dir / ms;
    }
    if (m_s_down) {
      m_eye -= m_look_dir / ms;
    }
    if (m_q_down) {
      m_eye(1) += 1 / ms;
    }
    if (m_e_down) {
      m_eye(1) -= 1 / ms;
    }
    if (m_a_down) {
      auto const right = cross(vec{0, 1, 0}, -m_look_dir);
      m_eye -= right / ms;
    }
    if (m_d_down) {
      auto const right = cross(vec{0, 1, 0}, -m_look_dir);
      m_eye += right / ms;
    }
    m_cam.look_at(m_eye, m_eye + m_look_dir + 0.1);
    f(dt);
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
