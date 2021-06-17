#ifndef TATOOINE_RENDERING_CAMERA_CONTROLLER_H
#define TATOOINE_RENDERING_CAMERA_CONTROLLER_H
//==============================================================================
#include <tatooine/rendering/orthographic_camera.h>
#include <tatooine/rendering/perspective_camera.h>
#include <yavin/window_listener.h>

#include <chrono>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename Real>
struct camera_controller;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Real>
struct fps_camera_controller;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Real>
struct orthographic_camera_controller;
//==============================================================================
template <typename Real>
struct camera_controller_interface : yavin::window_listener {
 private:
  camera_controller<Real>* m_controller;
  //----------------------------------------------------------------------------
 public:
  camera_controller_interface(camera_controller<Real>* controller)
      : m_controller{controller} {}
  virtual ~camera_controller_interface() = default;

  auto perspective_camera() -> auto& {
    return m_controller->perspective_camera();
  }
  auto perspective_camera() const -> auto const& {
    return m_controller->perspective_camera();
  }
  auto orthographic_camera() -> auto& {
    return m_controller->orthographic_camera();
  }
  auto orthographic_camera() const -> auto const& {
    return m_controller->orthographic_camera();
  }
  void look_at(vec3 const& eye, vec3 const& lookat,
               vec3 const& up = {0, 1, 0}) {
    m_controller->look_at(eye, lookat, up);
  }
  auto controller() -> auto& {
    return *m_controller;
  }
  auto controller() const -> auto const& {
    return *m_controller;
  }
  //----------------------------------------------------------------------------
  virtual auto type() const -> std::type_info const& = 0;
  //----------------------------------------------------------------------------
  virtual void update(std::chrono::duration<double> const& /*dt*/) {}
};
//==============================================================================
template <typename Real>
struct camera_controller : yavin::window_listener {
  friend struct camera_controller_interface<Real>;
  class perspective_camera<Real>                     m_pcam;
  class orthographic_camera<Real>                    m_ocam;
  camera<Real>*                                      m_active_cam;
  std::unique_ptr<camera_controller_interface<Real>> m_controller;
  //============================================================================
  camera_controller(size_t const res_x, size_t const res_y)
      : m_pcam{{Real(0), Real(0), Real(0)},
               {Real(0), Real(0), Real(-1)},
               {Real(0), Real(1), Real(0)},
               60,
               0.01,
               100,
               res_x,
               res_y},
        m_ocam{{Real(0), Real(0), Real(0)},
               {Real(0), Real(0), Real(-1)},
               {Real(0), Real(1), Real(0)},
               5,
               -100,
               100,
               res_x,
               res_y},
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
  void use_orthographic_controller() {
    m_controller = std::make_unique<orthographic_camera_controller<Real>>(this);
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
  auto controller() const -> auto const& { return *m_controller; }
  auto projection_matrix() const {
    return m_active_cam->projection_matrix();
  }
  auto transform_matrix() const -> mat4 {
    return look_at_matrix(m_active_cam->eye(), m_active_cam->lookat(),
                          m_active_cam->up());
  }
  auto view_matrix() const {
    return *inv(transform_matrix());
  }
  auto eye() const -> auto const& {
    return m_active_cam->eye();
  }
  auto set_eye(Real const x, Real const y, Real const z) {
    m_pcam.set_eye(x, y, z);
    m_ocam.set_eye(x, y, z);
  }
  auto set_eye(vec<Real, 3> const& eye) {
    m_pcam.set_eye(eye);
    m_ocam.set_eye(eye);
  }
  auto lookat() const -> auto const& {
    return m_active_cam->lookat();
  }
  auto set_lookat(Real const x, Real const y, Real const z) {
    m_pcam.set_lookat(x, y, z);
    m_ocam.set_lookat(x, y, z);
  }
  auto set_lookat(vec<Real, 3> const& lookat) {
    m_pcam.set_lookat(lookat);
    m_ocam.set_lookat(lookat);
  }
  auto up() const -> auto const& {
    return m_active_cam->up();
  }
  auto set_up(Real const x, Real const y, Real const z) {
    m_pcam.set_up(x, y, z);
    m_ocam.set_up(x, y, z);
  }
  auto set_up(vec<Real, 3> const& up) {
    m_pcam.set_up(up);
    m_ocam.set_up(up);
  }
  auto look_at(vec3 const& eye, vec3 const& lookat,
               vec3 const& up = {0, 1, 0}) {
    m_pcam.look_at(eye, lookat, up);
    m_ocam.look_at(eye, lookat, up);
  }
  auto plane_width() {
    return m_active_cam->plane_width();
  }
  auto plane_height() {
    return m_active_cam->plane_height();
  }
  auto ray(int x, int y) const { return m_active_cam->ray(x, y); }
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
  void on_cursor_moved(double x, double y) override {
    if (m_controller) {
      m_controller->on_cursor_moved(x, y);
    }
  }
  void on_wheel_up() override {
    if (m_controller) {
      m_controller->on_wheel_up();
    }
  }
  void on_wheel_down() override {
    if (m_controller) {
      m_controller->on_wheel_down();
    }
  }
  void on_wheel_left() override {
    if (m_controller) {
      m_controller->on_wheel_left();
    }
  }
  void on_wheel_right() override {
    if (m_controller) {
      m_controller->on_wheel_right();
    }
  }
  void on_resize(int w, int h) override {
    m_pcam.set_resolution(w, h);
    m_ocam.set_resolution(w, h);
    if (m_controller) {
      m_controller->on_resize(w, h);
    }
  }
  void update(std::chrono::duration<double> const& dt) {
    if (m_controller) {
      m_controller->update(dt);
    }
  }
};
//==============================================================================
template <typename Real>
struct fps_camera_controller : camera_controller_interface<Real> {
  using this_t = fps_camera_controller<Real>;
  using parent_t = camera_controller_interface<Real>;
  using parent_t::controller;

  Real         m_theta = M_PI / 2, m_phi = M_PI / 2;
  vec<Real, 3> m_look_dir;
  double       m_mouse_pos_x, m_mouse_pos_y;
  bool         m_right_button_down = false;
  bool         m_w_down            = false;
  bool         m_s_down            = false;
  bool         m_a_down            = false;
  bool         m_d_down            = false;
  bool         m_q_down            = false;
  bool         m_e_down            = false;
  //----------------------------------------------------------------------------
  fps_camera_controller(camera_controller<Real>* controller)
      : camera_controller_interface<Real>{controller} {
    m_look_dir = {std::sin(m_phi) * std::sin(m_theta),
                  std::cos(m_phi),
                  std::sin(m_phi) * std::cos(m_theta)};
    controller->look_at(controller->eye(), controller->eye() + m_look_dir);
  }
  virtual ~fps_camera_controller() = default;
  //----------------------------------------------------------------------------
  auto on_key_pressed(yavin::key k) -> void override {
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
    if (b == yavin::BUTTON_RIGHT) {
      m_right_button_down = true;
    }
  }
  //----------------------------------------------------------------------------
  void on_button_released(yavin::button b) override {
    if (b == yavin::BUTTON_RIGHT) {
      m_right_button_down = false;
    }
  }
  //----------------------------------------------------------------------------
  void on_cursor_moved(double x, double y) override {
    if (m_right_button_down) {
      auto offset_x = std::ceil(x) - m_mouse_pos_x;
      auto offset_y = std::ceil(y) - m_mouse_pos_y;
      m_theta -= offset_x * Real(0.01);
      m_phi = std::min<Real>(
          M_PI - Real(0.3),
          std::max<Real>(Real(0.3), m_phi + offset_y * Real(0.01)));
      m_look_dir = {std::sin(m_phi) * std::sin(m_theta),
                    std::cos(m_phi),
                    std::sin(m_phi) * std::cos(m_theta)};
    }
    m_mouse_pos_x = std::ceil(x);
    m_mouse_pos_y = std::ceil(y);
    controller().look_at(controller().eye(), controller().eye() + m_look_dir);
  }
  //----------------------------------------------------------------------------
  void update(std::chrono::duration<double> const& dt) override {
    auto const ms = static_cast<Real>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dt).count());

    if (m_w_down) {
      auto const new_eye = controller().eye() + m_look_dir * ms / 1000;
      controller().look_at(new_eye, new_eye + m_look_dir);
    }
    if (m_s_down) {
      auto const new_eye = controller().eye() - m_look_dir * ms / 1000;
      controller().look_at(new_eye, new_eye + m_look_dir);
    }
    if (m_q_down) {
      auto const& old_eye = controller().eye();
      auto const new_eye = vec{old_eye(0), old_eye(1) + 1 * ms / 1000, old_eye(2)};
      controller().look_at(new_eye, new_eye + m_look_dir);
    }
    if (m_e_down) {
      auto const& old_eye = controller().eye();
      auto const new_eye = vec{old_eye(0), old_eye(1) - 1 * ms / 1000, old_eye(2)};
      controller().look_at(new_eye, new_eye + m_look_dir);
    }
    if (m_a_down) {
      auto const right   = cross(vec{0, 1, 0}, -m_look_dir);
      auto const new_eye = controller().eye() - right * ms / 1000;
      controller().look_at(new_eye, new_eye + m_look_dir);
    }
    if (m_d_down) {
      auto const right   = cross(vec{0, 1, 0}, -m_look_dir);
      auto const new_eye = controller().eye() + right * ms / 1000;
      controller().look_at(new_eye, new_eye + m_look_dir);
    }
  }
  //----------------------------------------------------------------------------
  auto type() const -> std::type_info const& override { return typeid(this_t); }
};
//==============================================================================
template <typename Real>
struct orthographic_camera_controller : camera_controller_interface<Real> {
  using this_t = orthographic_camera_controller<Real>;
  using parent_t = camera_controller_interface<Real>;
  using parent_t::controller;

  //============================================================================
  // members
  //============================================================================
  int  m_mouse_pos_x, m_mouse_pos_y;
  bool m_right_button_down = false;

  //============================================================================
  // ctor
  //============================================================================
  orthographic_camera_controller(camera_controller<Real>* controller)
      : camera_controller_interface<Real>{controller} {
    auto new_eye = controller->eye();
    new_eye(2)   = 0;
    controller->look_at(new_eye, new_eye + vec{0, 0, -1});
  }
  virtual ~orthographic_camera_controller() = default;

  //============================================================================
  // methods
  //============================================================================
  void on_button_pressed(yavin::button b) override {
    if (b == yavin::BUTTON_RIGHT) {
      m_right_button_down = true;
    }
  }
  //----------------------------------------------------------------------------
  void on_button_released(yavin::button b) override {
    if (b == yavin::BUTTON_RIGHT) {
      m_right_button_down = false;
    }
  }
  //----------------------------------------------------------------------------
  auto on_cursor_moved(double x, double y) -> void override {
    if (m_right_button_down) {
      auto offset_x = std::ceil(x) - m_mouse_pos_x;
      auto offset_y = std::ceil(y) - m_mouse_pos_y;
      auto new_eye  = controller().eye();
      new_eye(0) -= static_cast<Real>(offset_x) *
                    controller().orthographic_camera().aspect_ratio() /
                    controller().orthographic_camera().plane_width() *
                    controller().orthographic_camera().height();
      new_eye(1) += static_cast<Real>(offset_y) /
                    controller().orthographic_camera().plane_height() *
                    controller().orthographic_camera().height();
      new_eye(2) = 0;
      controller().look_at(new_eye, new_eye + vec{0, 0, -1});
    }
    m_mouse_pos_x = std::ceil(x);
    m_mouse_pos_y = std::ceil(y);
  }
  //----------------------------------------------------------------------------
  auto on_wheel_down() -> void override {
    controller().orthographic_camera().setup(
        controller().eye(), controller().lookat(), controller().up(),
        controller().orthographic_camera().height() / 0.9,
        controller().orthographic_camera().near(),
        controller().orthographic_camera().far(),
        controller().plane_width(), controller().plane_height());
  }
  //----------------------------------------------------------------------------
  auto on_wheel_up() -> void override {
    controller().orthographic_camera().setup(
        controller().eye(), controller().lookat(), controller().up(),
        controller().orthographic_camera().height() * 0.9,
        controller().orthographic_camera().near(),
        controller().orthographic_camera().far(),
        controller().plane_width(), controller().plane_height());
  }
  //----------------------------------------------------------------------------
  auto type() const -> std::type_info const& override { return typeid(this_t); }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
