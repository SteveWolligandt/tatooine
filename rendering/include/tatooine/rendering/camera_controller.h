#ifndef TATOOINE_RENDERING_CAMERA_CONTROLLER_H
#define TATOOINE_RENDERING_CAMERA_CONTROLLER_H
//==============================================================================
#include <tatooine/rendering/orthographic_camera.h>
#include <tatooine/rendering/perspective_camera.h>
#include <tatooine/gl/window_listener.h>

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
struct camera_controller_interface : gl::window_listener {
  using vec3 = vec<Real, 3>;
  using vec4 = vec<Real, 4>;
  using mat3 = mat<Real, 3, 3>;
  using mat4 = mat<Real, 4, 4>;
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
struct camera_controller : gl::window_listener {
  using vec3 = vec<Real, 3>;
  using vec4 = vec<Real, 4>;
  using mat3 = mat<Real, 3, 3>;
  using mat4 = mat<Real, 4, 4>;
  friend struct camera_controller_interface<Real>;
  struct perspective_camera<Real>                    m_pcam;
  struct orthographic_camera<Real>                   m_ocam;
  camera_interface<Real>*                            m_active_cam;
  std::unique_ptr<camera_controller_interface<Real>> m_controller;
  Real                                               m_orthographic_height = 1;
  //============================================================================
  camera_controller(size_t const res_x, size_t const res_y)
      : m_pcam{{Real(0), Real(0), Real(0)},
               {Real(0), Real(0), Real(1)},
               60,
               0.01,
               100,
               res_x,
               res_y},
        m_ocam{{Real(0), Real(0), Real(0)},
               {Real(0), Real(0), Real(-1)},
               m_orthographic_height,
               -100,
               100,
               res_x,
               res_y},
        m_active_cam{&m_pcam} {
    use_fps_controller();
  }
  //============================================================================
  auto set_orthographic_height(Real const h) {
    m_orthographic_height = h;
    m_ocam.set_projection_matrix(m_orthographic_height);
  }
  auto orthographic_height() { return m_orthographic_height; }
  auto active_camera() const -> auto const& { return *m_active_cam; }
  auto unproject(Vec2<Real> const& x) {
    return m_active_cam->unproject(x);
  }
  auto unproject(Vec4<Real> const& x) {
    return m_active_cam->unproject(x);
  }
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
  auto orthographic_camera() const -> auto const& { return m_ocam; }
  auto controller() const -> auto const& { return *m_controller; }
  auto projection_matrix() const { return m_active_cam->projection_matrix(); }
  auto transform_matrix() const -> mat4 {
    return m_active_cam->transform_matrix();
  }
  auto view_matrix() const { return m_active_cam->view_matrix(); }
  auto eye() const { return m_active_cam->eye(); }
  auto right_direction() const { return m_active_cam->right_direction(); }
  auto up_direction() const { return m_active_cam->up_direction(); }
  auto view_direction() const { return m_active_cam->view_direction(); }
  auto look_at(vec3 const& eye, vec3 const& lookat,
               vec3 const& up = {0, 1, 0}) {
    m_pcam.look_at(eye, lookat, up);
    m_ocam.look_at(eye, lookat, up);
  }
  auto look_at(vec3 const& eye, arithmetic auto const pitch,
               arithmetic auto const yaw) {
    m_pcam.look_at(eye, pitch, yaw);
    m_ocam.look_at(eye, pitch, yaw);
  }
  auto plane_width() { return m_active_cam->plane_width(); }
  auto plane_height() {
    return m_active_cam->plane_height();
  }
  auto ray(int x, int y) const { return m_active_cam->ray(x, y); }
  //------------------------------------------------------------------------------
  void on_key_pressed(gl::key k) override {
    if (m_controller) {
      m_controller->on_key_pressed(k);
    }
  }
  void on_key_released(gl::key k) override {
    if (m_controller) {
      m_controller->on_key_released(k);
    }
  }
  void on_button_pressed(gl::button b) override {
    if (m_controller) {
      m_controller->on_button_pressed(b);
    }
  }
  void on_button_released(gl::button b) override {
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
    m_pcam.set_projection_matrix(60, 0.001, 1000);
    m_ocam.set_projection_matrix(m_orthographic_height);
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
  using vec3 = vec<Real, 3>;
  using vec4 = vec<Real, 4>;
  using mat3 = mat<Real, 3, 3>;
  using mat4 = mat<Real, 4, 4>;
  using this_type = fps_camera_controller<Real>;
  using parent_type = camera_controller_interface<Real>;
  using parent_type::controller;

  enum buttons : std::uint8_t { w = 1, a = 2, s = 4, d = 8, q = 16, e = 32 };

  std::uint8_t m_buttons_down = 0;
  double       m_mouse_pos_x, m_mouse_pos_y;
  bool         m_right_button_down = false;
  bool         m_shift_down        = false;
  bool         m_ctrl_down         = false;
  //----------------------------------------------------------------------------
  fps_camera_controller(camera_controller<Real>* controller)
      : camera_controller_interface<Real>{controller} {
    //controller->look_at(controller->eye(), controller->eye() + vec{0, 0, 1});
  }
  virtual ~fps_camera_controller() = default;
  //----------------------------------------------------------------------------
  auto on_key_pressed(gl::key k) -> void override {
    if (k == gl::KEY_CTRL_L || k == gl::KEY_CTRL_R) {
      m_ctrl_down = true;
    }
    if (k == gl::KEY_SHIFT_L || k == gl::KEY_SHIFT_R) {
      m_shift_down = true;
    }
    if (k == gl::KEY_W) {
      m_buttons_down = m_buttons_down | buttons::w;
    }
    if (k == gl::KEY_S) {
      m_buttons_down = m_buttons_down | buttons::s;
    }
    if (k == gl::KEY_A) {
      m_buttons_down = m_buttons_down | buttons::a;
    }
    if (k == gl::KEY_D) {
      m_buttons_down = m_buttons_down | buttons::d;
    }
    if (k == gl::KEY_Q) {
      m_buttons_down = m_buttons_down | buttons::q;
    }
    if (k == gl::KEY_E) {
      m_buttons_down = m_buttons_down | buttons::e;
    }
  }
  //----------------------------------------------------------------------------
  void on_key_released(gl::key k) override {
    if (k == gl::KEY_CTRL_L || k == gl::KEY_CTRL_R) {
      m_ctrl_down = false;
    }
    if (k == gl::KEY_SHIFT_L || k == gl::KEY_SHIFT_R) {
      m_shift_down = false;
    }
    if (k == gl::KEY_W) {
      m_buttons_down = m_buttons_down & ~buttons::w;
    }
    if (k == gl::KEY_S) {
      m_buttons_down = m_buttons_down & ~buttons::s;
    }
    if (k == gl::KEY_A) {
      m_buttons_down = m_buttons_down & ~buttons::a;
    }
    if (k == gl::KEY_D) {
      m_buttons_down = m_buttons_down & ~buttons::d;
    }
    if (k == gl::KEY_Q) {
      m_buttons_down = m_buttons_down & ~buttons::q;
    }
    if (k == gl::KEY_E) {
      m_buttons_down = m_buttons_down & ~buttons::e;
    }
  }
  //----------------------------------------------------------------------------
  void on_button_pressed(gl::button b) override {
    if (b == gl::button::right) {
      m_right_button_down = true;
    }
  }
  //----------------------------------------------------------------------------
  void on_button_released(gl::button b) override {
    if (b == gl::button::right) {
      m_right_button_down = false;
    }
  }
  //----------------------------------------------------------------------------
  void on_cursor_moved(double x, double y) override {
    if (m_right_button_down) {
      Real const offset_x = gcem::ceil(x) - m_mouse_pos_x;
      Real const offset_y = gcem::ceil(y) - m_mouse_pos_y;

      auto const old_view_dir = -controller().view_direction();
      auto       yaw          = gcem::atan2(old_view_dir(2), old_view_dir(0));
      auto       pitch        = gcem::asin(old_view_dir(1));

      yaw += offset_x * Real(0.001);
      pitch                = std::clamp<Real>(pitch + offset_y * Real(0.001),
                               -M_PI * 0.5 * 0.7, M_PI * 0.5 * 0.7);
      auto const cos_pitch = gcem::cos(pitch);
      auto const sin_pitch = gcem::sin(pitch);
      auto const cos_yaw   = gcem::cos(yaw);
      auto const sin_yaw   = gcem::sin(yaw);
      auto const eye       = controller().eye();
      auto const new_view_dir =
          vec{cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw};
      controller().look_at(eye, eye + normalize(new_view_dir));
    }                          
    m_mouse_pos_x = gcem::ceil(x);
    m_mouse_pos_y = gcem::ceil(y);
  }
  auto speed() const {
    if (m_shift_down) {
      return Real(1) / 250;
    }
    if (m_ctrl_down) {
      return Real(1) / 1000;
    }
    return Real(1) / 500;
  }
  //----------------------------------------------------------------------------
  void update(std::chrono::duration<double> const& dt) override {
    auto const look_dir = controller().view_direction();
    auto const right_dir      = controller().right_direction();
    auto       move_direction = vec3::zeros();
    if (m_buttons_down & buttons::w) {
      move_direction -= look_dir;
    }
    if (m_buttons_down & buttons::s) {
      move_direction += look_dir;
    }
    if (m_buttons_down & buttons::a) {
      move_direction -= right_dir;
    }
    if (m_buttons_down & buttons::d) {
      move_direction += right_dir;
    }
    if (m_buttons_down & buttons::q) {
      move_direction(1) += 1;
    }
    if (m_buttons_down & buttons::e) {
      move_direction(1) -= 1;
    }
    auto const passed_time = static_cast<Real>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dt).count());
    auto const new_eye =
        controller().eye() + normalize(move_direction) * passed_time * speed();
    controller().look_at(new_eye, new_eye - look_dir);

  }
  //----------------------------------------------------------------------------
  auto type() const -> std::type_info const& override {
    return typeid(this_type);
  }
};
//==============================================================================
template <typename Real>
struct orthographic_camera_controller : camera_controller_interface<Real> {
  using vec3        = vec<Real, 3>;
  using vec4        = vec<Real, 4>;
  using mat3        = mat<Real, 3, 3>;
  using mat4        = mat<Real, 4, 4>;
  using this_type   = orthographic_camera_controller<Real>;
  using parent_type = camera_controller_interface<Real>;
  using parent_type::controller;
  //============================================================================
  // members
  //============================================================================
  int  m_mouse_pos_x, m_mouse_pos_y;
  bool m_right_button_down = false;
  //============================================================================
  // ctor
  //============================================================================
  orthographic_camera_controller(camera_controller<Real>* controller)
      : camera_controller_interface<Real>{controller} {}
  virtual ~orthographic_camera_controller() = default;
  //============================================================================
  // methods
  //============================================================================
  void on_button_pressed(gl::button b) override {
    if (b == gl::button::right) {
      m_right_button_down = true;
    }
  }
  //----------------------------------------------------------------------------
  void on_button_released(gl::button b) override {
    if (b == gl::button::right) {
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
      new_eye(1) -= static_cast<Real>(offset_y) /
                    controller().orthographic_camera().plane_height() *
                    controller().orthographic_camera().height();
      std::cout << new_eye << '\n';
      this->look_at(new_eye, new_eye + vec{0, 0, -1});
    }
    m_mouse_pos_x = std::ceil(x);
    m_mouse_pos_y = std::ceil(y);
  }
  //----------------------------------------------------------------------------
  auto on_wheel_down() -> void override {
    controller().set_orthographic_height(controller().orthographic_height() / 0.9);
    //controller().orthographic_camera().setup(
    //    controller().eye(), controller().lookat(), controller().up(),
    //    controller().orthographic_camera().width() / 0.9,
    //    controller().orthographic_camera().height() / 0.9,
    //    controller().orthographic_camera().near(),
    //    controller().orthographic_camera().far(),
    //    controller().plane_width(), controller().plane_height());
  }
  //----------------------------------------------------------------------------
  auto on_wheel_up() -> void override {
    controller().set_orthographic_height(controller().orthographic_height() *
                                         0.9);
    //controller().orthographic_camera().setup(
    //    controller().eye(), controller().lookat(), controller().up(),
    //    controller().orthographic_camera().width() * 0.9,
    //    controller().orthographic_camera().height() * 0.9,
    //    controller().orthographic_camera().near(),
    //    controller().orthographic_camera().far(),
    //    controller().plane_width(), controller().plane_height());
  }
  //----------------------------------------------------------------------------
  auto type() const -> std::type_info const& override {
    return typeid(this_type);
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
