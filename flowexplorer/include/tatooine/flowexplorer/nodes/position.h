#ifndef TATOOINE_FLOWEXPLORER_NODES_POSITION_H
#define TATOOINE_FLOWEXPLORER_NODES_POSITION_H
//==============================================================================
#include <tatooine/flowexplorer/point_shader.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/vec.h>
#include <tatooine/gl/imgui.h>
#include <tatooine/gl/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <size_t N>
struct position : tatooine::vec<real_t, N>, renderable<position<N>> {
  using this_t   = position<N>;
  using parent_t = tatooine::vec<real_t, N>;
  using gpu_vec  = vec<GLfloat, 3>;
  using parent_t::at;
  //============================================================================
  gl::indexeddata<gpu_vec>   m_gpu_data;
  int                           m_pointsize = 1;
  std::array<GLfloat, 4>        m_color{0.0f, 0.0f, 0.0f, 1.0f};
  std::array<ui::input_pin*, N> m_input_pins;
  //============================================================================
  auto pos()       -> vec<real_t, N>& { return *this; }
  auto pos() const -> vec<real_t, N> const& { return *this; }
  //----------------------------------------------------------------------------
  auto point_size()       -> auto& { return m_pointsize; }
  auto point_size() const -> auto const& { return m_pointsize; }
  //----------------------------------------------------------------------------
  auto color()       -> auto& { return m_color; }
  auto color() const -> auto const& { return m_color; }
  //============================================================================
  constexpr position(position const&)     = default;
  constexpr position(position&&) noexcept = default;
  constexpr auto operator=(position const&) -> position& = default;
  constexpr auto operator=(position&&) noexcept -> position& = default;
  //============================================================================
  constexpr position(flowexplorer::scene& s)
      : renderable<position>{"Position", s,
                             *dynamic_cast<tatooine::vec<real_t, N>*>(this)} {
    create_indexed_data();
    for (size_t i = 0; i < N; ++i) {
      this->insert_input_pin_property_link(
          this->template insert_input_pin<real_t>(""), at(i));
    }
  }
  //============================================================================
  auto render(mat4f const& P, mat4f const& V)
      -> void override {
    set_vbo_data();
    point_shader::get().bind();
    point_shader::get().set_color(m_color[0], m_color[1], m_color[2], m_color[3]);
    point_shader::get().set_projection_matrix(P);
    point_shader::get().set_modelview_matrix(V);
    gl::point_size(m_pointsize);
    m_gpu_data.draw_points();
  }
  //============================================================================
  void set_vbo_data() {
    auto vbomap = m_gpu_data.vertexbuffer().map();
    vbomap[0]   = [this]() -> gpu_vec {
      if constexpr (N == 3) {
        return {static_cast<GLfloat>(this->at(0)),
                static_cast<GLfloat>(this->at(1)),
                static_cast<GLfloat>(this->at(2))};
      } else if constexpr (N == 2) {
        return {static_cast<GLfloat>(this->at(0)),
                static_cast<GLfloat>(this->at(1)),
                0.0f};
     }
    }();
  }
  //----------------------------------------------------------------------------
  auto create_indexed_data() -> void {
    m_gpu_data.vertexbuffer().resize(1);
    m_gpu_data.indexbuffer().resize(1);
    set_vbo_data();
    m_gpu_data.indexbuffer() = {0};
  }
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool override {
    return m_color[3] < 1;
  }
  //----------------------------------------------------------------------------
  auto on_mouse_drag(int offset_x, int offset_y) -> bool override {
    auto const P = this->scene().camera_controller().projection_matrix();
    auto const V = this->scene().camera_controller().view_matrix();

    auto x = [this]() -> vec4 {
      if constexpr (N == 2) {
        return {at(0), at(1), 0, 1};
      } else if constexpr (N == 3) {
        return {at(0), at(1), at(2), 1};
      }
    }();

    x    = P * (V * x);
    x(0) = (x(0) * 0.5 + 0.5) * (this->scene().window().width() - 1) + offset_x;
    x(1) = (x(1) * 0.5 + 0.5) * (this->scene().window().height() - 1) - offset_y;

    x(0) = x(0) / (this->scene().window().width() - 1) * 2 - 1;
    x(1) = x(1) / (this->scene().window().height() - 1) * 2 - 1;

    x = *inv(V) * *inv(P) * x;

    at(0) = x(0);
    at(1) = x(1);
    if constexpr (N == 3) {
      at(2) = x(2);
    }

    for (auto l : this->self_pin().links()) {
      l->input().node().on_property_changed();
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool override {
    bool changed = false;
    constexpr auto label   = [](size_t const i) {
      switch (i) {
        case 0:
          return "x";
        case 1:
          return "y";
        case 2:
          return "z";
        case 3:
          return "w";
        default:
          return "";
      }
    };
    for (size_t i = 0; i < N; ++i) {
      changed |= ImGui::DragDouble(label(i), &at(i), 0.01);
    }
    changed |= ImGui::DragInt("point size", &m_pointsize, 0, 20);
    changed |= ImGui::ColorEdit3("color", m_color.data());
    return changed;
  }
  //----------------------------------------------------------------------------
  auto check_intersection(ray<float, 3> const& r) const
      -> std::optional<intersection<double, 3>> override {
    if constexpr (N == 3) {
      geometry::sphere<real_t, 3> s{0.01, vec3{at(0), at(1), at(2)}};
      return s.check_intersection(ray<real_t, 3>{r});
    } else if constexpr (N == 2) {
      geometry::sphere<real_t, 3> s{0.01, vec3{at(0), at(1), 0}};
      return s.check_intersection(ray<real_t, 3>{r});
    }
    return {};
  }
};
using position2 = position<2>;
using position3 = position<3>;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::position2,
    TATOOINE_REFLECTION_INSERT_METHOD(position, pos()),
    TATOOINE_REFLECTION_INSERT_GETTER(point_size),
    TATOOINE_REFLECTION_INSERT_GETTER(color))
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::position3,
    TATOOINE_REFLECTION_INSERT_METHOD(position, pos()),
    TATOOINE_REFLECTION_INSERT_GETTER(point_size),
    TATOOINE_REFLECTION_INSERT_GETTER(color))
#endif
