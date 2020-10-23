#ifndef TATOOINE_FLOWEXPLORER_NODES_POSITION_H
#define TATOOINE_FLOWEXPLORER_NODES_POSITION_H
//==============================================================================
#include <tatooine/flowexplorer/point_shader.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/vec.h>
#include <yavin/imgui.h>
#include <yavin/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <size_t N>
struct position : tatooine::vec<double, N>, renderable<position<N>> {
  using this_t   = position<N>;
  using parent_t = tatooine::vec<double, N>;
  using gpu_vec  = vec<GLfloat, 3>;
  using vbo_t    = yavin::vertexbuffer<gpu_vec>;
  //============================================================================
  yavin::indexeddata<gpu_vec> m_gpu_data;
  point_shader                m_shader;
  int                         m_pointsize = 1;
  std::array<GLfloat, 4>      m_color{0.0f, 0.0f, 0.0f, 1.0f};
  //============================================================================
  constexpr position(position const&)     = default;
  constexpr position(position&&) noexcept = default;
  constexpr auto operator=(position const&) -> position& = default;
  constexpr auto operator=(position&&) noexcept -> position& = default;
  //============================================================================
  constexpr position(flowexplorer::scene& s) : renderable<position>{"Position", s} {
    this->template insert_output_pin<this_t>("Out");
    create_indexed_data();
  }
  //----------------------------------------------------------------------------
  template <typename Real>
  constexpr position(flowexplorer::scene& s, vec<Real, N>&& pos) noexcept
      : parent_t{std::move(pos)}, renderable<position>{"Position", s} {
    this->template insert_output_pin<this_t>("Out");
    create_indexed_data();
  }
  //----------------------------------------------------------------------------
  template <typename Real>
  constexpr position(flowexplorer::scene& s, vec<Real, N> const& pos)
      : parent_t{pos}, renderable<position>{"Position", s} {
    this->template insert_output_pin<this_t>("Out");
    create_indexed_data();
  }
  //============================================================================
  void render(mat<GLfloat, 4, 4> const& projection_matrix,
              mat<GLfloat, 4, 4> const& view_matrix) override {
    set_vbo_data();
    m_shader.bind();
    m_shader.set_color(m_color[0], m_color[1], m_color[2], m_color[3]);
    m_shader.set_projection_matrix(projection_matrix);
    m_shader.set_modelview_matrix(view_matrix);
    yavin::gl::point_size(m_pointsize);
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
  void create_indexed_data() {
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
  auto pos()       -> vec<double, N>& { return *this; }
  auto pos() const -> vec<double, N> const& { return *this; }
  //----------------------------------------------------------------------------
  auto point_size()       -> auto& { return m_pointsize; }
  auto point_size() const -> auto const& { return m_pointsize; }
  //----------------------------------------------------------------------------
  auto color()       -> auto& { return m_color; }
  auto color() const -> auto const& { return m_color; }
};
using position2 = position<2>;
using position3 = position<3>;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
REGISTER_NODE(tatooine::flowexplorer::nodes::position2,
              TATOOINE_REFLECTION_INSERT_METHOD(position, pos()),
              TATOOINE_REFLECTION_INSERT_GETTER(point_size),
              TATOOINE_REFLECTION_INSERT_GETTER(color))
REGISTER_NODE(tatooine::flowexplorer::nodes::position3,
              TATOOINE_REFLECTION_INSERT_METHOD(position, pos()),
              TATOOINE_REFLECTION_INSERT_GETTER(point_size),
              TATOOINE_REFLECTION_INSERT_GETTER(color))
#endif
