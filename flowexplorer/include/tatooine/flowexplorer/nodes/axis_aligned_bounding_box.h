#ifndef TATOOINE_FLOWEXPLORER_NODES_AXIS_ALIGNED_BOUNDING_BOX_H
#define TATOOINE_FLOWEXPLORER_NODES_AXIS_ALIGNED_BOUNDING_BOX_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/gl/imgui.h>
#include <tatooine/gl/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <size_t N>
struct axis_aligned_bounding_box
    : tatooine::axis_aligned_bounding_box<double, N>,
      renderable<axis_aligned_bounding_box<N>> {
  using this_t   = axis_aligned_bounding_box<N>;
  using parent_t = tatooine::axis_aligned_bounding_box<double, N>;
  using gpu_vec  = vec<float, N>;
  using vbo_t    = rendering::gl::vertexbuffer<gpu_vec>;
  using parent_t::max;
  using parent_t::min;
  //============================================================================
  rendering::gl::indexeddata<vec<float, N>> m_gpu_data;
  int                               m_line_width = 1;
  std::array<GLfloat, 4>            m_line_color{0.0f, 0.0f, 0.0f, 1.0f};
  //============================================================================
  axis_aligned_bounding_box(flowexplorer::scene& s)
      : tatooine::axis_aligned_bounding_box<double, N>{vec<double, N>{
                                                           tag::fill{-1}},
                                                       vec<double, N>{
                                                           tag::fill{1}}},
        renderable<axis_aligned_bounding_box>{"Axis Aligned Bounding Box", s,
                                              *this} {
    create_indexed_data();
  }
  axis_aligned_bounding_box(const axis_aligned_bounding_box&)     = default;
  axis_aligned_bounding_box(axis_aligned_bounding_box&&) noexcept = default;
  auto operator                     =(const axis_aligned_bounding_box&)
      -> axis_aligned_bounding_box& = default;
  auto operator                     =(axis_aligned_bounding_box&&) noexcept
      -> axis_aligned_bounding_box& = default;
  //============================================================================
  void render(mat4f const& P, mat4f const& V) override {
    set_vbo_data();
    auto& shader = line_shader::get();
    shader.bind();
    shader.set_color(m_line_color[0], m_line_color[1], m_line_color[2],
                     m_line_color[3]);
    shader.set_projection_matrix(P);
    shader.set_modelview_matrix(V);
    rendering::gl::line_width(m_line_width);
    m_gpu_data.draw_lines();
  }
  //============================================================================
  void set_vbo_data() {
    auto vbomap = m_gpu_data.vertexbuffer().map();
    if constexpr (N == 3) {
      vbomap[0] = gpu_vec{float(min(0)), float(min(1)), float(min(2))};
      vbomap[1] = gpu_vec{float(max(0)), float(min(1)), float(min(2))};
      vbomap[2] = gpu_vec{float(min(0)), float(max(1)), float(min(2))};
      vbomap[3] = gpu_vec{float(max(0)), float(max(1)), float(min(2))};
      vbomap[4] = gpu_vec{float(min(0)), float(min(1)), float(max(2))};
      vbomap[5] = gpu_vec{float(max(0)), float(min(1)), float(max(2))};
      vbomap[6] = gpu_vec{float(min(0)), float(max(1)), float(max(2))};
      vbomap[7] = gpu_vec{float(max(0)), float(max(1)), float(max(2))};
    } else if constexpr (N == 2) {
      vbomap[0] = gpu_vec{float(min(0)), float(min(1))};
      vbomap[1] = gpu_vec{float(max(0)), float(min(1))};
      vbomap[2] = gpu_vec{float(min(0)), float(max(1))};
      vbomap[3] = gpu_vec{float(max(0)), float(max(1))};
      vbomap[4] = gpu_vec{float(min(0)), float(min(1))};
      vbomap[5] = gpu_vec{float(max(0)), float(min(1))};
      vbomap[6] = gpu_vec{float(min(0)), float(max(1))};
      vbomap[7] = gpu_vec{float(max(0)), float(max(1))};
    }
  }
  //----------------------------------------------------------------------------
  void create_indexed_data() {
    m_gpu_data.vertexbuffer().resize(8);
    m_gpu_data.indexbuffer().resize(24);
    set_vbo_data();
    m_gpu_data.indexbuffer() = {0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6,
                                5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7};
  }
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool override { return m_line_color[3] < 1; }
  //----------------------------------------------------------------------------
  auto line_width() -> auto& { return m_line_width; }
  auto line_width() const { return m_line_width; }
  //----------------------------------------------------------------------------
  auto line_color() -> auto& { return m_line_color; }
  auto line_color() const -> auto const& { return m_line_color; }
};
//==============================================================================
using aabb2d = axis_aligned_bounding_box<2>;
using aabb3d = axis_aligned_bounding_box<3>;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::aabb2d,
    TATOOINE_REFLECTION_INSERT_GETTER(min),
    TATOOINE_REFLECTION_INSERT_GETTER(max),
    TATOOINE_REFLECTION_INSERT_GETTER(line_width),
    TATOOINE_REFLECTION_INSERT_GETTER(line_color));
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::aabb3d,
    TATOOINE_REFLECTION_INSERT_GETTER(min),
    TATOOINE_REFLECTION_INSERT_GETTER(max),
    TATOOINE_REFLECTION_INSERT_GETTER(line_width),
    TATOOINE_REFLECTION_INSERT_GETTER(line_color));
#endif
