#ifndef TATOOINE_FLOWEXPLORER_BOUNDINGBOX_H
#define TATOOINE_FLOWEXPLORER_BOUNDINGBOX_H
//==============================================================================
#include <tatooine/boundingbox.h>
#include <yavin/imgui.h>
#include <yavin/indexeddata.h>

#include "line_shader.h"
#include "renderable.h"
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
template <typename Real, size_t N>
struct boundingbox : tatooine::boundingbox<Real, N>, renderable {
  using parent_t = tatooine::boundingbox<Real, N>;
  using gpu_vec  = yavin::vec<float, N>;
  using vbo_t    = yavin::vertexbuffer<gpu_vec>;
  using parent_t::max;
  using parent_t::min;
  //============================================================================
  yavin::indexeddata<yavin::vec<float, N>> m_gpu_data;
  line_shader                              m_shader;
  int                                      m_linewidth = 1;
  std::array<GLfloat, 4>                   m_color{0.0f, 0.0f, 0.0f, 1.0f};
  //============================================================================
  boundingbox()                       = default;
  boundingbox(const boundingbox&)     = default;
  boundingbox(boundingbox&&) noexcept = default;
  boundingbox& operator=(const boundingbox&) = default;
  boundingbox& operator=(boundingbox&&) noexcept = default;
  //============================================================================
  template <typename Real0, typename Real1>
  constexpr boundingbox(vec<Real0, N>&& min, vec<Real1, N>&& max) noexcept
      : parent_t{std::move(min), std::move(max)} {
    create_indexed_data();
  }
  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  constexpr boundingbox(const vec<Real0, N>& min, const vec<Real1, N>& max)
      : parent_t{min, max} {
    create_indexed_data();
  }
  //----------------------------------------------------------------------------
  template <typename Tensor0, typename Tensor1, typename Real0, typename Real1>
  constexpr boundingbox(const base_tensor<Tensor0, Real0, N>& min,
                        const base_tensor<Tensor1, Real1, N>& max)
      : parent_t{min, max} {
    create_indexed_data();
  }
  //============================================================================
  void render(const yavin::mat4& projection_matrix,
              const yavin::mat4& view_matrix) override {
    set_vbo_data();
    m_shader.bind();
    m_shader.set_color(m_color[0], m_color[1], m_color[2], m_color[3]);
    m_shader.set_projection_matrix(projection_matrix);
    m_shader.set_modelview_matrix(view_matrix);
    yavin::gl::line_width(m_linewidth);
    m_gpu_data.draw_lines();
  }
  //----------------------------------------------------------------------------
  void draw_ui() override {
    ImGui::DragDouble3("min", this->min().data_ptr(), 0.1);
    ImGui::DragDouble3("max", this->max().data_ptr(), 0.1);
    ImGui::DragInt("line size", &m_linewidth, 1, 1, 10);
    ImGui::ColorEdit4("line color", m_color.data());
  }
  //----------------------------------------------------------------------------
  std::string name() const override { return "Bounding Box"; }
  //============================================================================
  void set_vbo_data() {
    auto vbomap = m_gpu_data.vertex_buffer().map();
    yavin::get<0>(vbomap[0]) =
        gpu_vec{float(min(0)), float(min(1)), float(min(2))};
    yavin::get<0>(vbomap[1]) =
        gpu_vec{float(max(0)), float(min(1)), float(min(2))};
    yavin::get<0>(vbomap[2]) =
        gpu_vec{float(min(0)), float(max(1)), float(min(2))};
    yavin::get<0>(vbomap[3]) =
        gpu_vec{float(max(0)), float(max(1)), float(min(2))};
    yavin::get<0>(vbomap[4]) =
        gpu_vec{float(min(0)), float(min(1)), float(max(2))};
    yavin::get<0>(vbomap[5]) =
        gpu_vec{float(max(0)), float(min(1)), float(max(2))};
    yavin::get<0>(vbomap[6]) =
        gpu_vec{float(min(0)), float(max(1)), float(max(2))};
    yavin::get<0>(vbomap[7]) =
        gpu_vec{float(max(0)), float(max(1)), float(max(2))};
  }
  //----------------------------------------------------------------------------
  void create_indexed_data() {
    m_gpu_data.vertex_buffer().resize(8);
    m_gpu_data.index_buffer().resize(24);
    set_vbo_data();
    m_gpu_data.index_buffer() = {0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6,
                                 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7};
    m_gpu_data.setup_vao();
  }
};
//==============================================================================
// deduction guides
//==============================================================================
template <typename Real0, typename Real1, size_t N>
boundingbox(const vec<Real0, N>&, const vec<Real1, N>&)
    -> boundingbox<promote_t<Real0, Real1>, N>;
//------------------------------------------------------------------------------
template <typename Real0, typename Real1, size_t N>
boundingbox(vec<Real0, N>&&, vec<Real1, N> &&)
    -> boundingbox<promote_t<Real0, Real1>, N>;
//------------------------------------------------------------------------------
template <typename Tensor0, typename Tensor1, typename Real0, typename Real1,
          size_t N>
boundingbox(base_tensor<Tensor0, Real0, N>&&, base_tensor<Tensor1, Real1, N> &&)
    -> boundingbox<promote_t<Real0, Real1>, N>;
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
