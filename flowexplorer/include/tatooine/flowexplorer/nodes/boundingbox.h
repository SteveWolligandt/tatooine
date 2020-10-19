#ifndef TATOOINE_FLOWEXPLORER_NODES_BOUNDINGBOX_H
#define TATOOINE_FLOWEXPLORER_NODES_BOUNDINGBOX_H
//==============================================================================
#include <tatooine/boundingbox.h>
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/flowexplorer/renderable.h>
#include <yavin/imgui.h>
#include <yavin/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <size_t N>
struct boundingbox : tatooine::boundingbox<double, N>,
                     renderable<boundingbox<N>> {
  using this_t   = boundingbox<N>;
  using parent_t = tatooine::boundingbox<double, N>;
  using gpu_vec  = vec<float, N>;
  using vbo_t    = yavin::vertexbuffer<gpu_vec>;
  using parent_t::max;
  using parent_t::min;
  static constexpr std::string_view bb2_name = "boundingbox2d";
  static constexpr std::string_view bb3_name = "boundingbox3d";
  //============================================================================
  yavin::indexeddata<vec<float, N>> m_gpu_data;
  line_shader                       m_shader;
  int                               m_linewidth = 1;
  std::array<GLfloat, 4>            m_color{0.0f, 0.0f, 0.0f, 1.0f};
  //============================================================================
  boundingbox(flowexplorer::scene& s) : renderable<boundingbox>{"Bounding Box", s} {
    this->template insert_output_pin<this_t>("Out");
    create_indexed_data();
  }
  boundingbox(const boundingbox&)     = default;
  boundingbox(boundingbox&&) noexcept = default;
  auto operator=(const boundingbox&) -> boundingbox& = default;
  auto operator=(boundingbox&&) noexcept -> boundingbox& = default;
  //============================================================================
  template <typename Real0, typename Real1>
  constexpr boundingbox(vec<Real0, N>&& min, vec<Real1, N>&& max,
                        flowexplorer::scene& s) noexcept
      : parent_t{std::move(min), std::move(max)},
        renderable<boundingbox>{"Bounding Box", s} {
    this->template insert_output_pin<this_t>("Out");
    create_indexed_data();
  }
  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  constexpr boundingbox(const vec<Real0, N>& min, const vec<Real1, N>& max,
                        flowexplorer::scene& s)
      : parent_t{min, max}, renderable<boundingbox>{"Bounding Box", s} {
    this->template insert_output_pin<this_t>("Out");
    create_indexed_data();
  }
  //----------------------------------------------------------------------------
  template <typename Tensor0, typename Tensor1, typename Real0, typename Real1>
  constexpr boundingbox(const base_tensor<Tensor0, Real0, N>& min,
                        const base_tensor<Tensor1, Real1, N>& max,
                        flowexplorer::scene&                  s)
      : parent_t{min, max}, renderable<boundingbox>{"Bounding Box", s} {
    this->template insert_output_pin<this_t>("Out");
    create_indexed_data();
  }
  //============================================================================
  void render(mat<float, 4, 4> const& projection_matrix,
              mat<float, 4, 4> const& view_matrix) override {
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
    if constexpr (N == 3) {
      ImGui::DragDouble3("min", this->min().data_ptr(), 0.1);
      ImGui::DragDouble3("max", this->max().data_ptr(), 0.1);
    } else if constexpr (N == 2) {
      ImGui::DragDouble2("min", this->min().data_ptr(), 0.1);
      ImGui::DragDouble2("max", this->max().data_ptr(), 0.1);
    }
    ImGui::DragInt("line size", &m_linewidth, 1, 1, 10);
    ImGui::ColorEdit4("line color", m_color.data());
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
  auto is_transparent() const -> bool override {
    return m_color[3] < 1;
  }
  auto serialize() const -> toml::table override {
    toml::table serialization;
    if constexpr (N == 2) {
      serialization.insert("min", toml::array{min(0), min(1)});
      serialization.insert("max", toml::array{max(0), max(1)});
    } else if constexpr (N == 3) {
      serialization.insert("min", toml::array{min(0), min(1), min(2)});
      serialization.insert("max", toml::array{max(0), max(1), min(2)});
    }
    serialization.insert(
        "color", toml::array{m_color[0], m_color[1], m_color[2], m_color[3]});
    serialization.insert("linewidth", m_linewidth);

    return serialization;
  }
  void deserialize(toml::table const& serialization) override {
    auto const& serialized_min   = *serialization["min"].as_array();
    auto const& serialized_max   = *serialization["max"].as_array();
    auto const& serialized_color = *serialization["color"].as_array();

    min(0) = serialized_min[0].as_floating_point()->get();
    min(1) = serialized_min[1].as_floating_point()->get();
    max(0) = serialized_max[0].as_floating_point()->get();
    max(1) = serialized_max[1].as_floating_point()->get();
    if constexpr (N == 3) {
      min(2) = serialized_min[2].as_floating_point()->get();
      max(2) = serialized_max[2].as_floating_point()->get();
    }
    m_color[0]  = serialized_color[0].as_floating_point()->get();
    m_color[1]  = serialized_color[1].as_floating_point()->get();
    m_color[2]  = serialized_color[2].as_floating_point()->get();
    m_color[3]  = serialized_color[3].as_floating_point()->get();
    m_linewidth = serialization["linewidth"].as_integer()->get();
  }
  constexpr auto node_type_name() const -> std::string_view override {
    if constexpr (N == 2) {
      return bb2_name;
    } else if constexpr (N == 3) {
      return bb3_name;
    }
  }
};
using boundingbox2d = boundingbox<2>;
using boundingbox3d = boundingbox<3>;
REGISTER_NODE(boundingbox2d);
REGISTER_NODE(boundingbox3d);
//==============================================================================
// deduction guides
//==============================================================================
template <typename Real0, typename Real1, size_t N>
boundingbox(const vec<Real0, N>&, const vec<Real1, N>&, flowexplorer::scene&)
    -> boundingbox<N>;
//------------------------------------------------------------------------------
template <typename Real0, typename Real1, size_t N>
boundingbox(vec<Real0, N>&&, vec<Real1, N>&&, flowexplorer::scene&)
    -> boundingbox<N>;
//------------------------------------------------------------------------------
template <typename Tensor0, typename Tensor1, typename Real0, typename Real1,
          size_t N>
boundingbox(base_tensor<Tensor0, Real0, N>&&, base_tensor<Tensor1, Real1, N>&&,
            flowexplorer::scene&) -> boundingbox<N>;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
