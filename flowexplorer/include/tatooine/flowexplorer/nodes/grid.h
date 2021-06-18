#ifndef TATOOINE_FLOWEXPLORER_NODES_GRID_H
#define TATOOINE_FLOWEXPLORER_NODES_GRID_H
//==============================================================================
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/grid.h>
#include <tatooine/linspace.h>
#include <tatooine/rendering/gl/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <size_t N>
struct grid_renderer;
template <>
struct grid_renderer<2> : tatooine::non_uniform_grid<real_t, 2> {
  rendering::gl::indexeddata<vec3f> m_inner_geometry;
  rendering::gl::indexeddata<vec3f> m_outer_geometry;
  //----------------------------------------------------------------------------
  auto render(mat4f const& P, mat4f const& V) -> void {
    auto& shader = line_shader::get();
    shader.bind();
    shader.set_projection_matrix(P);
    shader.set_modelview_matrix(V);
    shader.set_color(0.8, 0.8, 0.8, 1);
    m_inner_geometry.draw_lines();
    shader.set_color(0, 0, 0, 1);
    m_outer_geometry.draw_lines();
  }
  //----------------------------------------------------------------------------
  auto update_geometry() {
    m_inner_geometry.clear();
    m_outer_geometry.clear();
    auto const& dim0 = this->template dimension<0>();
    auto const& dim1 = this->template dimension<1>();

    size_t j = 0;
    // left
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()),
              static_cast<float>(dim1.front()), 0.0f});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),
              0.0f});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // right
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),
              0.0f});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),
              0.0f});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // bottom
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()),
              static_cast<float>(dim1.front()), 0.0f});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),
              0.0f});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // top
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),
              0.0f});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),
              0.0f});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);

    j = 0;
    for (size_t i = 1; i < dim0.size() - 1; ++i) {
      auto const x = dim0[i];
      m_inner_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(x), static_cast<float>(dim1.front()), 0.0f});
      m_inner_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(x), static_cast<float>(dim1.back()), 0.0f});
      m_inner_geometry.indexbuffer().push_back(j++);
      m_inner_geometry.indexbuffer().push_back(j++);
    }
    for (size_t i = 1; i < dim1.size() - 1; ++i) {
      auto const x = dim1[i];
      m_inner_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(x), 0.0f});
      m_inner_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(x), 0.0f});
      m_inner_geometry.indexbuffer().push_back(j++);
      m_inner_geometry.indexbuffer().push_back(j++);
    }
  }
};
template <>
struct grid_renderer<3> : tatooine::non_uniform_grid<real_t, 3> {
  rendering::gl::indexeddata<vec3f> m_outer_geometry;
  rendering::gl::indexeddata<vec3f> m_left_geometry;
  rendering::gl::indexeddata<vec3f> m_right_geometry;
  rendering::gl::indexeddata<vec3f> m_top_geometry;
  rendering::gl::indexeddata<vec3f> m_bottom_geometry;
  rendering::gl::indexeddata<vec3f> m_front_geometry;
  rendering::gl::indexeddata<vec3f> m_back_geometry;
  //----------------------------------------------------------------------------
  auto render(mat4f const& P, mat4f const& V, vec3f const& eye) -> void {
    auto& shader = line_shader::get();
    shader.bind();
    shader.set_projection_matrix(P);
    shader.set_modelview_matrix(V);


    shader.set_color(0.8, 0.8, 0.8, 1);
    auto const&            dim0 = this->template dimension<0>();
    auto const&            dim1 = this->template dimension<1>();
    auto const&            dim2 = this->template dimension<2>();

    vec3f left{dim0.front(),
               (dim1.front() + dim1.back()) * 2,
               (dim2.front() + dim2.back()) * 2};
    vec3f right{dim0.back(),
                (dim1.front() + dim1.back()) * 2,
                (dim2.front() + dim2.back()) * 2};
    vec3f bottom{(dim0.front() + dim0.back()) * 2,
                 dim1.front(),
                 (dim2.front() + dim2.back()) * 2};
    vec3f top{(dim0.front() + dim0.back()) * 2,
               dim1.back(),
              (dim2.front() + dim2.back()) * 2};
    vec3f front{(dim0.front() + dim0.back()) * 2,
                (dim1.front() + dim1.back()) * 2,
                dim2.front()};
    vec3f back{(dim0.front() + dim0.back()) * 2,
               (dim1.front() + dim1.back()) * 2,
               dim2.back()};
    if (dot(normalize(left - eye), vec3f{-1, 0, 0}) < 0) {
      m_left_geometry.draw_lines();
    }
    if (dot(normalize(right - eye), vec3f{1, 0, 0}) < 0) {
      m_right_geometry.draw_lines();
    }
    if (dot(normalize(bottom - eye), vec3f{0, -1, 0}) < 0) {
      m_bottom_geometry.draw_lines();
    }
    if (dot(normalize(top - eye), vec3f{0, 1, 0}) < 0) {
      m_top_geometry.draw_lines();
    }
    if (dot(normalize(front - eye), vec3f{0, 0, -1}) < 0) {
      m_front_geometry.draw_lines();
    }
    if (dot(normalize(back - eye), vec3f{0, 0, 1}) < 0) {
      m_back_geometry.draw_lines();
    }

    shader.set_color(0, 0, 0, 1);
    m_outer_geometry.draw_lines();
  }
  //----------------------------------------------------------------------------
  auto update_geometry() {
    m_left_geometry.clear();
    m_right_geometry.clear();
    m_bottom_geometry.clear();
    m_top_geometry.clear();
    m_front_geometry.clear();
    m_back_geometry.clear();
    m_outer_geometry.clear();
    auto const& dim0 = this->template dimension<0>();
    auto const& dim1 = this->template dimension<1>();
    auto const& dim2 = this->template dimension<2>();

    size_t j = 0;
    // left
    m_outer_geometry.vertexbuffer().push_back(vec3f{
        static_cast<float>(dim0.front()), static_cast<float>(dim1.front()),
        static_cast<float>(dim2.front())});
    m_outer_geometry.vertexbuffer().push_back(vec3f{
        static_cast<float>(dim0.front()), static_cast<float>(dim1.front()),
        static_cast<float>(dim2.back())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // left
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.front())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.back())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // left
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),
              static_cast<float>(dim2.front())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),
              static_cast<float>(dim2.back())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // left
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.front())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.back())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);

    // left
    m_outer_geometry.vertexbuffer().push_back(vec3f{
        static_cast<float>(dim0.front()), static_cast<float>(dim1.front()),
        static_cast<float>(dim2.front())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.front())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // left
    m_outer_geometry.vertexbuffer().push_back(vec3f{
        static_cast<float>(dim0.front()), static_cast<float>(dim1.front()),
        static_cast<float>(dim2.back())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.back())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // left
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),
              static_cast<float>(dim2.front())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.front())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // left
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),
              static_cast<float>(dim2.back())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.back())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);

    // left
    m_outer_geometry.vertexbuffer().push_back(vec3f{
        static_cast<float>(dim0.front()), static_cast<float>(dim1.front()),
        static_cast<float>(dim2.front())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),
              static_cast<float>(dim2.front())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // left
    m_outer_geometry.vertexbuffer().push_back(vec3f{
        static_cast<float>(dim0.front()), static_cast<float>(dim1.front()),
        static_cast<float>(dim2.back())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),
              static_cast<float>(dim2.back())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // left
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.front())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.front())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);
    // left
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.back())});
    m_outer_geometry.vertexbuffer().push_back(
        vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),
              static_cast<float>(dim2.back())});
    m_outer_geometry.indexbuffer().push_back(j++);
    m_outer_geometry.indexbuffer().push_back(j++);

    j = 0;
    for (size_t i = 1; i < dim0.size() - 1; ++i) {
      auto const x = dim0[i];
      m_front_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(x), static_cast<float>(dim1.front()),
                dim2.front()});
      m_front_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(x), static_cast<float>(dim1.back()),
                dim2.front()});
      m_front_geometry.indexbuffer().push_back(j++);
      m_front_geometry.indexbuffer().push_back(j++);
    }
    for (size_t i = 1; i < dim1.size() - 1; ++i) {
      auto const x = dim1[i];
      m_front_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(x),
                dim2.front()});
      m_front_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(x),
                dim2.front()});
      m_front_geometry.indexbuffer().push_back(j++);
      m_front_geometry.indexbuffer().push_back(j++);
    }
    j = 0;
    for (size_t i = 1; i < dim0.size() - 1; ++i) {
      auto const x = dim0[i];
      m_back_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(x), static_cast<float>(dim1.front()),
                dim2.back()});
      m_back_geometry.vertexbuffer().push_back(vec3f{
          static_cast<float>(x), static_cast<float>(dim1.back()), dim2.back()});
      m_back_geometry.indexbuffer().push_back(j++);
      m_back_geometry.indexbuffer().push_back(j++);
    }
    for (size_t i = 1; i < dim1.size() - 1; ++i) {
      auto const x = dim1[i];
      m_back_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(x),
                dim2.back()});
      m_back_geometry.vertexbuffer().push_back(vec3f{
          static_cast<float>(dim0.back()), static_cast<float>(x), dim2.back()});
      m_back_geometry.indexbuffer().push_back(j++);
      m_back_geometry.indexbuffer().push_back(j++);
    }

    j = 0;
    for (size_t i = 1; i < dim0.size() - 1; ++i) {
      auto const x = dim0[i];
      m_bottom_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(x), static_cast<float>(dim1.front()),
                dim2.front()});
      m_bottom_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(x), static_cast<float>(dim1.front()),
                dim2.back()});
      m_bottom_geometry.indexbuffer().push_back(j++);
      m_bottom_geometry.indexbuffer().push_back(j++);
    }
    for (size_t i = 1; i < dim2.size() - 1; ++i) {
      auto const x = dim2[i];
      m_bottom_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), dim1.front(),
                static_cast<float>(x)});
      m_bottom_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), dim1.front(),
                static_cast<float>(x)});
      m_bottom_geometry.indexbuffer().push_back(j++);
      m_bottom_geometry.indexbuffer().push_back(j++);
    }
    j = 0;
    for (size_t i = 1; i < dim0.size() - 1; ++i) {
      auto const x = dim0[i];
      m_top_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(x), static_cast<float>(dim1.back()),
                dim2.front()});
      m_top_geometry.vertexbuffer().push_back(vec3f{
          static_cast<float>(x), static_cast<float>(dim1.back()), dim2.back()});
      m_top_geometry.indexbuffer().push_back(j++);
      m_top_geometry.indexbuffer().push_back(j++);
    }
    for (size_t i = 1; i < dim2.size() - 1; ++i) {
      auto const x = dim2[i];
      m_top_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), dim1.back(),
                static_cast<float>(x)});
      m_top_geometry.vertexbuffer().push_back(vec3f{
          static_cast<float>(dim0.back()), dim1.back(), static_cast<float>(x)});
      m_top_geometry.indexbuffer().push_back(j++);
      m_top_geometry.indexbuffer().push_back(j++);
    }

    j = 0;
    for (size_t i = 1; i < dim1.size() - 1; ++i) {
      auto const x = dim1[i];
      m_left_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(x),
                dim2.front()});
      m_left_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(x),
                dim2.back()});
      m_left_geometry.indexbuffer().push_back(j++);
      m_left_geometry.indexbuffer().push_back(j++);
    }
    for (size_t i = 1; i < dim2.size() - 1; ++i) {
      auto const x = dim2[i];
      m_left_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), dim1.front(),
                static_cast<float>(x)});
      m_left_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), dim1.back(),
                static_cast<float>(x)});
      m_left_geometry.indexbuffer().push_back(j++);
      m_left_geometry.indexbuffer().push_back(j++);
    }
    j = 0;
    for (size_t i = 1; i < dim1.size() - 1; ++i) {
      auto const x = dim1[i];
      m_right_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(x),
                dim2.front()});
      m_right_geometry.vertexbuffer().push_back(vec3f{
          static_cast<float>(dim0.back()), static_cast<float>(x), dim2.back()});
      m_right_geometry.indexbuffer().push_back(j++);
      m_right_geometry.indexbuffer().push_back(j++);
    }
    for (size_t i = 1; i < dim2.size() - 1; ++i) {
      auto const x = dim2[i];
      m_right_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), dim1.front(),
                static_cast<float>(x)});
      m_right_geometry.vertexbuffer().push_back(vec3f{
          static_cast<float>(dim0.back()), dim1.back(), static_cast<float>(x)});
      m_right_geometry.indexbuffer().push_back(j++);
      m_right_geometry.indexbuffer().push_back(j++);
    }
  }
};
template <size_t N>
struct grid : renderable<grid<N>>, grid_renderer<N> {
  std::array<ui::input_pin*, N> m_input_pins;
  //============================================================================
  grid(flowexplorer::scene& s)
      : renderable<grid<N>>{
            "Grid", s,
            *dynamic_cast<tatooine::non_uniform_grid<real_t, N>*>(this)} {
    for (size_t i = 0; i < N; ++i) {
      m_input_pins[i] =
          &this->template insert_input_pin<linspace<real_t>>("dim");
    }
  }
  //----------------------------------------------------------------------------
  auto render(mat4f const& P, mat4f const& V) -> void override {
    if (!all_pins_linked()) {
      return;
    }
    for (size_t i = 0; i < N; ++i) {
      if (dimension(i).size() < 1) {
        return;
      }
    }
    if constexpr (N == 3) {
      grid_renderer<N>::render(P, V, this->scene().camera().eye());
    } else {
      grid_renderer<N>::render(P, V);
    }
  }
  //----------------------------------------------------------------------------
  auto all_pins_linked() const -> bool {
    bool all_linked = true;
    for (size_t i = 0; i < N; ++i) {
      if (!m_input_pins[i]->is_linked()) {
        all_linked = false;
        break;
      }
    }
    return all_linked;
  }
  //----------------------------------------------------------------------------
  using non_uniform_grid<real_t, N>::dimension;
  auto dimension(size_t i) -> auto& {
    if (i == 0) {
      return this->template dimension<0>();
    }
    if (i == 1) {
      return this->template dimension<1>();
    }
    if constexpr (N > 2) {
      if (i == 2) {
        return this->template dimension<2>();
      }
    }
    return this->template dimension<0>();
  }
  //----------------------------------------------------------------------------
  auto on_property_changed() -> void override {
    size_t i = 0;
    for (auto ptr : m_input_pins) {
      auto& pin = *ptr;
      if (pin.is_linked()) {
        if (pin.linked_type() == typeid(linspace<real_t>)) {
          auto& data = pin.template get_linked_as<linspace<real_t>>();
          auto& d    = dimension(i);
          d.clear();
          std::copy(begin(data), end(data), std::back_inserter(d));
        }
      }
      ++i;
    }
    if (all_pins_linked()) {
      for (size_t i = 0; i < N; ++i) {
        if (dimension(i).size() < 1) {
          return;
        }
      }
      this->update_geometry();
      this->notify_property_changed(false);
    }
  }
  //----------------------------------------------------------------------------
  auto on_pin_connected(ui::input_pin& p, ui::output_pin&) -> void override {
    for (size_t i = 0; i < N; ++i) {
      if (&p == m_input_pins[i]) {
        if (p.linked_type() == typeid(linspace<real_t>)) {
          auto& data = p.get_linked_as<linspace<real_t>>();
          auto& d    = dimension(i);
          d.clear();
          std::copy(begin(data), end(data), std::back_inserter(d));
        }
        break;
      }
    }

    if (all_pins_linked()) {
      for (size_t i = 0; i < N; ++i) {
        if (dimension(i).size() < 1) {
          return;
        }
      }
      this->update_geometry();
      this->notify_property_changed(false);
    }
  }
};
//==============================================================================
using grid2 = grid<2>;
using grid3 = grid<3>;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(tatooine::flowexplorer::nodes::grid2)
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(tatooine::flowexplorer::nodes::grid3)
#endif
