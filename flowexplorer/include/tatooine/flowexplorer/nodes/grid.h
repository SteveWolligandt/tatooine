#ifndef TATOOINE_FLOWEXPLORER_NODES_GRID_H
#define TATOOINE_FLOWEXPLORER_NODES_GRID_H
//==============================================================================
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/linspace.h>
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/grid.h>
#include <yavin/indexeddata.h>
#include <tatooine/rendering/yavin_interop.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <size_t N>
struct grid : renderable<grid<N>>, tatooine::non_uniform_grid<real_t, N> {
  yavin::indexeddata<vec3f> m_inner_geometry;
  yavin::indexeddata<vec3f> m_outer_geometry;
  std::array<ui::input_pin*, N> m_input_pins;
  //============================================================================
  grid(flowexplorer::scene& s)
      : renderable<grid<N>>{
            "Grid", s,
            *dynamic_cast<tatooine::non_uniform_grid<real_t, N>*>(this)} {
    for (size_t i = 0; i < N; ++i) {
      m_input_pins[i] = &this->template insert_input_pin<linspace<real_t>>("dim");
    }
  }
  //----------------------------------------------------------------------------
  auto render(mat4f const& P, mat4f const& V) -> void override {
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
  auto all_pins_linked() const -> bool{
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
      update_geometry();
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
      update_geometry();
    }
  }
  //----------------------------------------------------------------------------
  auto update_geometry() {
    m_inner_geometry.clear();
    m_outer_geometry.clear();
    if constexpr (N == 2) {
      auto const& dim0 = dimension(0);
      auto const& dim1 = dimension(1);

      size_t j = 0;
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.front()), 0.0f});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()), 0.0f});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // right
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()), 0.0f});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()), 0.0f});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // bottom
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.front()), 0.0f});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()), 0.0f});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // top
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()), 0.0f});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()), 0.0f});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);

      j = 0;
      for (size_t i = 1; i < size(dim0) - 1; ++i) {
        auto const x = dim0[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.front()), 0.0f});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.back()), 0.0f});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim1) - 1; ++i) {
        auto const x = dim1[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.front()), static_cast<float>(x), 0.0f});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.back()), static_cast<float>(x), 0.0f});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
    }
    else if constexpr (N == 3) {
      auto const& dim0 = dimension(0);
      auto const& dim1 = dimension(1);
      auto const& dim2 = dimension(2);

      size_t j = 0;
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.front()), static_cast<float>(dim2.front())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.front()),static_cast<float>(dim2.back())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()), static_cast<float>(dim2.front())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),static_cast<float>(dim2.back())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()), static_cast<float>(dim2.front())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),static_cast<float>(dim2.back())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()), static_cast<float>(dim2.front())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),static_cast<float>(dim2.back())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);


      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.front()), static_cast<float>(dim2.front())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),static_cast<float>(dim2.front())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.front()), static_cast<float>(dim2.back())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()),static_cast<float>(dim2.back())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()), static_cast<float>(dim2.front())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),static_cast<float>(dim2.front())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()), static_cast<float>(dim2.back())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),static_cast<float>(dim2.back())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);

      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.front()), static_cast<float>(dim2.front())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),static_cast<float>(dim2.front())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.front()), static_cast<float>(dim2.back())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.front()),static_cast<float>(dim2.back())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()), static_cast<float>(dim2.front())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),static_cast<float>(dim2.front())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);
      // left
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.front()), static_cast<float>(dim1.back()), static_cast<float>(dim2.back())});
      m_outer_geometry.vertexbuffer().push_back(
          vec3f{static_cast<float>(dim0.back()), static_cast<float>(dim1.back()),static_cast<float>(dim2.back())});
      m_outer_geometry.indexbuffer().push_back(j++);
      m_outer_geometry.indexbuffer().push_back(j++);

      j = 0;
      for (size_t i = 1; i < size(dim0) - 1; ++i) {
        auto const x = dim0[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.front()), dim2.front()});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.back()), dim2.front()});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim1) - 1; ++i) {
        auto const x = dim1[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.front()), static_cast<float>(x), dim2.front()});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.back()), static_cast<float>(x), dim2.front()});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim0) - 1; ++i) {
        auto const x = dim0[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.front()), dim2.back()});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.back()), dim2.back()});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim1) - 1; ++i) {
        auto const x = dim1[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.front()), static_cast<float>(x), dim2.back()});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.back()), static_cast<float>(x), dim2.back()});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }

      for (size_t i = 1; i < size(dim0) - 1; ++i) {
        auto const x = dim0[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.front()), dim2.front()});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.front()), dim2.back()});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim2) - 1; ++i) {
        auto const x = dim2[i];
        m_inner_geometry.vertexbuffer().push_back(
            vec3f{static_cast<float>(dim0.front()), dim1.front(), static_cast<float>(x)});
        m_inner_geometry.vertexbuffer().push_back(
            vec3f{static_cast<float>(dim0.back()), dim1.front(), static_cast<float>(x)});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim0) - 1; ++i) {
        auto const x = dim0[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.back()), dim2.front()});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(x), static_cast<float>(dim1.back()), dim2.back()});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim2) - 1; ++i) {
        auto const x = dim2[i];
        m_inner_geometry.vertexbuffer().push_back(
            vec3f{static_cast<float>(dim0.front()), dim1.back(), static_cast<float>(x)});
        m_inner_geometry.vertexbuffer().push_back(
            vec3f{static_cast<float>(dim0.back()), dim1.back(), static_cast<float>(x)});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }


      for (size_t i = 1; i < size(dim1) - 1; ++i) {
        auto const x = dim1[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.front()), static_cast<float>(x), dim2.front()});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.front()), static_cast<float>(x), dim2.back()});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim2) - 1; ++i) {
        auto const x = dim2[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.front()), dim1.front(), static_cast<float>(x)});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.front()), dim1.back(), static_cast<float>(x)});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim1) - 1; ++i) {
        auto const x = dim1[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.back()), static_cast<float>(x), dim2.front()});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.back()), static_cast<float>(x), dim2.back()});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
      for (size_t i = 1; i < size(dim2) - 1; ++i) {
        auto const x = dim2[i];
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.back()), dim1.front(), static_cast<float>(x)});
        m_inner_geometry.vertexbuffer().push_back(vec3f{
            static_cast<float>(dim0.back()), dim1.back(), static_cast<float>(x)});
        m_inner_geometry.indexbuffer().push_back(j++);
        m_inner_geometry.indexbuffer().push_back(j++);
      }
    }
    this->notify_property_changed(false);
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
