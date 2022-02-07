#ifndef TATOOINE_RENDERING_INTERACTIVE_H
#define TATOOINE_RENDERING_INTERACTIVE_H
//==============================================================================
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/rendering/interactive/ellipse.h>
#include <tatooine/rendering/interactive/pointset.h>
#include <tatooine/rendering/interactive/rectilinear_grid.h>
#include <tatooine/type_set.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
namespace detail {
//==============================================================================
template <typename... Renderers>
auto set_view_matrices(Mat4<GLfloat> const& V, type_set_impl<Renderers...>) {
  (
      [&] {
        if constexpr (requires { Renderers::set_view_matrix(V); }) {
          Renderers::set_view_matrix(V);
        }
      }(),
      ...);
}
//==============================================================================
template <typename... Renderers>
auto set_projection_matrices(Mat4<GLfloat> const& P,
                             type_set_impl<Renderers...>) {
  (
      [&] {
        if constexpr (requires { Renderers::set_projection_matrix(P); }) {
          Renderers::set_projection_matrix(P);
        }
      }(),
      ...);
}
//==============================================================================
template <typename... Renderers>
auto set_view_projection_matrices(Mat4<GLfloat> const& VP,
                                  type_set_impl<Renderers...>) {
  (
      [&] {
        if constexpr (requires { Renderers::set_view_projection_matrix(VP); }) {
          Renderers::set_view_projection_matrix(VP);
        }
      }(),
      ...);
}
struct window {
  [[nodiscard]]static auto get() -> auto& {
    static auto w = first_person_window{500, 500};
    return w;
  }
};
//==============================================================================
}  // namespace detail
//==============================================================================
/// Call this function if you need to create gpu data before calling render.
auto pre_setup() { [[maybe_unused]] auto& w = detail::window::get(); }
//==============================================================================
template <std::size_t... Is,
          interactively_renderable... Renderables>
auto render(std::index_sequence<Is...> /*seq*/, Renderables&&... renderables) {
  using namespace detail;
  using renderer_type_set = type_set<renderer<std::decay_t<Renderables>>...>;

  auto& window = detail::window::get();
  window.add_resize_event([&](int width, int height) {
    auto const P =
        window.camera_controller().active_camera().projection_matrix();
    set_projection_matrices(P, renderer_type_set{});
  });
  window.add_wheel_up_event([&]() {
    auto const P =
        window.camera_controller().active_camera().projection_matrix();
    set_projection_matrices(P, renderer_type_set{});
  });
  window.add_wheel_down_event([&]() {
    auto const P =
        window.camera_controller().active_camera().projection_matrix();
    set_projection_matrices(P, renderer_type_set{});
  });

  gl::clear_color(255, 255, 255, 255);

  auto renderers = std::tuple{[](auto&& renderable) {
    using type = decltype(renderable);
    using decayed_type = std::decay_t<type>;
    if constexpr (requires(type t) { t.render(); }) {
      return renderable;
    } else if constexpr (requires(renderer<decayed_type> t) { t.render(); }) {
      return renderer<decayed_type>(renderable);
    }
  }(renderables)...};

  auto foreach_renderer = [&](auto&& f) {
    (f(std::get<Is>(renderers), renderables, Is), ...);
  };

  window.add_button_released_event([&](gl::button b) {
    foreach_renderer([&](auto& renderer, auto& renderable, auto i) {
      if constexpr (requires { renderer.on_button_released(b); }) {
        renderer.on_button_released(b);
      }
    });
  });
  window.add_button_pressed_event([&](gl::button b) {
    foreach_renderer([&](auto& renderer, auto& renderable, auto i) {
      if constexpr (requires { renderer.on_button_pressed(b); }) {
        renderer.on_button_pressed(b);
      }
    });
  });
  window.add_cursor_moved_event([&](double const x, double const y) {
    foreach_renderer([&](auto& renderer, auto& renderable, auto i) {
      if constexpr (requires { renderer.on_cursor_moved(x, y); }) {
        renderer.on_cursor_moved(x, y);
      }
    });
  });

  window.render_loop([&](auto const dt) {
    auto const V = window.camera_controller().view_matrix();
    set_view_matrices(V, renderer_type_set{});

    gl::clear_color_depth_buffer();
    ImGui::Begin("Properties");
    foreach_renderer([&](auto& renderer, auto& renderable, auto i) {
      if constexpr (requires {
                      renderer.update(
                          renderable,
                          window.camera_controller().active_camera(), dt);
                    }) {
        renderer.update(renderable, window.camera_controller().active_camera(),
                        dt);
      }
      else if constexpr (requires {
                      renderer.update(
                          window.camera_controller().active_camera(), dt);
                    }) {
        renderer.update(window.camera_controller().active_camera(),
                        dt);
      }

      if constexpr (requires { renderer.properties(renderable); }) {
        ImGui::PushID(i);
        ImGui::BeginGroup();
        renderer.properties(renderable);
        ImGui::EndGroup();
        ImGui::PopID();
      }
      renderer.render();
    });
    ImGui::End();
  });
}
//------------------------------------------------------------------------------
auto render(interactively_renderable auto&&... renderables) {
  render(std::make_index_sequence<sizeof...(renderables)>{},
         std::forward<decltype(renderables)>(renderables)...);
}
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
