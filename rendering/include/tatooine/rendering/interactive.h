#ifndef TATOOINE_RENDERING_INTERACTIVE_H
#define TATOOINE_RENDERING_INTERACTIVE_H
//==============================================================================
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/rendering/interactive/edgeset2.h>
#include <tatooine/rendering/interactive/ellipse.h>
#include <tatooine/rendering/interactive/pointset.h>
#include <tatooine/rendering/interactive/rectilinear_grid.h>
#include <tatooine/rendering/interactive/axis_aligned_bounding_box.h>
#include <tatooine/rendering/interactive/unstructured_triangular_grid2.h>
#include <tatooine/rendering/interactive/unstructured_triangular_grid3.h>
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
  [[nodiscard]] static auto get() -> auto& {
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
template <std::size_t... Is, interactively_renderable... Renderables>
auto show(std::index_sequence<Is...> /*seq*/, Renderables&&... renderables) {
  using namespace detail;
  using renderer_type_set = type_set<renderer<std::decay_t<Renderables>>...>;

  auto&      window             = detail::window::get();
  auto const max_num_dimensions = tatooine::max([&] {
    using renderable_t = std::decay_t<decltype(renderables)>;
    if constexpr (range<renderable_t>) {
      return renderable_t::value_type::num_dimensions();
    } else {
      return renderables.num_dimensions();
    }
  }()...);
  if (max_num_dimensions == 2) {
    window.camera_controller().use_orthographic_camera();
    window.camera_controller().use_orthographic_controller();
  } else if (max_num_dimensions == 3) {
    window.camera_controller().use_perspective_camera();
    window.camera_controller().use_fps_controller();
  }
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

  auto& io     = ImGui::GetIO();
  auto  roboto = io.Fonts->AddFontFromFileTTF(
       "/home/steve/libs/tatooine2/resources/fonts/Roboto-Regular.ttf", 25);
  auto enable_renderer = std::array{((void)renderables, true)...};
  auto renderers = std::tuple{[&](auto&& renderable) {
    using type         = decltype(renderable);
    using decayed_type = std::decay_t<type>;
    if constexpr (
        requires(type t) { t.render(); } ||
        requires(type t) { t.late_render(); } ||
        requires(type t) {
          t.render(renderable, window.camera_controller().active_camera());
        }) {
      return renderable;
    } else if constexpr (
        requires(renderer<decayed_type> t) { t.render(); } ||
        requires(renderer<decayed_type> t) { t.render(renderable); } ||
        requires(renderer<decayed_type> t) { t.late_render(); } ||
        requires(renderer<decayed_type> t) { t.late_render(renderable); } ||
        requires(renderer<decayed_type> t) {
          t.render(renderable, window.camera_controller().active_camera());
        }) {
      return renderer<decayed_type>(renderable);
    }
  }(renderables)...};

  auto foreach_renderer = [&](auto&& f) {
    (f(std::get<Is>(renderers), renderables, Is), ...);
  };

  window.add_button_released_event([&](gl::button b) {
    foreach_renderer([&](auto& renderer, auto& renderable, auto i) {
      if (enable_renderer[i]) {
        if constexpr (requires { renderer.on_button_released(b); }) {
          renderer.on_button_released(b);
        }
        if constexpr (requires {
                        renderer.on_button_released(
                            b, window.camera_controller().active_camera());
                      }) {
          renderer.on_button_released(
              b, window.camera_controller().active_camera());
        }
      }
    });
  });
  window.add_button_pressed_event([&](gl::button b) {
    foreach_renderer([&](auto& renderer, auto& renderable, auto i) {
      if (enable_renderer[i]) {
        if constexpr (requires { renderer.on_button_pressed(b); }) {
          renderer.on_button_pressed(b);
        }
        if constexpr (requires {
                        renderer.on_button_pressed(
                            b, window.camera_controller().active_camera());
                      }) {
          renderer.on_button_pressed(
              b, window.camera_controller().active_camera());
        }
      }
    });
  });
  window.add_cursor_moved_event([&](double const x, double const y) {
    foreach_renderer([&](auto& renderer, auto& renderable, auto i) {
      if (enable_renderer[i]) {
        if constexpr (requires { renderer.on_cursor_moved(x, y); }) {
          renderer.on_cursor_moved(x, y);
        }
        if constexpr (requires {
                        renderer.on_cursor_moved(
                            x, y, window.camera_controller().active_camera());
                      }) {
          renderer.on_cursor_moved(x, y,
                                   window.camera_controller().active_camera());
        }
      }
    });
  });

  window.render_loop([&](auto const dt) {
    auto const V = window.camera_controller().view_matrix();
    set_view_matrices(V, renderer_type_set{});

    gl::clear_color_depth_buffer();
    ImGui::Begin("Properties");
    //ImGui::PushFont(roboto);
    foreach_renderer([&](auto& renderer, auto& renderable, auto i) {
      if (enable_renderer[i]) {
        if constexpr (requires { renderer.update(dt); }) {
          renderer.update(dt);
        }
        if constexpr (requires { renderer.update(dt, renderable); }) {
          renderer.update(dt, renderable);
        }
        if constexpr (requires {
                        renderer.update(
                            dt, renderable,
                            window.camera_controller().active_camera());
                      }) {
          renderer.update(dt, renderable,
                          window.camera_controller().active_camera());
        }
        if constexpr (requires { renderer.render(); }) {
          renderer.render();
        }
        if constexpr (requires { renderer.render(renderable); }) {
          renderer.render(renderable);
        }
        if constexpr (requires {
                        renderer.render(
                            renderable,
                            window.camera_controller().active_camera());
                      }) {
          renderer.render(renderable,
                          window.camera_controller().active_camera());
        }
      }

      ImGui::PushID(i);
      ImGui::ToggleButton("##enable", &enable_renderer[i]);
      if constexpr (requires { renderer.properties(renderable); }) {
        ImGui::BeginGroup();
        renderer.properties(renderable);
        ImGui::EndGroup();
      }
      if constexpr (requires { renderer.properties(); }) {
        ImGui::BeginGroup();
        renderer.properties();
        ImGui::EndGroup();
      }
      ImGui::PopID();
    });
    //ImGui::PopFont();
    ImGui::End();
    gl::clear_depth_buffer();
    // In this late render area you can use custom cameras.
    foreach_renderer([&](auto& renderer, auto& renderable, auto i) {
      if (enable_renderer[i]) {
        if constexpr (requires { renderer.late_render(); }) {
          renderer.late_render();
        }
        if constexpr (requires { renderer.late_render(renderable); }) {
          renderer.late_render(renderable);
        }
      }
    });
  });
}
//------------------------------------------------------------------------------
auto show(interactively_renderable auto&&... renderables) {
  show(std::make_index_sequence<sizeof...(renderables)>{},
       std::forward<decltype(renderables)>(renderables)...);
}
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
