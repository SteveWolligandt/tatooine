#ifndef TATOOINE_RENDERING_INTERACTIVE_H
#define TATOOINE_RENDERING_INTERACTIVE_H
//==============================================================================
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/rendering/interactive/ellipse.h>
#include <tatooine/rendering/interactive/rectilinear_grid.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <std::size_t... Is, typename... Renderables>
auto interactive(std::index_sequence<Is...> /*seq*/,
                 Renderables&&... renderables) {
  auto window = first_person_window{500, 500};
  gl::clear_color(255, 255, 255, 255);
  auto data = std::tuple{rendering::detail::interactive::renderer<
      std::decay_t<decltype(renderables)>>::init(renderables)...};
  window.render_loop([&](auto const dt) {
    ImGui::Begin("Properties");
    gl::clear_color_depth_buffer();
    (
        [&] {
          using renderer = rendering::detail::interactive::renderer<
              std::decay_t<decltype(renderables)>>;
          auto& render_data = std::get<Is>(data);
          ImGui::PushID(static_cast<int>(Is));
          ImGui::BeginGroup();
          renderer::properties(renderables, render_data);
          renderer::render(window.camera_controller().active_camera(),
                           std::forward<Renderables>(renderables),
                           render_data);
          ImGui::EndGroup();
          ImGui::PopID();
        }(),
        ...);
    ImGui::End();
  });
}
//------------------------------------------------------------------------------
template <std::size_t... Is, typename... Renderables>
auto interactive(Renderables&&... renderables) {
  interactive(std::make_index_sequence<sizeof...(Renderables)>{},
              std::forward<Renderables>(renderables)...);
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
