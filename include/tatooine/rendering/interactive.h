#ifndef TATOOINE_RENDERING_INTERACTIVE_H
#define TATOOINE_RENDERING_INTERACTIVE_H
//==============================================================================
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/rendering/interactive/ellipse.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename... Renderables>
auto interactive(Renderables&&... renderables) {
  auto window = first_person_window{500, 500};
  gl::clear_color(255, 255, 255, 255);
  window.render_loop([&](auto const dt) {
    gl::clear_color_depth_buffer();
    (rendering::detail::interactive::render(
         window.camera_controller().active_camera(),
         std::forward<Renderables>(renderables)),
     ...);
  });
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
