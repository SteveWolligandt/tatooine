#ifndef TATOOINE_RENDERING_INTERACTIVE_INTERACTIVELY_RENDERABLE_H
#define TATOOINE_RENDERING_INTERACTIVE_INTERACTIVELY_RENDERABLE_H
//==============================================================================
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/renderer.h>

#include <type_traits>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <typename T>
concept interactively_renderable = requires(T t) {
  t.render();
}
|| requires(T t) { t.late_render(); }
|| requires(T t) {
  t.render(std::declval<camera_interface<double>>());
}
|| requires(T t) {
  t.render(std::declval<camera_interface<float>>());
}
|| requires(renderer<std::decay_t<T>> t) { t.render(); }
|| requires(renderer<std::decay_t<T>> t) {
  t.render(std::declval<std::decay_t<T>>());
}
|| requires(renderer<std::decay_t<T>> t) {
  t.render(std::declval<std::decay_t<T>>(),
           std::declval<camera_interface<double>>());
}
|| requires(renderer<std::decay_t<T>> t) {
  t.render(std::declval<std::decay_t<T>>(),
           std::declval<camera_interface<float>>());
}
|| requires(renderer<std::decay_t<T>> t) { t.late_render(); }
|| requires(renderer<std::decay_t<T>> t) {
  t.late_render(std::declval<std::decay_t<T>>());
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
