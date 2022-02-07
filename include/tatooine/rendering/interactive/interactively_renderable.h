#ifndef TATOOINE_RENDERING_INTERACTIVE_INTERACTIVELY_RENDERABLE_H
#define TATOOINE_RENDERING_INTERACTIVE_INTERACTIVELY_RENDERABLE_H
//==============================================================================
#include <type_traits>
#include <tatooine/rendering/interactive/renderer.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <typename T>
concept interactively_renderable =
requires(T t) { t.render(); } ||
requires(renderer<std::decay_t<T>> t) { t.render(); };
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
