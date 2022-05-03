#ifndef TATOOINE_RENDERING_INTERACTIVE_RENDERER_H
#define TATOOINE_RENDERING_INTERACTIVE_RENDERER_H
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <typename T>
struct renderer;
template <typename T>
struct is_renderer_impl : std::false_type {};
template <typename T>
struct is_renderer_impl<renderer<T>> : std::true_type {};
template <typename T>
static auto constexpr is_renderer = is_renderer_impl<T>::value;
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
