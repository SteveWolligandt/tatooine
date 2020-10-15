#ifndef TATOOINE_FLOWEXPLORER_RENDERABLE_H
#define TATOOINE_FLOWEXPLORER_RENDERABLE_H
//==============================================================================
#include <chrono>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/mat.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window;

struct base_renderable : ui::base_node {
  flowexplorer::window* m_window;

  base_renderable(flowexplorer::window& w, std::string const& name);
  base_renderable(base_renderable const& w)                        = default;
  base_renderable(base_renderable&& w) noexcept                    = default;
  auto operator=(base_renderable const& w) -> base_renderable&     = default;
  auto operator=(base_renderable&& w) noexcept -> base_renderable& = default;
  virtual ~base_renderable()                                       = default;

  virtual void        update(const std::chrono::duration<double>& dt) {}
  virtual void        render(const mat<float, 4, 4>& projection_matrix,
                             const mat<float, 4, 4>& view_matrix) = 0;
  virtual bool        is_transparent() const                      = 0;

  auto window() const -> auto const& {
    return *m_window;
  }
  auto window() -> auto& {
    return *m_window;
  }
};
template <typename Derived>
struct renderable : base_renderable {
  using base_renderable::base_renderable;
  virtual ~renderable()                                       = default;
  //static constexpr auto node_type_name() -> std::string_view {
  //  return Derived::name();
  //}
  //constexpr auto node_type_name() const -> std::string_view override {
  //  return node_type_name();
  //}
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
