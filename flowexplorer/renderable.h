#ifndef TATOOINE_FLOWEXPLORER_RENDERABLE_H
#define TATOOINE_FLOWEXPLORER_RENDERABLE_H
//==============================================================================
#include <chrono>
#include <tatooine/boundingbox.h>
#include <tatooine/rendering/yavin_interop.h>
#include "ui/node.h"
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window;
struct renderable : ui::node {
  window* m_window;

  renderable(window& w, std::string const& name)
      : m_window{&w}, ui::node{name} {}
  renderable(renderable const& w)     = default;
  renderable(renderable&& w) noexcept = default;
  auto operator=(renderable const& w) -> renderable& = default;
  auto operator=(renderable&& w) noexcept -> renderable& = default;
  virtual ~renderable()         = default;

  virtual void        draw_ui() = 0;
  virtual void        update(const std::chrono::duration<double>& dt) {}
  virtual void        render(const mat<float, 4, 4>& projection_matrix,
                             const mat<float, 4, 4>& view_matrix) = 0;

  auto window() const -> auto const& {
    return *m_window;
  }
  auto window() -> auto& {
    return *m_window;
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
