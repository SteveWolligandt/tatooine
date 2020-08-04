#ifndef TATOOINE_FLOWEXPLORER_RENDERABLE_H
#define TATOOINE_FLOWEXPLORER_RENDERABLE_H
//==============================================================================
#include <tatooine/boundingbox.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window;
struct renderable {
  bool    m_active = true;
  window* m_window;

  renderable(window& w) : m_window{&w}{}
  renderable(renderable const& w)     = default;
  renderable(renderable&& w) noexcept = default;
  auto operator=(renderable const& w) -> renderable& = default;
  auto operator=(renderable&& w) noexcept -> renderable& = default;
  virtual ~renderable()         = default;

  virtual void        draw_ui() = 0;
  virtual void        update(const std::chrono::duration<double>& dt) {}
  virtual void        render(const yavin::mat4& projection_matrix,
                             const yavin::mat4& view_matrix) = 0;
  virtual std::string name() const                           = 0;

  bool& is_active() { return m_active; }
  bool  is_active() const { return m_active; }
  void set_active(bool a = true) { m_active = a; }

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
