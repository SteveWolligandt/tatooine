#ifndef TATOOINE_FLOWEXPLORER_RENDERABLE_H
#define TATOOINE_FLOWEXPLORER_RENDERABLE_H
//==============================================================================
#include <tatooine/boundingbox.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct renderable {
  bool m_active                                              = true;

  virtual ~renderable()                                      = default;
  virtual void        draw_ui()                              = 0;
  virtual void        update(const std::chrono::duration<double>& dt) {}
  virtual void        render(const yavin::mat4& projection_matrix,
                             const yavin::mat4& view_matrix) = 0;
  virtual std::string name() const                           = 0;

  bool& is_active() { return m_active; }
  bool  is_active() const { return m_active; }
  void set_active(bool a = true) { m_active = a; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
