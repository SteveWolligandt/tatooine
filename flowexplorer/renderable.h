#ifndef TATOOINE_FLOWEXPLORER_RENDERABLE_H
#define TATOOINE_FLOWEXPLORER_RENDERABLE_H
//==============================================================================
#include <tatooine/boundingbox.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct renderable {
  virtual ~renderable()                                      = default;
  virtual void        draw_ui()                              = 0;
  virtual void        render(const yavin::mat4& projection_matrix,
                             const yavin::mat4& view_matrix) = 0;
  virtual std::string name() const                           = 0;
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
