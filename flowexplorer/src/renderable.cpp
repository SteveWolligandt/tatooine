#include <tatooine/flowexplorer/renderable.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
renderable::renderable(std::string const& title, scene const& s)
    : ui::node{title, s} {}

//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
