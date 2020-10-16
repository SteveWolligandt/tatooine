#include <tatooine/flowexplorer/renderable.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
renderable::renderable(std::string const& title, flowexplorer::scene& s)
    : ui::node{title, s} {}

//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
