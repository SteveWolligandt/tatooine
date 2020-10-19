#include <tatooine/flowexplorer/renderable.h>
//==============================================================================
namespace tatooine::flowexplorer::base {
//==============================================================================
renderable::renderable(std::string const& title, flowexplorer::scene& s)
    : ui::base::node{title, s} {}

//==============================================================================
}  // namespace tatooine::flowexplorer::base
//==============================================================================
