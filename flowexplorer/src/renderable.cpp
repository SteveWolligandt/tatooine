#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/flowexplorer/window.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
renderable::renderable(flowexplorer::window& w, std::string const& title, scene const& s)
    : m_window{&w}, ui::node{title, s} {}

//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
