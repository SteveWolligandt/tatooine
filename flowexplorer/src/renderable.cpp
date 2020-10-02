#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/flowexplorer/window.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
renderable::renderable(flowexplorer::window& w, std::string const& name)
    : m_window{&w}, ui::node{name} {}

//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
