#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/flowexplorer/window.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
base_renderable::base_renderable(flowexplorer::window& w,
                                 std::string const&    name)
    : m_window{&w}, ui::base_node{name} {}

//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
