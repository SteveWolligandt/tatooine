#include <tatooine/flowexplorer/renderable.h>
//==============================================================================
namespace tatooine::flowexplorer::base {
//==============================================================================
renderable::renderable(std::string const& title, flowexplorer::scene& s)
    : ui::base::node{title, s} {}
//------------------------------------------------------------------------------
renderable::renderable(std::string const& title, flowexplorer::scene& s,
                       std::type_info const& self_type)
    : ui::base::node{title, s, self_type} {}
//==============================================================================
}  // namespace tatooine::flowexplorer::base
//==============================================================================
