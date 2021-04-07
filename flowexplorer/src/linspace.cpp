#include <tatooine/flowexplorer/scene.h>
//
#include <tatooine/flowexplorer/nodes/linspace.h>
namespace tatooine::flowexplorer::nodes {
linspace::linspace(flowexplorer::scene& s)
    : node<linspace>{"Linspace", s,
                     *dynamic_cast<tatooine::linspace<real_t>*>(this)} {}
}
