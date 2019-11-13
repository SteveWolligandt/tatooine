#include "renderers.h"

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>

//==============================================================================
namespace tatooine {
//==============================================================================
void streamsurface_renderer::draw() const { draw_triangles(); }
//==============================================================================
}  // namespace tatooine
//==============================================================================
