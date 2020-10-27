#ifndef TATOOINE_AUTONOMOUS_PARTICLES_ELLIPSIS_VERTICES_H
#define TATOOINE_AUTONOMOUS_PARTICLES_ELLIPSIS_VERTICES_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto ellipsis_vertices(mat<double, 2, 2> const& S, vec<double, 2> const& x0,
                       size_t const resolution) -> std::vector<vec<double, 3>>;
//==============================================================================
#endif
