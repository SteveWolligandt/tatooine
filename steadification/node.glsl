#ifndef TATOOINE_STEADIFICATION_NODE
#define TATOOINE_STEADIFICATION_NODE
//------------------------------------------------------------------------------
struct node {
  vec2  v;
  float tau;
  float curvature;
  uint  render_index;
  uint  layer;
  uint  next_index;
  float pad;
};
#endif
