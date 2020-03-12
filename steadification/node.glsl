#ifndef TATOOINE_STEADIFICATION_NODE
#define TATOOINE_STEADIFICATION_NODE
//------------------------------------------------------------------------------
struct node {
  vec2  pos;
  vec2  v;
  float tau;
  float curvature;
  uint  render_index;
  uint  next_index;
};
#endif
