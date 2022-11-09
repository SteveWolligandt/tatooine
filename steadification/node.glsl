#ifndef TATOOINE_STEADIFICATION_NODE
#define TATOOINE_STEADIFICATION_NODE
//------------------------------------------------------------------------------
struct node {
  vec2  v;
  float t;
  float t0;
  float curvature;
  uint  renderindex;
  uint  layer;
  float padding;
};
#endif
