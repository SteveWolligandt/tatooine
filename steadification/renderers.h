#ifndef RENDERERS_H
#define RENDERERS_H

#include <tatooine/streamsurface.h>
#include <yavin>
using namespace yavin;

//==============================================================================

class StreamsurfaceRenderer : public indexeddata<vec2, vec2, vec2> {
 public:
  using real_t       = double;
  using parent_t     = indexeddata<vec2, vec2, vec2>;
  using vbo_data_vec = parent_t::vbo_data_vec;
  using ibo_data_vec = parent_t::ibo_data_vec;

  explicit StreamsurfaceRenderer(const tatooine::mesh<real_t, 2>& mesh);
  StreamsurfaceRenderer(const StreamsurfaceRenderer& other) : parent_t{other} {}
  StreamsurfaceRenderer(StreamsurfaceRenderer&& other)
      : parent_t{std::move(other)} {}

  //----------------------------------------------------------------------------
  static vbo_data_vec mesh_to_vbo_data(const tatooine::mesh<real_t, 2>& mesh);

  //----------------------------------------------------------------------------
  static ibo_data_vec mesh_to_ibo_data(const tatooine::mesh<real_t, 2>& mesh);
  void                draw() const { draw_triangles(); }
};

//==============================================================================
class ScreenSpaceQuad : public indexeddata<vec2> {
 public:
  using parent_t = indexeddata<vec2>;

  ScreenSpaceQuad();
  ScreenSpaceQuad(const ScreenSpaceQuad& other) : parent_t{other} {}
  ScreenSpaceQuad(ScreenSpaceQuad&& other) : parent_t{std::move(other)} {}
  void draw() const { draw_triangles(); }
};

#endif
