#include <yavin/vertexarray.h>
//==============================================================================
namespace yavin {
//==============================================================================
vertexarray::vertexarray() { gl::create_vertex_arrays(1, id_ptr()); }
//------------------------------------------------------------------------------
vertexarray::~vertexarray() { destroy_handle(); }
//------------------------------------------------------------------------------
void vertexarray::destroy_handle() {
  if (id() != 0) {
    gl::delete_vertex_arrays(1, id_ptr());
  }
  set_id(0);
}
//------------------------------------------------------------------------------
void vertexarray::bind() const { gl::bind_vertex_array(id()); }
//------------------------------------------------------------------------------
void vertexarray::unbind() const { gl::bind_vertex_array(0); }
//------------------------------------------------------------------------------
void vertexarray::draw(Primitive primitive, size_t num_primitives) const {
  gl::draw_elements(primitive, num_primitives, UINT, 0);
}
//------------------------------------------------------------------------------
void vertexarray::draw_points(size_t num_primitives) const {
  draw(POINTS, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_line_strip(size_t num_primitives) const {
  draw(LINE_STRIP, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_line_loop(size_t num_primitives) const {
  draw(LINE_LOOP, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_lines(size_t num_primitives) const {
  draw(LINES, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_line_strip_adjacency(size_t num_primitives) const {
  draw(LINE_STRIP_ADJACENCY, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_triangle_strip(size_t num_primitives) const {
  draw(TRIANGLE_STRIP, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_triangle_fan(size_t num_primitives) const {
  draw(TRIANGLE_FAN, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_triangles(size_t num_primitives) const {
  draw(TRIANGLES, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_triangle_strip_adjacency(size_t num_primitives) const {
  draw(TRIANGLE_STRIP_ADJACENCY, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_triangles_adjacency(size_t num_primitives) const {
  draw(TRIANGLES_ADJACENCY, num_primitives);
}
//------------------------------------------------------------------------------
void vertexarray::draw_patches(size_t num_primitives) const {
  draw(PATCHES, num_primitives);
}
//==============================================================================
}  // namespace yavin
//==============================================================================
