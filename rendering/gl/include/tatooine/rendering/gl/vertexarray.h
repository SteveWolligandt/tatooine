#ifndef YAVIN_VERTEXARRAY_H
#define YAVIN_VERTEXARRAY_H
//==============================================================================
#include <iostream>
#include <yavin/indexbuffer.h>
#include <yavin/primitive.h>
#include <yavin/type.h>
#include <yavin/vertexbuffer.h>
#include <yavin/dllexport.h>
#include <yavin/glfunctions.h>
//==============================================================================
namespace yavin {
//==============================================================================
class vertexarray : public id_holder<GLuint> {
 public:
  using this_t = vertexarray;

  DLL_API vertexarray();
  DLL_API vertexarray(vertexarray const& other) = delete;
  DLL_API vertexarray(vertexarray&& other)      = default;
  DLL_API vertexarray& operator=(vertexarray const& other) = delete;
  DLL_API vertexarray& operator=(vertexarray&& other) = default;
  DLL_API ~vertexarray();

  DLL_API void destroy_handle();
  DLL_API void bind() const;
  DLL_API void unbind() const;
  DLL_API void draw(Primitive primitive, size_t num_primitives) const;
  DLL_API void draw(size_t num_primitives) const;
  DLL_API void draw_points(size_t num_primitives) const;
  DLL_API void draw_line_strip(size_t num_primitives) const;
  DLL_API void draw_line_loop(size_t num_primitives) const;
  DLL_API void draw_lines(size_t num_primitives) const;
  DLL_API void draw_line_strip_adjacency(size_t num_primitives) const;
  DLL_API void draw_triangle_strip(size_t num_primitives) const;
  DLL_API void draw_triangle_fan(size_t num_primitives) const;
  DLL_API void draw_triangles(size_t num_primitives) const;
  DLL_API void draw_triangle_strip_adjacency(size_t num_primitives) const;
  DLL_API void draw_triangles_adjacency(size_t num_primitives) const;
  DLL_API void draw_patches(size_t num_primitives) const;
};
//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
