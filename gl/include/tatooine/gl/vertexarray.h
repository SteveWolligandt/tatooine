#ifndef TATOOINE_GL_VERTEXARRAY_H
#define TATOOINE_GL_VERTEXARRAY_H
//==============================================================================
#include <tatooine/gl/dllexport.h>
#include <tatooine/gl/glfunctions.h>
#include <tatooine/gl/indexbuffer.h>
#include <tatooine/gl/primitive.h>
#include <tatooine/gl/type.h>
#include <tatooine/gl/vertexbuffer.h>

#include <iostream>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class vertexarray : public id_holder<GLuint> {
 public:
  using this_type = vertexarray;

  DLL_API              vertexarray();
  DLL_API              vertexarray(vertexarray const& other) = delete;
  DLL_API              vertexarray(vertexarray&& other)      = default;
  DLL_API vertexarray& operator=(vertexarray const& other)   = delete;
  DLL_API vertexarray& operator=(vertexarray&& other)        = default;
  DLL_API ~vertexarray();

  DLL_API void destroy_handle();
  DLL_API void bind() const;
  DLL_API void unbind() const;
  DLL_API void draw(Primitive primitive, GLsizei num_primitives) const;
  DLL_API void draw(GLsizei num_primitives) const;
  DLL_API void draw_points(GLsizei num_primitives) const;
  DLL_API void draw_line_strip(GLsizei num_primitives) const;
  DLL_API void draw_line_loop(GLsizei num_primitives) const;
  DLL_API void draw_lines(GLsizei num_primitives) const;
  DLL_API void draw_line_strip_adjacency(GLsizei num_primitives) const;
  DLL_API void draw_triangle_strip(GLsizei num_primitives) const;
  DLL_API void draw_triangle_fan(GLsizei num_primitives) const;
  DLL_API void draw_triangles(GLsizei num_primitives) const;
  DLL_API void draw_triangle_strip_adjacency(GLsizei num_primitives) const;
  DLL_API void draw_triangles_adjacency(GLsizei num_primitives) const;
  DLL_API void draw_patches(GLsizei num_primitives) const;
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
