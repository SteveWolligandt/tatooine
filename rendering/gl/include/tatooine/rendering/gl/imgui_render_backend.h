#ifndef YAVIN_IMGUI_RENDER_BACKEND_H
#define YAVIN_IMGUI_RENDER_BACKEND_H
//==============================================================================
#include "glincludes.h"

#include <yavin/imgui_includes.h>

#include <iostream>

#include "imgui_shader.h"
#include "indexbuffer.h"
#include "texture.h"
#include "vertexarray.h"
#include "vertexbuffer.h"
//==============================================================================
namespace yavin {
//==============================================================================
struct imgui_render_backend {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  GLuint m_gl_version = 0;  // Extracted at runtime using GL_MAJOR_VERSION,
                            // GL_MINOR_VERSION queries.
  char m_glsl_version_string[32] =
      "";  // Specified by user or detected based on compile time GL settings.
  tex2rgba<std::uint8_t>  m_font_texture;
  imgui_shader m_shader;
  vertexbuffer<std::array<GLfloat, 2>, std::array<GLfloat, 2>,
               std::array<std::uint8_t, 4>>
              m_vbo;
  indexbuffer m_ibo;

  //----------------------------------------------------------------------------
  // ctor
  //----------------------------------------------------------------------------
 public:
  imgui_render_backend();

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  void setup_render_state(ImDrawData* draw_data, int fb_width, int fb_height,
                          vertexarray& vertex_array_object);
  //----------------------------------------------------------------------------
  /// OpenGL3 Render function.
  /// (this used to be set in io.RenderDrawListsFn and called by
  /// ImGui::Render(), but you can now call this directly from your main loop)
  /// Note that this implementation is little overcomplicated because we are
  /// saving/setting up/restoring every OpenGL state explicitly, in order to be
  /// able to run within any OpenGL engine that doesn't do so.
  void render_draw_data(ImDrawData* draw_data);
  //----------------------------------------------------------------------------
  bool create_fonts_texture();
  //----------------------------------------------------------------------------
  /// If you get an error please report on github. You may try different GL
  /// context version or GLSL version. See GL<>GLSL version table at the top of
  /// this file.
  bool check_shader(GLuint handle, const char* desc);
  //----------------------------------------------------------------------------
  /// If you get an error please report on GitHub. You may try different GL
  /// context version or GLSL version.
  bool check_program(GLuint handle, const char* desc);
  //----------------------------------------------------------------------------
  bool create_device_objects();
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
