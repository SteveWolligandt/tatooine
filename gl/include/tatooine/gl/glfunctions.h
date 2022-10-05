#ifndef TATOOINE_GL_GL_FUNCTIONS_WRAPPER_H
#define TATOOINE_GL_GL_FUNCTIONS_WRAPPER_H
//==============================================================================
#include <tatooine/gl/errorcheck.h>
#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/mutexhandler.h>
#include <tatooine/gl/tostring.h>

#include <iostream>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class camera;
static bool          verbose         = TATOOINE_GL_VERBOSE;
static std::ostream& verbose_ostream = std::cerr;
//============================================================================
// MISC
//============================================================================
DLL_API auto polygon_mode(GLenum face, GLenum mode) -> void;
DLL_API auto cull_face(GLenum mode) -> void;
DLL_API auto front_face(GLenum mode) -> void;
DLL_API auto point_size(GLfloat size) -> void;
//----------------------------------------------------------------------------
DLL_API auto line_width(GLfloat width) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_boolean_v(GLenum pname, GLboolean* params) -> void;
DLL_API auto get_double_v(GLenum pname, GLdouble* params) -> void;
DLL_API auto get_float_v(GLenum pname, GLfloat* params) -> void;
DLL_API auto get_integer_v(GLenum pname, GLint* params) -> void;
DLL_API auto get_integer64_v(GLenum pname, GLint64* params) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_booleani_v(GLenum target, GLuint index, GLboolean* data)
    -> void;
DLL_API auto get_integeri_v(GLenum target, GLuint index, GLint* data) -> void;
DLL_API auto get_floati_v(GLenum target, GLuint index, GLfloat* data) -> void;
DLL_API auto get_doublei_v(GLenum target, GLuint index, GLdouble* data) -> void;
DLL_API auto get_integer64i_v(GLenum target, GLuint index, GLint64* data)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto enable(GLenum cap) -> void;
//----------------------------------------------------------------------------
DLL_API auto is_enabled(GLenum cap) -> GLboolean;
//----------------------------------------------------------------------------
DLL_API auto disable(GLenum cap) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_string(GLenum name) -> const GLubyte*;
//----------------------------------------------------------------------------
DLL_API auto get_error() -> GLenum;
//----------------------------------------------------------------------------
DLL_API auto depth_func(GLenum func) -> void;
//----------------------------------------------------------------------------
DLL_API auto scissor(GLint x, GLint y, GLsizei width, GLsizei height) -> void;

//============================================================================
// BACKBUFFER RELATED
//============================================================================
DLL_API auto clear(GLbitfield mask) -> void;
//----------------------------------------------------------------------------
DLL_API auto clear_color(GLfloat red, GLfloat green, GLfloat blue,
                         GLfloat alpha) -> void;
//----------------------------------------------------------------------------
DLL_API auto viewport(GLint x, GLint y, GLsizei width, GLsizei height) -> void;
//----------------------------------------------------------------------------
DLL_API auto viewport(camera const& cam) -> void;
//----------------------------------------------------------------------------
DLL_API auto flush() -> void;
//----------------------------------------------------------------------------
DLL_API auto depth_mask(GLboolean flag) -> void;
//----------------------------------------------------------------------------
DLL_API auto blend_func(GLenum sfactor, GLenum dfactor) -> void;
DLL_API auto blend_func_i(GLuint buf, GLenum sfactor, GLenum dfactor) -> void;
//----------------------------------------------------------------------------
DLL_API auto blend_func_separate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha,
                                 GLenum dstAlpha) -> void;
DLL_API auto blend_func_separate_i(GLuint buf, GLenum srcRGB, GLenum dstRGB,
                                   GLenum srcAlpha, GLenum dstAlpha) -> void;
//----------------------------------------------------------------------------
DLL_API auto blend_equation(GLenum mode) -> void;
DLL_API auto blend_equation_i(GLuint buf, GLenum mode) -> void;
//----------------------------------------------------------------------------
DLL_API auto blend_equation_separate(GLenum modeRGB, GLenum modeAlpha) -> void;
DLL_API auto blend_equation_separate_i(GLuint buf, GLenum modeRGB,
                                       GLenum modeAlpha) -> void;

//============================================================================
// VERTEXBUFFER RELATED
//============================================================================
DLL_API auto enable_vertex_attrib_array(GLuint index) -> void;
//----------------------------------------------------------------------------
DLL_API auto disable_vertex_attrib_array(GLuint index) -> void;
//----------------------------------------------------------------------------
DLL_API auto enable_vertex_attrib_array(GLuint vaobj, GLuint index) -> void;
//----------------------------------------------------------------------------
DLL_API auto vertex_attrib_pointer(GLuint index, GLint size, GLenum type,
                                   GLboolean normalized, GLsizei stride,
                                   const GLvoid* pointer) -> void;
//----------------------------------------------------------------------------
DLL_API auto vertex_attrib_i_pointer(GLuint index, GLint size, GLenum type,
                                     GLsizei stride, const GLvoid* pointer)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto vertex_attrib_l_pointer(GLuint index, GLint size, GLenum type,
                                     GLsizei stride, const GLvoid* pointer)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto draw_arrays(GLenum mode, GLint first, GLsizei count) -> void;
//============================================================================
// VERTEXARRAY RELATED
//============================================================================
DLL_API auto create_vertex_arrays(GLsizei n, GLuint* arrays) -> void;
//----------------------------------------------------------------------------
DLL_API auto delete_vertex_arrays(GLsizei n, GLuint* arrays) -> void;
//----------------------------------------------------------------------------
DLL_API auto bind_vertex_array(GLuint array) -> void;
//----------------------------------------------------------------------------
DLL_API auto draw_elements(GLenum mode, GLsizei count, GLenum type,
                           const GLvoid* indices) -> void;
//============================================================================
// BUFFER RELATED
//============================================================================
DLL_API auto buffer_data(GLenum target, GLsizeiptr size, const void* data,
                         GLenum usage) -> void;
//----------------------------------------------------------------------------
DLL_API auto named_buffer_data(GLuint buffer, GLsizeiptr size, const void* data,
                               GLenum usage) -> void;
//----------------------------------------------------------------------------
DLL_API auto bind_buffer(GLenum target, GLuint buffer) -> void;
//----------------------------------------------------------------------------
DLL_API auto bind_buffer_base(GLenum target, GLuint index, GLuint buffer)
    -> void;

//----------------------------------------------------------------------------
DLL_API auto create_buffers(GLsizei n, GLuint* buffers) -> void;
//----------------------------------------------------------------------------
DLL_API auto delete_buffers(GLsizei n, GLuint* buffers) -> void;
//----------------------------------------------------------------------------
DLL_API auto copy_named_buffer_sub_data(GLuint readBuffer, GLuint writeBuffer,
                                        GLintptr readOffset,
                                        GLintptr writeOffset, GLsizei size)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto map_buffer(GLenum target, GLenum access) -> void*;
//----------------------------------------------------------------------------
DLL_API auto map_named_buffer(GLuint buffer, GLenum access) -> void*;
//----------------------------------------------------------------------------
DLL_API auto map_buffer_range(GLenum target, GLintptr offset, GLsizeiptr length,
                              GLbitfield access) -> void*;
//----------------------------------------------------------------------------
DLL_API auto map_named_buffer_range(GLuint buffer, GLintptr offset,
                                    GLsizei length, GLbitfield access) -> void*;
//----------------------------------------------------------------------------
template <typename T>
auto named_buffer_data(GLuint buffer, GLsizei size, const T* data, GLenum usage)
    -> void {
  if (verbose) {
    if constexpr (std::is_arithmetic_v<T>) {
      verbose_ostream << "glNamedBufferData(" << buffer << ", " << size << ", "
                      << "[";
      verbose_ostream << data[0];
      for (GLsizei i = 1;
           i < std::min<GLsizei>(size / static_cast<GLsizei>(sizeof(T)), 3);
           ++i) {
        verbose_ostream << ", " << data[i];
      }
      if (size / static_cast<GLsizei>(sizeof(T)) > 3) {
        verbose_ostream << ", ...";
      }
      verbose_ostream << "], " << to_string(usage) << ")\n";
    } else {
      verbose_ostream << "glNamedBufferData(" << buffer << ", " << size << ", "
                      << data << ", " << to_string(usage) << ")\n";
    }
  }
  glNamedBufferData(buffer, size, data, usage);
  auto err = gl::get_error();
  if (err != GL_NO_ERROR) {
    auto err_str = gl_error_to_string(err);
    if (err == GL_INVALID_VALUE) {
      throw gl_error{
          "glNamedBufferData",
          err_str + " (size should be negative: " + std::to_string(size) + ")"};
    }
    throw gl_error{"glNamedBufferData", err_str};
  }
}
//----------------------------------------------------------------------------
DLL_API auto unmap_buffer(GLenum target) -> GLboolean;
//----------------------------------------------------------------------------
DLL_API auto unmap_named_buffer(GLuint buffer) -> GLboolean;
//----------------------------------------------------------------------------
DLL_API auto buffer_sub_data(GLenum target, GLintptr offset, GLsizeiptr size,
                             const GLvoid* data) -> void;
//----------------------------------------------------------------------------
DLL_API auto named_buffer_sub_data(GLuint buffer, GLintptr offset, GLsizei size,
                                   const void* data) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_buffer_parameter_iv(GLenum target, GLenum value, GLint* data)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto clear_named_buffer_data(GLuint buffer, GLenum internalformat,
                                     GLenum format, GLenum type,
                                     const void* data) -> void;
//============================================================================
// SHADER RELATED
//============================================================================
DLL_API auto create_program() -> GLuint;
//----------------------------------------------------------------------------
DLL_API auto attach_shader(GLuint program, GLuint shader) -> void;
//----------------------------------------------------------------------------
DLL_API auto link_program(GLuint program) -> void;
//----------------------------------------------------------------------------
DLL_API auto delete_program(GLuint program) -> void;
//----------------------------------------------------------------------------
DLL_API auto use_program(GLuint program) -> void;
//----------------------------------------------------------------------------
DLL_API auto create_shader(GLenum shaderType) -> GLuint;
//----------------------------------------------------------------------------
DLL_API auto shader_source(GLuint shader, GLsizei count, const GLchar** string,
                           const GLint* length) -> void;
//----------------------------------------------------------------------------
DLL_API auto compile_shader(GLuint shader) -> void;
//----------------------------------------------------------------------------
DLL_API auto delete_shader(GLuint shader) -> void;
//----------------------------------------------------------------------------
DLL_API auto dispatch_compute(GLuint num_groups_x, GLuint num_groups_y,
                              GLuint num_groups_z) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_shader_iv(GLuint shader, GLenum pname, GLint* params) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_shader_info_log_length(GLuint shader) -> GLint;
//----------------------------------------------------------------------------
DLL_API auto get_shader_info_log(GLuint shader, GLsizei maxLength,
                                 GLsizei* length, GLchar* infoLog) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_shader_info_log(GLuint shader, GLsizei maxLength)
    -> std::string;
//----------------------------------------------------------------------------
DLL_API auto get_shader_info_log(GLuint shader) -> std::string;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_1f(GLuint program, GLint location, GLfloat v0)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_2f(GLuint program, GLint location, GLfloat v0,
                                GLfloat v1) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_3f(GLuint program, GLint location, GLfloat v0,
                                GLfloat v1, GLfloat v2) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_4f(GLuint program, GLint location, GLfloat v0,
                                GLfloat v1, GLfloat v2, GLfloat v3) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_1i(GLuint program, GLint location, GLint v0)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_2i(GLuint program, GLint location, GLint v0,
                                GLint v1) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_3i(GLuint program, GLint location, GLint v0,
                                GLint v1, GLint v2) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_4i(GLuint program, GLint location, GLint v0,
                                GLint v1, GLint v2, GLint v3) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_1ui(GLuint program, GLint location, GLuint v0)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_2ui(GLuint program, GLint location, GLuint v0,
                                 GLuint v1) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_3ui(GLuint program, GLint location, GLuint v0,
                                 GLuint v1, GLuint v2) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_4ui(GLuint program, GLint location, GLuint v0,
                                 GLuint v1, GLuint v2, GLuint v3) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_1fv(GLuint program, GLint location, GLsizei count,
                                 const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_2fv(GLuint program, GLint location, GLsizei count,
                                 const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_3fv(GLuint program, GLint location, GLsizei count,
                                 const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_4fv(GLuint program, GLint location, GLsizei count,
                                 const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_1iv(GLuint program, GLint location, GLsizei count,
                                 const GLint* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_2iv(GLuint program, GLint location, GLsizei count,
                                 const GLint* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_3iv(GLuint program, GLint location, GLsizei count,
                                 const GLint* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_4iv(GLuint program, GLint location, GLsizei count,
                                 const GLint* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_1uiv(GLuint program, GLint location, GLsizei count,
                                  const GLuint* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_2uiv(GLuint program, GLint location, GLsizei count,
                                  const GLuint* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_3uiv(GLuint program, GLint location, GLsizei count,
                                  const GLuint* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_4uiv(GLuint program, GLint location, GLsizei count,
                                  const GLuint* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_matrix_2fv(GLuint program, GLint location,
                                        GLsizei count, GLboolean transpose,
                                        const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_matrix_3fv(GLuint program, GLint location,
                                        GLsizei count, GLboolean transpose,
                                        const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_matrix_4fv(GLuint program, GLint location,
                                        GLsizei count, GLboolean transpose,
                                        const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_matrix_2x3fv(GLuint program, GLint location,
                                          GLsizei count, GLboolean transpose,
                                          const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_matrix_3x2fv(GLuint program, GLint location,
                                          GLsizei count, GLboolean transpose,
                                          const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_matrix_2x4fv(GLuint program, GLint location,
                                          GLsizei count, GLboolean transpose,
                                          const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_matrix_4x2fv(GLuint program, GLint location,
                                          GLsizei count, GLboolean transpose,
                                          const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_matrix_3x4fv(GLuint program, GLint location,
                                          GLsizei count, GLboolean transpose,
                                          const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto program_uniform_matrix_4x3fv(GLuint program, GLint location,
                                          GLsizei count, GLboolean transpose,
                                          const GLfloat* value) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_program_iv(GLuint program, GLenum pname, GLint* params)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto get_program_info_log(GLuint program, GLsizei maxLength,
                                  GLsizei* length, GLchar* infoLog) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_uniform_location(GLuint program, const GLchar* name) -> GLint;
//============================================================================
// TEXTURE RELATED
//============================================================================
DLL_API auto create_textures(GLenum target, GLsizei n, GLuint* textures)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto delete_textures(GLsizei n, GLuint* textures) -> void;
//----------------------------------------------------------------------------
DLL_API auto active_texture(GLenum texture) -> void;
//----------------------------------------------------------------------------
DLL_API auto bind_texture(GLenum target, GLuint texture) -> void;
//----------------------------------------------------------------------------
DLL_API auto bind_image_texture(GLuint unit, GLuint texture, GLint level,
                                GLboolean layered, GLint layer, GLenum access,
                                GLenum format) -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_image_1d(GLenum target, GLint level, GLint internal_format,
                          GLsizei width, GLint border, GLenum format,
                          GLenum type, const GLvoid* data) -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_image_2d(GLenum target, GLint level, GLint internal_format,
                          GLsizei width, GLsizei height, GLint border,
                          GLenum format, GLenum type, const GLvoid* data)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_sub_image_2d(GLenum target, GLint level, GLint xoffset,
                              GLint yoffset, GLsizei width, GLsizei height,
                              GLenum format, GLenum type, const GLvoid* pixels)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto texture_sub_image_1d(GLuint texture, GLint level, GLint xoffset,
                                  GLsizei width, GLenum format, GLenum type,
                                  const void* pixels) -> void;
//----------------------------------------------------------------------------
DLL_API auto texture_sub_image_2d(GLuint texture, GLint level, GLint xoffset,
                                  GLint yoffset, GLsizei width, GLsizei height,
                                  GLenum format, GLenum type,
                                  const void* pixels) -> void;
//----------------------------------------------------------------------------
DLL_API auto texture_sub_image_3d(GLuint texture, GLint level, GLint xoffset,
                                  GLint yoffset, GLint zoffset, GLsizei width,
                                  GLsizei height, GLsizei depth, GLenum format,
                                  GLenum type, const void* pixels) -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_image_3d(GLenum target, GLint level, GLint internal_format,
                          GLsizei width, GLsizei height, GLsizei depth,
                          GLint border, GLenum format, GLenum type,
                          const GLvoid* data) -> void;
//----------------------------------------------------------------------------
DLL_API auto copy_image_sub_data(GLuint srcName, GLenum srcTarget,
                                 GLint srcLevel, GLint srcX, GLint srcY,
                                 GLint srcZ, GLuint dstName, GLenum dstTarget,
                                 GLint dstLevel, GLint dstX, GLint dstY,
                                 GLint dstZ, GLsizei srcWidth,
                                 GLsizei srcHeight, GLsizei srcDepth) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_tex_image(GLenum target, GLint level, GLenum format,
                           GLenum type, GLvoid* pixels) -> void;
//----------------------------------------------------------------------------
DLL_API auto get_n_tex_image(GLenum target, GLint level, GLenum format,
                             GLenum type, GLsizei bufSize, void* pixels)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto get_texture_image(GLuint texture, GLint level, GLenum format,
                               GLenum type, GLsizei bufSize, void* pixels)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto get_texture_sub_image(GLuint texture, GLint level, GLint xoffset,
                                   GLint yoffset, GLint zoffset, GLsizei width,
                                   GLsizei height, GLsizei depth, GLenum format,
                                   GLenum type, GLsizei bufSize, void* pixels)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_parameter_f(GLenum target, GLenum pname, GLfloat param)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_parameter_i(GLenum target, GLenum pname, GLint param) -> void;
//----------------------------------------------------------------------------
DLL_API auto texture_parameter_f(GLuint texture, GLenum pname, GLfloat param)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto texture_parameter_i(GLuint texture, GLenum pname, GLint param)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_parameter_fv(GLenum target, GLenum pname,
                              const GLfloat* params) -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_parameter_iv(GLenum target, GLenum pname, const GLint* params)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_parameter_Iiv(GLenum target, GLenum pname, const GLint* params)
    -> void;
//----------------------------------------------------------------------------
DLL_API auto tex_parameter_Iuiv(GLenum target, GLenum pname,
                                const GLuint* params) -> void;
//----------------------------------------------------------------------------
DLL_API auto texture_parameter_fv(GLuint texture, GLenum pname,
                                  const GLfloat* paramtexture) -> void;
//----------------------------------------------------------------------------
DLL_API auto texture_parameter_iv(GLuint texture, GLenum pname,
                                  const GLint* param) -> void;
//----------------------------------------------------------------------------
DLL_API auto texture_parameter_Iiv(GLuint texture, GLenum pname,
                                   const GLint* param) -> void;
//----------------------------------------------------------------------------
DLL_API auto texture_parameter_Iuiv(GLuint texture, GLenum pname,
                                    const GLuint* param) -> void;
//----------------------------------------------------------------------------
DLL_API auto clear_tex_image(GLuint texture, GLint level, GLenum format,
                             GLenum type, const void* data) -> void;
//----------------------------------------------------------------------------
DLL_API auto is_texture(GLuint texture) -> GLboolean;
//----------------------------------------------------------------------------
DLL_API auto bind_sampler(GLuint unit, GLuint sampler) -> void;
//============================================================================
// FRAMEBUFFER RELATED
//============================================================================
DLL_API auto create_framebuffers(GLsizei n, GLuint* ids) -> void;
//----------------------------------------------------------------------------
DLL_API auto delete_framebuffers(GLsizei n, GLuint* ids) -> void;
//----------------------------------------------------------------------------
DLL_API auto bind_framebuffer(GLenum target, GLuint framebuffer) -> void;
//----------------------------------------------------------------------------
DLL_API auto framebuffer_texture(GLenum target, GLenum attachment,
                                 GLuint texture, GLint level) -> void;
//----------------------------------------------------------------------------
DLL_API auto framebuffer_texture_1d(GLenum target, GLenum attachment,
                                    GLenum textarget, GLuint texture,
                                    GLint level) -> void;
//----------------------------------------------------------------------------
DLL_API auto framebuffer_texture_2d(GLenum target, GLenum attachment,
                                    GLenum textarget, GLuint texture,
                                    GLint level) -> void;
//----------------------------------------------------------------------------
DLL_API auto framebuffer_texture_3d(GLenum target, GLenum attachment,
                                    GLenum textarget, GLuint texture,
                                    GLint level, GLint layer) -> void;
//----------------------------------------------------------------------------
DLL_API auto named_framebuffer_texture(GLuint framebuffer, GLenum attachment,
                                       GLuint texture, GLint level) -> void;
//----------------------------------------------------------------------------
DLL_API auto check_named_framebuffer_status(GLuint framebuffer, GLenum target)
    -> GLenum;
//----------------------------------------------------------------------------
DLL_API auto named_framebuffer_draw_buffers(GLuint framebuffer, GLsizei n,
                                            const GLenum* bufs) -> void;
//----------------------------------------------------------------------------
DLL_API auto memory_barrier(GLbitfield barriers) -> void;
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
#ifdef Always
#undef Always
#endif
