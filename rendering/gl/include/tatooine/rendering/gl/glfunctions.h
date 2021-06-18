#ifndef TATOOINE_RENDERING_GL_GL_FUNCTIONS_WRAPPER_H
#define TATOOINE_RENDERING_GL_GL_FUNCTIONS_WRAPPER_H
//==============================================================================
#include <tatooine/rendering/gl/glincludes.h>

#include <tatooine/rendering/gl/errorcheck.h>
#include <tatooine/rendering/gl/mutexhandler.h>
#include <tatooine/rendering/gl/tostring.h>

#include <ostream>
#include <vector>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
class camera;
struct gl {
  static bool verbose;

 private:
  static std::ostream* out;

 public:
  DLL_API static auto output_stream() -> auto& { return out; }
  DLL_API static void set_output_stream(std::ostream& stream) { out = &stream; }

  //============================================================================
  // MISC
  //============================================================================
  DLL_API static void polygon_mode(GLenum face, GLenum mode);
  DLL_API static void cull_face(GLenum mode);
  DLL_API static void front_face(GLenum mode);
  DLL_API static void point_size(GLfloat size);
  //----------------------------------------------------------------------------
  DLL_API static void line_width(GLfloat width);
  //----------------------------------------------------------------------------
  DLL_API static void get_boolean_v(GLenum pname, GLboolean* params);
  DLL_API static void get_double_v(GLenum pname, GLdouble* params);
  DLL_API static void get_float_v(GLenum pname, GLfloat* params);
  DLL_API static void get_integer_v(GLenum pname, GLint* params);
  DLL_API static void get_integer64_v(GLenum pname, GLint64* params);
  //----------------------------------------------------------------------------
  DLL_API static void get_booleani_v(GLenum target, GLuint index,
                                     GLboolean* data);
  DLL_API static void get_integeri_v(GLenum target, GLuint index, GLint* data);
  DLL_API static void get_floati_v(GLenum target, GLuint index, GLfloat* data);
  DLL_API static void get_doublei_v(GLenum target, GLuint index,
                                    GLdouble* data);
  DLL_API static void get_integer64i_v(GLenum target, GLuint index,
                                       GLint64* data);
  //----------------------------------------------------------------------------
  DLL_API static void enable(GLenum cap);
  //----------------------------------------------------------------------------
  DLL_API static auto is_enabled(GLenum cap) -> GLboolean;
  //----------------------------------------------------------------------------
  DLL_API static void disable(GLenum cap);
  //----------------------------------------------------------------------------
  DLL_API static auto get_string(GLenum name) -> const GLubyte*;
  //----------------------------------------------------------------------------
  DLL_API static auto get_error() -> GLenum;
  //----------------------------------------------------------------------------
  DLL_API static void depth_func(GLenum func);
  //----------------------------------------------------------------------------
  DLL_API static void scissor(GLint x, GLint y, GLsizei width, GLsizei height);

  //============================================================================
  // BACKBUFFER RELATED
  //============================================================================
  DLL_API static void clear(GLbitfield mask);
  //----------------------------------------------------------------------------
  DLL_API static void clear_color(GLfloat red, GLfloat green, GLfloat blue,
                                  GLfloat alpha);
  //----------------------------------------------------------------------------
  DLL_API static void viewport(GLint x, GLint y, GLsizei width, GLsizei height);
  //----------------------------------------------------------------------------
  DLL_API static void viewport(const camera& cam);
  //----------------------------------------------------------------------------
  DLL_API static void flush();
  //----------------------------------------------------------------------------
  DLL_API static void depth_mask(GLboolean flag);
  //----------------------------------------------------------------------------
  DLL_API static void blend_func(GLenum sfactor, GLenum dfactor);
  DLL_API static void blend_func_i(GLuint buf, GLenum sfactor, GLenum dfactor);
  //----------------------------------------------------------------------------
  DLL_API static void blend_func_separate(GLenum srcRGB, GLenum dstRGB,
                                          GLenum srcAlpha, GLenum dstAlpha);
  DLL_API static void blend_func_separate_i(GLuint buf, GLenum srcRGB,
                                            GLenum dstRGB, GLenum srcAlpha,
                                            GLenum dstAlpha);
  //----------------------------------------------------------------------------
  DLL_API static void blend_equation(GLenum mode);
  DLL_API static void blend_equation_i(GLuint buf, GLenum mode);
  //----------------------------------------------------------------------------
  DLL_API static void blend_equation_separate(GLenum modeRGB, GLenum modeAlpha);
  DLL_API static void blend_equation_separate_i(GLuint buf, GLenum modeRGB,
                                                GLenum modeAlpha);

  //============================================================================
  // VERTEXBUFFER RELATED
  //============================================================================
  DLL_API static void enable_vertex_attrib_array(GLuint index);
  //----------------------------------------------------------------------------
  DLL_API static void disable_vertex_attrib_array(GLuint index);
  //----------------------------------------------------------------------------
  DLL_API static void enable_vertex_attrib_array(GLuint vaobj, GLuint index);
  //----------------------------------------------------------------------------
  DLL_API static void vertex_attrib_pointer(GLuint index, GLint size,
                                            GLenum type, GLboolean normalized,
                                            GLsizei       stride,
                                            const GLvoid* pointer);
  //----------------------------------------------------------------------------
  DLL_API static void vertex_attrib_i_pointer(GLuint index, GLint size,
                                              GLenum type, GLsizei stride,
                                              const GLvoid* pointer);
  //----------------------------------------------------------------------------
  DLL_API static void vertex_attrib_l_pointer(GLuint index, GLint size,
                                              GLenum type, GLsizei stride,
                                              const GLvoid* pointer);
  //----------------------------------------------------------------------------
  DLL_API static void draw_arrays(GLenum mode, GLint first, GLsizei count);
  //============================================================================
  // VERTEXARRAY RELATED
  //============================================================================
  DLL_API static void create_vertex_arrays(GLsizei n, GLuint* arrays);
  //----------------------------------------------------------------------------
  DLL_API static void delete_vertex_arrays(GLsizei n, GLuint* arrays);
  //----------------------------------------------------------------------------
  DLL_API static void bind_vertex_array(GLuint array);
  //----------------------------------------------------------------------------
  DLL_API static void draw_elements(GLenum mode, GLsizei count, GLenum type,
                                    const GLvoid* indices);
  //============================================================================
  // BUFFER RELATED
  //============================================================================
  DLL_API static void buffer_data(GLenum target, GLsizeiptr size,
                                  const void* data, GLenum usage);
  //----------------------------------------------------------------------------
  DLL_API static void named_buffer_data(GLuint buffer, GLsizeiptr size,
                                        const void* data, GLenum usage);
  //----------------------------------------------------------------------------
  DLL_API static void bind_buffer(GLenum target, GLuint buffer);
  //----------------------------------------------------------------------------
  DLL_API static void bind_buffer_base(GLenum target, GLuint index,
                                       GLuint buffer);

  //----------------------------------------------------------------------------
  DLL_API static void create_buffers(GLsizei n, GLuint* buffers);
  //----------------------------------------------------------------------------
  DLL_API static void delete_buffers(GLsizei n, GLuint* buffers);
  //----------------------------------------------------------------------------
  DLL_API static void copy_named_buffer_sub_data(GLuint   readBuffer,
                                                 GLuint   writeBuffer,
                                                 GLintptr readOffset,
                                                 GLintptr writeOffset,
                                                 GLsizei  size);
  //----------------------------------------------------------------------------
  DLL_API static auto map_buffer(GLenum target, GLenum access) -> void*;
  //----------------------------------------------------------------------------
  DLL_API static auto map_named_buffer(GLuint buffer, GLenum access) -> void*;
  //----------------------------------------------------------------------------
  DLL_API static auto map_buffer_range(GLenum target, GLintptr offset,
                                       GLsizeiptr length, GLbitfield access)
      -> void*;
  //----------------------------------------------------------------------------
  DLL_API static auto map_named_buffer_range(GLuint buffer, GLintptr offset,
                                             GLsizei length, GLbitfield access)
      -> void*;
  //----------------------------------------------------------------------------
  template <typename T>
  static void named_buffer_data(GLuint buffer, GLsizei size, const T* data,
                                GLenum usage) {
    if (verbose) {
      if constexpr (std::is_arithmetic_v<T>) {
        *out << "glNamedBufferData(" << buffer << ", " << size << ", "
             << "[";
        *out << data[0];
        for (GLsizei i = 1; i < std::min<GLsizei>(size / sizeof(T), 3); ++i) {
          *out << ", " << data[i];
        }
        if (size / sizeof(T) > 3) {
          *out << ", ...";
        }
        *out << "], " << to_string(usage) << ")\n";
      } else {
        *out << "glNamedBufferData(" << buffer << ", " << size << ", " << data
             << ", " << to_string(usage) << ")\n";
      }
    }
    glNamedBufferData(buffer, size, data, usage);
    auto err = gl::get_error();
    if (err != GL_NO_ERROR) {
      auto err_str = gl_error_to_string(err);
      if (err == GL_INVALID_VALUE) {
        throw gl_error{"glNamedBufferData", err_str +
                                                " (size should be negative: " +
                                                std::to_string(size) + ")"};
      }
      throw gl_error{"glNamedBufferData", err_str};
    }
  }
  //----------------------------------------------------------------------------
  DLL_API static auto unmap_buffer(GLenum target) -> GLboolean;
  //----------------------------------------------------------------------------
  DLL_API static auto unmap_named_buffer(GLuint buffer) -> GLboolean;
  //----------------------------------------------------------------------------
  DLL_API static void buffer_sub_data(GLenum target, GLintptr offset,
                                      GLsizeiptr size, const GLvoid* data);
  //----------------------------------------------------------------------------
  DLL_API static void named_buffer_sub_data(GLuint buffer, GLintptr offset,
                                            GLsizei size, const void* data);
  //----------------------------------------------------------------------------
  DLL_API static void get_buffer_parameter_iv(GLenum target, GLenum value,
                                              GLint* data);
  //----------------------------------------------------------------------------
  DLL_API static void clear_named_buffer_data(GLuint buffer,
                                              GLenum internalformat,
                                              GLenum format, GLenum type,
                                              const void* data);
  //============================================================================
  // SHADER RELATED
  //============================================================================
  DLL_API static auto create_program() -> GLuint;
  //----------------------------------------------------------------------------
  DLL_API static void attach_shader(GLuint program, GLuint shader);
  //----------------------------------------------------------------------------
  DLL_API static void link_program(GLuint program);
  //----------------------------------------------------------------------------
  DLL_API static void delete_program(GLuint program);
  //----------------------------------------------------------------------------
  DLL_API static void use_program(GLuint program);
  //----------------------------------------------------------------------------
  DLL_API static auto create_shader(GLenum shaderType) -> GLuint;
  //----------------------------------------------------------------------------
  DLL_API static void shader_source(GLuint shader, GLsizei count,
                                    const GLchar** string, const GLint* length);
  //----------------------------------------------------------------------------
  DLL_API static void compile_shader(GLuint shader);
  //----------------------------------------------------------------------------
  DLL_API static void delete_shader(GLuint shader);
  //----------------------------------------------------------------------------
  DLL_API static void dispatch_compute(GLuint num_groups_x, GLuint num_groups_y,
                                       GLuint num_groups_z);
  //----------------------------------------------------------------------------
  DLL_API static void get_shader_iv(GLuint shader, GLenum pname, GLint* params);
  //----------------------------------------------------------------------------
  DLL_API static auto get_shader_info_log_length(GLuint shader) -> GLint;
  //----------------------------------------------------------------------------
  DLL_API static void get_shader_info_log(GLuint shader, GLsizei maxLength,
                                          GLsizei* length, GLchar* infoLog);
  //----------------------------------------------------------------------------
  DLL_API static auto get_shader_info_log(GLuint shader, GLsizei maxLength)
      -> std::string;
  //----------------------------------------------------------------------------
  DLL_API static auto get_shader_info_log(GLuint shader) -> std::string;
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_1f(GLuint program, GLint location,
                                         GLfloat v0);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_2f(GLuint program, GLint location,
                                         GLfloat v0, GLfloat v1);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_3f(GLuint program, GLint location,
                                         GLfloat v0, GLfloat v1, GLfloat v2);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_4f(GLuint program, GLint location,
                                         GLfloat v0, GLfloat v1, GLfloat v2,
                                         GLfloat v3);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_1i(GLuint program, GLint location,
                                         GLint v0);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_2i(GLuint program, GLint location,
                                         GLint v0, GLint v1);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_3i(GLuint program, GLint location,
                                         GLint v0, GLint v1, GLint v2);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_4i(GLuint program, GLint location,
                                         GLint v0, GLint v1, GLint v2,
                                         GLint v3);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_1ui(GLuint program, GLint location,
                                          GLuint v0);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_2ui(GLuint program, GLint location,
                                          GLuint v0, GLuint v1);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_3ui(GLuint program, GLint location,
                                          GLuint v0, GLuint v1, GLuint v2);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_4ui(GLuint program, GLint location,
                                          GLuint v0, GLuint v1, GLuint v2,
                                          GLuint v3);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_1fv(GLuint program, GLint location,
                                          GLsizei count, const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_2fv(GLuint program, GLint location,
                                          GLsizei count, const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_3fv(GLuint program, GLint location,
                                          GLsizei count, const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_4fv(GLuint program, GLint location,
                                          GLsizei count, const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_1iv(GLuint program, GLint location,
                                          GLsizei count, const GLint* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_2iv(GLuint program, GLint location,
                                          GLsizei count, const GLint* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_3iv(GLuint program, GLint location,
                                          GLsizei count, const GLint* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_4iv(GLuint program, GLint location,
                                          GLsizei count, const GLint* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_1uiv(GLuint program, GLint location,
                                           GLsizei count, const GLuint* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_2uiv(GLuint program, GLint location,
                                           GLsizei count, const GLuint* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_3uiv(GLuint program, GLint location,
                                           GLsizei count, const GLuint* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_4uiv(GLuint program, GLint location,
                                           GLsizei count, const GLuint* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_matrix_2fv(GLuint program, GLint location,
                                                 GLsizei        count,
                                                 GLboolean      transpose,
                                                 const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_matrix_3fv(GLuint program, GLint location,
                                                 GLsizei        count,
                                                 GLboolean      transpose,
                                                 const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_matrix_4fv(GLuint program, GLint location,
                                                 GLsizei        count,
                                                 GLboolean      transpose,
                                                 const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_matrix_2x3fv(GLuint         program,
                                                   GLint          location,
                                                   GLsizei        count,
                                                   GLboolean      transpose,
                                                   const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_matrix_3x2fv(GLuint         program,
                                                   GLint          location,
                                                   GLsizei        count,
                                                   GLboolean      transpose,
                                                   const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_matrix_2x4fv(GLuint         program,
                                                   GLint          location,
                                                   GLsizei        count,
                                                   GLboolean      transpose,
                                                   const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_matrix_4x2fv(GLuint         program,
                                                   GLint          location,
                                                   GLsizei        count,
                                                   GLboolean      transpose,
                                                   const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_matrix_3x4fv(GLuint         program,
                                                   GLint          location,
                                                   GLsizei        count,
                                                   GLboolean      transpose,
                                                   const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void program_uniform_matrix_4x3fv(GLuint         program,
                                                   GLint          location,
                                                   GLsizei        count,
                                                   GLboolean      transpose,
                                                   const GLfloat* value);
  //----------------------------------------------------------------------------
  DLL_API static void get_program_iv(GLuint program, GLenum pname,
                                     GLint* params);
  //----------------------------------------------------------------------------
  DLL_API static void get_program_info_log(GLuint program, GLsizei maxLength,
                                           GLsizei* length, GLchar* infoLog);
  //----------------------------------------------------------------------------
  DLL_API static auto get_uniform_location(GLuint program, const GLchar* name)
      -> GLint;
  //============================================================================
  // TEXTURE RELATED
  //============================================================================
  DLL_API static void create_textures(GLenum target, GLsizei n,
                                      GLuint* textures);
  //----------------------------------------------------------------------------
  DLL_API static void delete_textures(GLsizei n, GLuint* textures);
  //----------------------------------------------------------------------------
  DLL_API static void active_texture(GLenum texture);
  //----------------------------------------------------------------------------
  DLL_API static void bind_texture(GLenum target, GLuint texture);
  //----------------------------------------------------------------------------
  DLL_API static void bind_image_texture(GLuint unit, GLuint texture,
                                         GLint level, GLboolean layered,
                                         GLint layer, GLenum access,
                                         GLenum format);
  //----------------------------------------------------------------------------
  DLL_API static void tex_image_1d(GLenum target, GLint level,
                                   GLint internal_format, GLsizei width,
                                   GLint border, GLenum format, GLenum type,
                                   const GLvoid* data);
  //----------------------------------------------------------------------------
  DLL_API static void tex_image_2d(GLenum target, GLint level,
                                   GLint internal_format, GLsizei width,
                                   GLsizei height, GLint border, GLenum format,
                                   GLenum type, const GLvoid* data);
  //----------------------------------------------------------------------------
  DLL_API static void tex_sub_image_2d(GLenum target, GLint level,
                                       GLint xoffset, GLint yoffset,
                                       GLsizei width, GLsizei height,
                                       GLenum format, GLenum type,
                                       const GLvoid* pixels);
  //----------------------------------------------------------------------------
  DLL_API static void texture_sub_image_1d(GLuint texture, GLint level,
                                           GLint xoffset, GLsizei width,
                                           GLenum format, GLenum type,
                                           const void* pixels);
  //----------------------------------------------------------------------------
  DLL_API static void texture_sub_image_2d(GLuint texture, GLint level,
                                           GLint xoffset, GLint yoffset,
                                           GLsizei width, GLsizei height,
                                           GLenum format, GLenum type,
                                           const void* pixels);
  //----------------------------------------------------------------------------
  DLL_API static void texture_sub_image_3d(GLuint texture, GLint level,
                                           GLint xoffset, GLint yoffset,
                                           GLint zoffset, GLsizei width,
                                           GLsizei height, GLsizei depth,
                                           GLenum format, GLenum type,
                                           const void* pixels);
  //----------------------------------------------------------------------------
  DLL_API static void tex_image_3d(GLenum target, GLint level,
                                   GLint internal_format, GLsizei width,
                                   GLsizei height, GLsizei depth, GLint border,
                                   GLenum format, GLenum type,
                                   const GLvoid* data);
  //----------------------------------------------------------------------------
  DLL_API static void copy_image_sub_data(
      GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY,
      GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX,
      GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight,
      GLsizei srcDepth);
  //----------------------------------------------------------------------------
  DLL_API static void get_tex_image(GLenum target, GLint level, GLenum format,
                                    GLenum type, GLvoid* pixels);
  //----------------------------------------------------------------------------
  DLL_API static void get_n_tex_image(GLenum target, GLint level, GLenum format,
                                      GLenum type, GLsizei bufSize,
                                      void* pixels);
  //----------------------------------------------------------------------------
  DLL_API static void get_texture_image(GLuint texture, GLint level,
                                        GLenum format, GLenum type,
                                        GLsizei bufSize, void* pixels);
  //----------------------------------------------------------------------------
  DLL_API static void get_texture_sub_image(GLuint texture, GLint level,
                                            GLint xoffset, GLint yoffset,
                                            GLint zoffset, GLsizei width,
                                            GLsizei height, GLsizei depth,
                                            GLenum format, GLenum type,
                                            GLsizei bufSize, void* pixels);
  //----------------------------------------------------------------------------
  DLL_API static void tex_parameter_f(GLenum target, GLenum pname,
                                      GLfloat param);
  //----------------------------------------------------------------------------
  DLL_API static void tex_parameter_i(GLenum target, GLenum pname, GLint param);
  //----------------------------------------------------------------------------
  DLL_API static void texture_parameter_f(GLuint texture, GLenum pname,
                                          GLfloat param);
  //----------------------------------------------------------------------------
  DLL_API static void texture_parameter_i(GLuint texture, GLenum pname,
                                          GLint param);
  //----------------------------------------------------------------------------
  DLL_API static void tex_parameter_fv(GLenum target, GLenum pname,
                                       const GLfloat* params);
  //----------------------------------------------------------------------------
  DLL_API static void tex_parameter_iv(GLenum target, GLenum pname,
                                       const GLint* params);
  //----------------------------------------------------------------------------
  DLL_API static void tex_parameter_Iiv(GLenum target, GLenum pname,
                                        const GLint* params);
  //----------------------------------------------------------------------------
  DLL_API static void tex_parameter_Iuiv(GLenum target, GLenum pname,
                                         const GLuint* params);
  //----------------------------------------------------------------------------
  DLL_API static void texture_parameter_fv(GLuint texture, GLenum pname,
                                           const GLfloat* paramtexture);
  //----------------------------------------------------------------------------
  DLL_API static void texture_parameter_iv(GLuint texture, GLenum pname,
                                           const GLint* param);
  //----------------------------------------------------------------------------
  DLL_API static void texture_parameter_Iiv(GLuint texture, GLenum pname,
                                            const GLint* param);
  //----------------------------------------------------------------------------
  DLL_API static void texture_parameter_Iuiv(GLuint texture, GLenum pname,
                                             const GLuint* param);
  //----------------------------------------------------------------------------
  DLL_API static void clear_tex_image(GLuint texture, GLint level,
                                      GLenum format, GLenum type,
                                      const void* data);
  //----------------------------------------------------------------------------
  DLL_API static auto is_texture(GLuint texture) -> GLboolean;
  //----------------------------------------------------------------------------
  DLL_API static void bind_sampler(GLuint unit, GLuint sampler);
  //============================================================================
  // FRAMEBUFFER RELATED
  //============================================================================
  DLL_API static void create_framebuffers(GLsizei n, GLuint* ids);
  //----------------------------------------------------------------------------
  DLL_API static void delete_framebuffers(GLsizei n, GLuint* ids);
  //----------------------------------------------------------------------------
  DLL_API static void bind_framebuffer(GLenum target, GLuint framebuffer);
  //----------------------------------------------------------------------------
  DLL_API static void framebuffer_texture(GLenum target, GLenum attachment,
                                          GLuint texture, GLint level);
  //----------------------------------------------------------------------------
  DLL_API static void framebuffer_texture_1d(GLenum target, GLenum attachment,
                                             GLenum textarget, GLuint texture,
                                             GLint level);
  //----------------------------------------------------------------------------
  DLL_API static void framebuffer_texture_2d(GLenum target, GLenum attachment,
                                             GLenum textarget, GLuint texture,
                                             GLint level);
  //----------------------------------------------------------------------------
  DLL_API static void framebuffer_texture_3d(GLenum target, GLenum attachment,
                                             GLenum textarget, GLuint texture,
                                             GLint level, GLint layer);
  //----------------------------------------------------------------------------
  DLL_API static void named_framebuffer_texture(GLuint framebuffer,
                                                GLenum attachment,
                                                GLuint texture, GLint level);
  //----------------------------------------------------------------------------
  DLL_API static auto check_named_framebuffer_status(GLuint framebuffer,
                                                     GLenum target) -> GLenum;
  //----------------------------------------------------------------------------
  DLL_API static void named_framebuffer_draw_buffers(GLuint        framebuffer,
                                                     GLsizei       n,
                                                     const GLenum* bufs);
  //----------------------------------------------------------------------------
  DLL_API static void memory_barrier(GLbitfield barriers);
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
