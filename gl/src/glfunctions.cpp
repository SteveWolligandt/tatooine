#include <tatooine/gl/glfunctions.h>

#include <cassert>
#include <vector>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
// MISC
//==============================================================================
void polygon_mode(GLenum face, GLenum mode) {
  if (verbose) {
    verbose_ostream << "glPolygonMode(" << to_string(face) << ", "
                     << to_string(mode) << ")\n";
  }
  glPolygonMode(face, mode);
  gl_error_check("glPolygonMode");
}
//------------------------------------------------------------------------------
void cull_face(GLenum mode) {
  if (verbose) {
    verbose_ostream << "glCullFace(" << to_string(mode) << ")\n";
  }
  glCullFace(mode);
  gl_error_check("glCullFace");
}
//------------------------------------------------------------------------------
void front_face(GLenum mode) {
  if (verbose) {
    verbose_ostream << "glFrontFace(" << to_string(mode) << ")\n";
  }
  glFrontFace(mode);
  gl_error_check("glFrontFace");
}
//------------------------------------------------------------------------------
void point_size(GLfloat size) {
  if (verbose) {
    verbose_ostream << "glPointSize(" << size << ")\n";
  }
  glPointSize(size);
  gl_error_check("glPointSize");
}
//------------------------------------------------------------------------------
void line_width(GLfloat width) {
  if (verbose) {
    verbose_ostream << "glLineWidth(" << width << ")\n";
  }
  glLineWidth(width);
  gl_error_check("glLineWidth");
}
//------------------------------------------------------------------------------
void get_boolean_v(GLenum pname, GLboolean* params) {
  if (verbose)
    verbose_ostream << "glGetBooleanv(" << to_string(pname) << ", " << params
                     << ")\n";
  glGetBooleanv(pname, params);
  gl_error_check("glGetBooleanv");
}
//------------------------------------------------------------------------------
void get_double_v(GLenum pname, GLdouble* params) {
  if (verbose)
    verbose_ostream << "glGetDoublev(" << to_string(pname) << ", " << params
                     << ")\n";
  glGetDoublev(pname, params);
  gl_error_check("glGetDoublev");
}

//------------------------------------------------------------------------------
void get_float_v(GLenum pname, GLfloat* params) {
  if (verbose)
    verbose_ostream << "glGetFloatv(" << to_string(pname) << ", " << params
                     << ")\n";
  glGetFloatv(pname, params);
  gl_error_check("glGetFloatv");
}

//------------------------------------------------------------------------------
void get_integer_v(GLenum pname, GLint* params) {
  if (verbose)
    verbose_ostream << "glGetIntegerv(" << to_string(pname) << ", " << params
                     << ")\n";
  glGetIntegerv(pname, params);
  gl_error_check("glGetIntergerv");
}
//------------------------------------------------------------------------------
void get_integer64_v(GLenum pname, GLint64* params) {
  if (verbose) {
    verbose_ostream << "glGetInteger64v(" << to_string(pname) << ", " << params
                     << ")\n";
  }
  glGetInteger64v(pname, params);
  gl_error_check("glGetInterger64v");
}
//------------------------------------------------------------------------------
void get_booleani_v(GLenum target, GLuint index, GLboolean* data) {
  if (verbose) {
    verbose_ostream << "glGetBooleani_v(" << to_string(target) << ", " << index
                     << ", " << data << ")\n";
  }
  glGetBooleani_v(target, index, data);
  gl_error_check("glGetBooleani_v");
}
//------------------------------------------------------------------------------
void get_integeri_v(GLenum target, GLuint index, GLint* data) {
  if (verbose) {
    verbose_ostream << "glGetIntegeri_v(" << to_string(target) << ", " << index
                     << ", " << data << ")\n";
  }
  glGetIntegeri_v(target, index, data);
  gl_error_check("glGetIntegeri_v");
}
//------------------------------------------------------------------------------
void get_floati_v(GLenum target, GLuint index, GLfloat* data) {
  if (verbose) {
    verbose_ostream << "glGetFloati_v(" << to_string(target) << ", " << index
                     << ", " << data << ")\n";
  }
  glGetFloati_v(target, index, data);
  gl_error_check("glGetFloati_v");
}
//------------------------------------------------------------------------------
void get_doublei_v(GLenum target, GLuint index, GLdouble* data) {
  if (verbose) {
    verbose_ostream << "glGetDoublei_v(" << to_string(target) << ", " << index
                     << ", " << data << ")\n";
  }
  glGetDoublei_v(target, index, data);
  gl_error_check("glGetDoublei_v");
}
//------------------------------------------------------------------------------
void get_integer64i_v(GLenum target, GLuint index, GLint64* data) {
  if (verbose) {
    verbose_ostream << "glGetInteger64i_v(" << to_string(target) << ", "
                     << index << ", " << data << ")\n";
  }
  glGetInteger64i_v(target, index, data);
  gl_error_check("glGetInteger64i_v");
}
//------------------------------------------------------------------------------
void enable(GLenum cap) {
  if (verbose) verbose_ostream << "glEnable(" << to_string(cap) << ")\n";
  glEnable(cap);
  gl_error_check("glEnable");
}

//------------------------------------------------------------------------------
GLboolean is_enabled(GLenum cap) {
  if (verbose) verbose_ostream << "glIsEnabled(" << to_string(cap) << ")\n";
  auto result = glIsEnabled(cap);
  gl_error_check("glIsEnabled");
  return result;
}

//------------------------------------------------------------------------------
void disable(GLenum cap) {
  if (verbose) verbose_ostream << "glDisable(" << to_string(cap) << ")\n";
  glDisable(cap);
  gl_error_check("glDisable");
}

//------------------------------------------------------------------------------
const GLubyte* get_string(GLenum name) {
  if (verbose) verbose_ostream << "glGetString\n";
  auto result = glGetString(name);
  gl_error_check("glGetString");
  return result;
}
//------------------------------------------------------------------------------
GLenum get_error() {
  // if (verbose) { verbose_ostream << "glGetError\n"; }
  return glGetError();
}
//------------------------------------------------------------------------------
void depth_func(GLenum func) {
  if (verbose) {
    verbose_ostream << "glDepthFunc(" << to_string(func) << ")\n";
  }
  glDepthFunc(func);
  gl_error_check("glDepthFunc");
}
//------------------------------------------------------------------------------
void scissor(GLint x, GLint y, GLsizei width, GLsizei height) {
  if (verbose) {
    verbose_ostream << "glScissor(" << x << ", " << y << ", " << width << ", "
                     << height << ")\n";
  }
  glScissor(x, y, width, height);
  gl_error_check("glScissor");
}

//==============================================================================
// BACKBUFFER RELATED
//==============================================================================
void clear(GLbitfield mask) {
  if (verbose) {
    verbose_ostream << "glClear(";
    bool write_pipe = false;
    if (mask & GL_COLOR_BUFFER_BIT) {
      verbose_ostream << "GL_COLOR_BUFFER_BIT";
      write_pipe = true;
    }
    if (mask & GL_DEPTH_BUFFER_BIT) {
      if (write_pipe) verbose_ostream << " | ";
      verbose_ostream << "GL_DEPTH_BUFFER_BIT";
      write_pipe = true;
    }
    if (mask & GL_STENCIL_BUFFER_BIT) {
      if (write_pipe) verbose_ostream << " | ";
      verbose_ostream << "GL_STENCIL_BUFFER_BIT";
      write_pipe = true;
    }
    if (mask & GL_ACCUM_BUFFER_BIT) {
      if (write_pipe) verbose_ostream << " | ";
      verbose_ostream << "GL_ACCUM_BUFFER_BIT";
    }
    verbose_ostream << ")\n";
  }
  glClear(mask);
  gl_error_check("glClear");
}

//------------------------------------------------------------------------------
void clear_color(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
  if (verbose)
    verbose_ostream << "glClearColor(" << red << ", " << green << ", " << blue
                     << ", " << alpha << ")\n";
  glClearColor(red, green, blue, alpha);
  gl_error_check("glClearColor");
}

//------------------------------------------------------------------------------
void viewport(GLint x, GLint y, GLsizei width, GLsizei height) {
  if (verbose) {
    verbose_ostream << "glViewport(" << x << ", " << y << ", " << width << ", "
                     << height << ")\n";
  }
  glViewport(x, y, width, height);
  gl_error_check("glViewport");
}
//------------------------------------------------------------------------------
void flush() {
  if (verbose) verbose_ostream << "glFlush\n";
  glFlush();
  gl_error_check("glFlush");
}
//------------------------------------------------------------------------------
void depth_mask(GLboolean flag) {
  if (verbose)
    verbose_ostream << "glDepthMask(" << (flag ? "true" : "false") << ")\n";
  glDepthMask(flag);
  gl_error_check("glDepthMask");
}

//------------------------------------------------------------------------------
void blend_func(GLenum sfactor, GLenum dfactor) {
  if (verbose)
    verbose_ostream << "glBlendFunc(" << sfactor << ", " << dfactor << ")\n";
  glBlendFunc(sfactor, dfactor);
  gl_error_check("glBlendFunc");
}

//------------------------------------------------------------------------------
void blend_func_i(GLuint buf, GLenum sfactor, GLenum dfactor) {
  if (verbose) {
    verbose_ostream << "glBlendFunci(" << buf << ", " << sfactor << ", "
                     << dfactor << ")\n";
  }
  glBlendFunci(buf, sfactor, dfactor);
  gl_error_check("glBlendFunci");
}
//------------------------------------------------------------------------------
void blend_func_separate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha,
                         GLenum dstAlpha) {
  if (verbose) {
    verbose_ostream << "glBlendFuncSeparate(" << to_string(srcRGB) << ", "
                     << to_string(dstRGB) << ", " << to_string(srcAlpha) << ", "
                     << to_string(dstAlpha) << ")\n";
  }
  glBlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha);
  gl_error_check("glBlendFuncSeparate");
}
//------------------------------------------------------------------------------
void blend_func_separate_i(GLuint buf, GLenum srcRGB, GLenum dstRGB,
                           GLenum srcAlpha, GLenum dstAlpha) {
  if (verbose) {
    verbose_ostream << "glBlendFuncSeparatei(" << buf << ", "
                     << to_string(srcRGB) << ", " << to_string(dstRGB) << ", "
                     << to_string(srcAlpha) << ", " << to_string(dstAlpha)
                     << ")\n";
  }
  glBlendFuncSeparatei(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
  gl_error_check("glBlendFuncSeparatei");
}
//------------------------------------------------------------------------------
void blend_equation(GLenum mode) {
  if (verbose) {
    verbose_ostream << "glBlendEquation(" << to_string(mode) << ")\n";
  }
  glBlendEquation(mode);
  gl_error_check("glBlendEquation");
}
//------------------------------------------------------------------------------
void blend_equation_i(GLuint buf, GLenum mode) {
  if (verbose) {
    verbose_ostream << "glBlendEquationi(" << buf << ", " << to_string(mode)
                     << ")\n";
  }
  glBlendEquationi(buf, mode);
  gl_error_check("glBlendEquationi");
}
//------------------------------------------------------------------------------
void blend_equation_separate(GLenum modeRGB, GLenum modeAlpha) {
  if (verbose) {
    verbose_ostream << "glBlendEquationSeparate(" << to_string(modeRGB) << ", "
                     << to_string(modeAlpha) << ")\n";
  }
  glBlendEquationSeparate(modeRGB, modeAlpha);
  gl_error_check("glBlendEquationSeparate");
}
//------------------------------------------------------------------------------
void blend_equation_separate_i(GLuint buf, GLenum modeRGB, GLenum modeAlpha) {
  if (verbose) {
    verbose_ostream << "glBlendEquationSeparatei(" << buf << ", "
                     << to_string(modeRGB) << ", " << to_string(modeAlpha)
                     << ")\n";
  }
  glBlendEquationSeparatei(buf, modeRGB, modeAlpha);
  gl_error_check("glBlendEquationSeparatei");
}
//==============================================================================
// VERTEXBUFFER RELATED
//==============================================================================
void enable_vertex_attrib_array(GLuint index) {
  if (verbose)
    verbose_ostream << "glEnableVertexAttribArray(" << index << ")\n";
  glEnableVertexAttribArray(index);
  gl_error_check("glEnableVertexAttribArray");
}

//------------------------------------------------------------------------------
void disable_vertex_attrib_array(GLuint index) {
  if (verbose)
    verbose_ostream << "glDisableVertexAttribArray(" << index << ")\n";
  glDisableVertexAttribArray(index);
  gl_error_check("glDisableVertexAttribArray");
}

//------------------------------------------------------------------------------
void enable_vertex_attrib_array(GLuint vaobj, GLuint index) {
  if (verbose)
    verbose_ostream << "glEnableVertexArrayAttrib(" << vaobj << ", " << index
                     << ")\n";
  glEnableVertexArrayAttrib(vaobj, index);
  gl_error_check("glEnableVertexArrayAttrib");
}

//------------------------------------------------------------------------------
// void disable_vertex_attrib_array(GLuint vaobj, GLuint index) {
//  if(verbose)verbose_ostream<<"glDisableVertexArrayAttrib\n";
//  glDisableVertexArrayAttrib(vaobj, index);
//  gl_error_check("glDisableVertexArrayAttrib");
//}

//------------------------------------------------------------------------------
void vertex_attrib_pointer(GLuint index, GLint size, GLenum type,
                           GLboolean normalized, GLsizei stride,
                           const GLvoid* pointer) {
  if (verbose) {
    verbose_ostream << "glVertexAttribPointer(" << index << ", " << size
                     << ", " << to_string(type) << ", "
                     << (normalized ? "true" : "false") << ", " << stride
                     << ", " << pointer << ")\n";
  }
  glVertexAttribPointer(index, size, type, normalized, stride, pointer);
  gl_error_check("glVertexAttribPointer");
}

//------------------------------------------------------------------------------
void vertex_attrib_i_pointer(GLuint index, GLint size, GLenum type,
                             GLsizei stride, const GLvoid* pointer) {
  if (verbose) verbose_ostream << "glVertexAttribIPointer\n";
  glVertexAttribIPointer(index, size, type, stride, pointer);
  gl_error_check("glVertexAttribIPointer");
}

//------------------------------------------------------------------------------
void vertex_attrib_l_pointer(GLuint index, GLint size, GLenum type,
                             GLsizei stride, const GLvoid* pointer) {
  if (verbose) verbose_ostream << "glVertexAttribLPointer\n";
  glVertexAttribLPointer(index, size, type, stride, pointer);
  gl_error_check("glVertexAttribLPointer");
}

//------------------------------------------------------------------------------
void draw_arrays(GLenum mode, GLint first, GLsizei count) {
  if (verbose)
    verbose_ostream << "glDrawArrays(" << to_string(mode) << ", " << first
                     << ", " << count << ")\n";
  glDrawArrays(mode, first, count);
  gl_error_check("glDrawArrays");
}

//==============================================================================
// VERTEXARRAY RELATED
//==============================================================================
void create_vertex_arrays(GLsizei n, GLuint* arr) {
  glCreateVertexArrays(n, arr);
  if (verbose) {
    verbose_ostream << "glCreateVertexArrays = [" << arr[0];
    for (GLsizei i = 1; i < n; ++i) {
      verbose_ostream << ", " << arr[i];
    }
    verbose_ostream << "]\n";
  }
  gl_error_check("glCreateVertexArrays");
}

//------------------------------------------------------------------------------
void delete_vertex_arrays(GLsizei n, GLuint* ids) {
  if (verbose) {
    verbose_ostream << "glDeleteVertexArrays[" << ids[0];
    for (GLsizei i = 1; i < n; ++i) {
      verbose_ostream << ", " << ids[i];
    }
    verbose_ostream << "]\n";
  }
  glDeleteVertexArrays(n, ids);
  gl_error_check("glDeleteVertexArrays");
}

//------------------------------------------------------------------------------
void bind_vertex_array(GLuint array) {
  if (verbose) verbose_ostream << "glBindVertexArray(" << array << ")\n";
  glBindVertexArray(array);
  gl_error_check("glBindVertexArray");
}

//------------------------------------------------------------------------------
void draw_elements(GLenum mode, GLsizei count, GLenum type,
                   const GLvoid* indices) {
  if (verbose)
    verbose_ostream << "glDrawElements(" << to_string(mode) << ", " << count
                     << ", " << to_string(type) << ", " << indices << ")\n";
  glDrawElements(mode, count, type, indices);
  gl_error_check("glDrawElements");
}

//==============================================================================
// BUFFER RELATED
//==============================================================================
void buffer_data(GLenum target, GLsizeiptr size, const void* data,
                 GLenum usage) {
  if (verbose) {
    verbose_ostream << "glBufferData(" << to_string(target) << ", " << size
                     << ", " << data << ", " << to_string(usage) << ")\n";
  }
  glBufferData(target, size, data, usage);
  gl_error_check("glBufferData");
}
//----------------------------------------------------------------------------
void named_buffer_data(GLuint buffer, GLsizeiptr size, const void* data,
                       GLenum usage) {
  if (verbose) {
    verbose_ostream << "glNamedBufferData(" << buffer << ", " << size << ", "
                     << data << ", " << to_string(usage) << ")\n";
  }
  glNamedBufferData(buffer, size, data, usage);
  gl_error_check("glNamedBufferData");
}
//----------------------------------------------------------------------------
void bind_buffer(GLenum target, GLuint buffer) {
  if (verbose) {
    verbose_ostream << "glBindBuffer(" << to_string(target) << ", " << buffer
                     << ")\n";
  }
  glBindBuffer(target, buffer);
  gl_error_check("glBindBuffer");
}

//------------------------------------------------------------------------------
void bind_buffer_base(GLenum target, GLuint index, GLuint buffer) {
  if (verbose)
    verbose_ostream << "glBindBufferBase(" << to_string(target) << ", "
                     << index << ", " << buffer << ")\n";
  glBindBufferBase(target, index, buffer);
  gl_error_check("glBindBufferBase");
}

//------------------------------------------------------------------------------
void create_buffers(GLsizei n, GLuint* ids) {
  glCreateBuffers(n, ids);
  if (verbose) {
    verbose_ostream << "glCreateBuffers(" << n << ", " << ids << ") = [ ";
    for (GLsizei i = 0; i < n; ++i)
      verbose_ostream << ids[i] << ' ';
    verbose_ostream << "]\n";
  }
  gl_error_check("glCreateBuffers");
}

//------------------------------------------------------------------------------
void delete_buffers(GLsizei n, GLuint* ids) {
  if (verbose) {
    verbose_ostream << "glDeleteBuffers[ ";
    for (GLsizei i = 0; i < n; ++i)
      verbose_ostream << ids[i] << ' ';
    verbose_ostream << "]\n";
  }
  glDeleteBuffers(n, ids);
  gl_error_check("glDeleteBuffers");
}

//------------------------------------------------------------------------------
void copy_named_buffer_sub_data(GLuint readBuffer, GLuint writeBuffer,
                                GLintptr readOffset, GLintptr writeOffset,
                                GLsizei size) {
  if (verbose) verbose_ostream << "glCopyNamedBufferSubData\n";
  glCopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset,
                           size);
  gl_error_check("glCopyNamedBufferSubData");
}

//------------------------------------------------------------------------------
void* map_named_buffer(GLuint buffer, GLenum access) {
  if (verbose)
    verbose_ostream << "glMapNamedBuffer(" << buffer << ", "
                     << to_string(access) << ")\n";
  auto result = glMapNamedBuffer(buffer, access);
  gl_error_check("glMapNamedBuffer");
  return result;
}

//------------------------------------------------------------------------------
void* map_named_buffer_range(GLuint buffer, GLintptr offset, GLsizei length,
                             GLbitfield access) {
  if (verbose)
    verbose_ostream << "glMapNamedBufferRange(" << buffer << ", " << offset
                     << ", " << length << ", " << map_access_to_string(access)
                     << ")\n";
  auto result = glMapNamedBufferRange(buffer, offset, length, access);
  gl_error_check("glMapNamedBufferRange");
  return result;
}

//------------------------------------------------------------------------------
GLboolean unmap_named_buffer(GLuint buffer) {
  if (verbose) verbose_ostream << "glUnmapNamedBuffer(" << buffer << ")\n";
  auto result = glUnmapNamedBuffer(buffer);
  gl_error_check("glUnmapNamedBuffer");
  return result;
}

//------------------------------------------------------------------------------
void named_buffer_sub_data(GLuint buffer, GLintptr offset, GLsizei size,
                           const void* data) {
  if (verbose)
    verbose_ostream << "glNamedBufferSubData(" << buffer << ", " << offset
                     << ", " << size << ", " << data << ")\n";
  glNamedBufferSubData(buffer, offset, size, data);
  gl_error_check("glNamedBufferSubData");
}
//------------------------------------------------------------------------------
void get_buffer_parameter_iv(GLenum target, GLenum value, GLint* data) {
  assert(target == GL_ARRAY_BUFFER || target == GL_ELEMENT_ARRAY_BUFFER);
  assert(value == GL_BUFFER_SIZE || value == GL_BUFFER_USAGE);
  if (verbose)
    verbose_ostream << "glGetBufferParameteriv(" << to_string(target) << ", "
                     << to_string(value) << ", " << data << ", " << data
                     << ")\n";
  glGetBufferParameteriv(target, value, data);
  gl_error_check("glGetBufferParameteriv");
}
//------------------------------------------------------------------------------
void clear_named_buffer_data(GLuint buffer, GLenum internalformat,
                             GLenum format, GLenum type, const void* data) {
  if (verbose) {
    verbose_ostream << "glClearNamedBufferData(" << buffer << ", "
                     << to_string(internalformat) << ", " << to_string(format)
                     << ", " << to_string(type) << ", " << data << ")\n";
  }
  glClearNamedBufferData(buffer, internalformat, format, type, data);
  gl_error_check("glClearNamedBufferData");
}
//==============================================================================
// SHADER RELATED
//==============================================================================
GLuint create_program() {
  auto result = glCreateProgram();
  if (verbose) verbose_ostream << "glCreateProgram() = " << result << "\n";
  gl_error_check("glCreateProgram");
  return result;
}

//------------------------------------------------------------------------------
void attach_shader(GLuint program, GLuint shader) {
  if (verbose) verbose_ostream << "glAttachShader\n";
  glAttachShader(program, shader);
  gl_error_check("glAttachShader");
}

//------------------------------------------------------------------------------
void link_program(GLuint program) {
  if (verbose) verbose_ostream << "glLinkProgram(" << program << ")\n";
  glLinkProgram(program);
  gl_error_check("glLinkProgram");
}

//------------------------------------------------------------------------------
void delete_program(GLuint program) {
  if (verbose) verbose_ostream << "glDeleteProgram(" << program << ")\n";
  glDeleteProgram(program);
  gl_error_check("glDeleteProgram");
}

//------------------------------------------------------------------------------
void use_program(GLuint program) {
  if (verbose) verbose_ostream << "glUseProgram(" << program << ")\n";
  glUseProgram(program);
  gl_error_check("glUseProgram");
}

//------------------------------------------------------------------------------
GLuint create_shader(GLenum shaderType) {
  auto result = glCreateShader(shaderType);
  if (verbose) verbose_ostream << "glCreateShader() = " << result << "\n";
  gl_error_check("glCreateShader");
  return result;
}

//------------------------------------------------------------------------------
void shader_source(GLuint shader, GLsizei count, const GLchar** string,
                   const GLint* length) {
  if (verbose) verbose_ostream << "glShaderSource\n";
  glShaderSource(shader, count, string, length);
  gl_error_check("glShaderSource");
}

//------------------------------------------------------------------------------
void compile_shader(GLuint shader) {
  if (verbose) verbose_ostream << "glCompileShader\n";
  glCompileShader(shader);
  gl_error_check("glCompileShader");
}

//------------------------------------------------------------------------------
void delete_shader(GLuint shader) {
  if (verbose) verbose_ostream << "glDeleteShader(" << shader << ")\n";
  glDeleteShader(shader);
  gl_error_check("glDeleteShader");
}

//------------------------------------------------------------------------------
void dispatch_compute(GLuint num_groups_x, GLuint num_groups_y,
                      GLuint num_groups_z) {
  if (verbose)
    verbose_ostream << "glDispatchCompute(" << num_groups_x << ", "
                     << num_groups_y << ", " << num_groups_z << ")\n";
  glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
  gl_error_check("glDispatchCompute");
}

//------------------------------------------------------------------------------
void get_shader_iv(GLuint shader, GLenum pname, GLint* params) {
  if (verbose) verbose_ostream << "glGetShaderiv\n";
  glGetShaderiv(shader, pname, params);
  gl_error_check("glGetShaderiv");
}

//------------------------------------------------------------------------------
GLint get_shader_info_log_length(GLuint shader) {
  GLint info_log_length;
  get_shader_iv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
  return info_log_length;
}

//------------------------------------------------------------------------------
void get_shader_info_log(GLuint shader, GLsizei maxLength, GLsizei* length,
                         GLchar* infoLog) {
  if (verbose) verbose_ostream << "glGetShaderInfoLog\n";
  glGetShaderInfoLog(shader, maxLength, length, infoLog);
  gl_error_check("glGetShaderInfoLog");
}

//------------------------------------------------------------------------------
std::string get_shader_info_log(GLuint shader, GLsizei maxLength) {
  auto log = std::string(maxLength, ' ');
  if (verbose) {
    verbose_ostream << "data\n";
  }
  get_shader_info_log(shader, maxLength, nullptr, log.data());
  return log;
}

//------------------------------------------------------------------------------
std::string get_shader_info_log(GLuint shader) {
  return get_shader_info_log(shader, get_shader_info_log_length(shader));
}

//------------------------------------------------------------------------------
void program_uniform_1f(GLuint program, GLint location, GLfloat v0) {
  if (verbose)
    verbose_ostream << "glProgramUniform1f(" << program << ", " << location
                     << ", " << v0 << ")\n";
  glProgramUniform1f(program, location, v0);
  gl_error_check("glProgramUniform1f");
}

//------------------------------------------------------------------------------
void program_uniform_2f(GLuint program, GLint location, GLfloat v0,
                        GLfloat v1) {
  if (verbose)
    verbose_ostream << "glProgramUniform2f(" << program << ", " << location
                     << ", " << v0 << ", " << v1 << ")\n";
  glProgramUniform2f(program, location, v0, v1);
  gl_error_check("glProgramUniform2f");
}

//------------------------------------------------------------------------------
void program_uniform_3f(GLuint program, GLint location, GLfloat v0, GLfloat v1,
                        GLfloat v2) {
  if (verbose)
    verbose_ostream << "glProgramUniform3f(" << program << ", " << location
                     << ", " << v0 << ", " << v1 << ", " << v2 << ")\n";
  glProgramUniform3f(program, location, v0, v1, v2);
  gl_error_check("glProgramUniform3f");
}

//------------------------------------------------------------------------------
void program_uniform_4f(GLuint program, GLint location, GLfloat v0, GLfloat v1,
                        GLfloat v2, GLfloat v3) {
  if (verbose)
    verbose_ostream << "glProgramUniform4f(" << program << ", " << location
                     << ", " << v0 << ", " << v1 << ", " << v2 << ", " << v3
                     << ")\n";
  glProgramUniform4f(program, location, v0, v1, v2, v3);
  gl_error_check("glProgramUniform4f");
}

//------------------------------------------------------------------------------
void program_uniform_1i(GLuint program, GLint location, GLint v0) {
  if (verbose)
    verbose_ostream << "glProgramUniform1i(" << program << ", " << location
                     << ", " << v0 << ")\n";
  glProgramUniform1i(program, location, v0);
  gl_error_check("glProgramUniform1i");
}

//------------------------------------------------------------------------------
void program_uniform_2i(GLuint program, GLint location, GLint v0, GLint v1) {
  if (verbose)
    verbose_ostream << "glProgramUniform2i(" << program << ", " << location
                     << ", " << v0 << ", " << v1 << ")\n";
  glProgramUniform2i(program, location, v0, v1);
  gl_error_check("glProgramUniform2i(" + std::to_string(program) + ", " +
                 std::to_string(location) + ", " + std::to_string(v0) + ", " +
                 std::to_string(v1) + ")");
}

//------------------------------------------------------------------------------
void program_uniform_3i(GLuint program, GLint location, GLint v0, GLint v1,
                        GLint v2) {
  if (verbose)
    verbose_ostream << "glProgramUniform3i(" << program << ", " << location
                     << ", " << v0 << ", " << v1 << ", " << v2 << ")\n";
  glProgramUniform3i(program, location, v0, v1, v2);
  gl_error_check("glProgramUniform3i");
}

//------------------------------------------------------------------------------
void program_uniform_4i(GLuint program, GLint location, GLint v0, GLint v1,
                        GLint v2, GLint v3) {
  if (verbose)
    verbose_ostream << "glProgramUniform4i(" << program << ", " << location
                     << ", " << v0 << ", " << v1 << ", " << v2 << ", " << v3
                     << ")\n";
  glProgramUniform4i(program, location, v0, v1, v2, v3);
  gl_error_check("glProgramUniform4i");
}

//------------------------------------------------------------------------------
void program_uniform_1ui(GLuint program, GLint location, GLuint v0) {
  if (verbose) {
    verbose_ostream << "glProgramUniform1ui(" << program << ", " << location
                     << ", " << v0 << ")\n";
  }
  glProgramUniform1ui(program, location, v0);
  gl_error_check("glProgramUniform1ui");
}

//------------------------------------------------------------------------------
void program_uniform_2ui(GLuint program, GLint location, GLuint v0, GLuint v1) {
  if (verbose)
    verbose_ostream << "glProgramUniform2ui(" << program << ", " << location
                     << ", " << v0 << ", " << v1 << ")\n";
  glProgramUniform2ui(program, location, v0, v1);
  gl_error_check("glProgramUniform2ui(" + std::to_string(program) + ", " +
                 std::to_string(location) + ", " + std::to_string(v0) + ", " +
                 std::to_string(v1) + ")");
}

//------------------------------------------------------------------------------
void program_uniform_3ui(GLuint program, GLint location, GLuint v0, GLuint v1,
                         GLuint v2) {
  if (verbose) verbose_ostream << "glProgramUniform3ui\n";
  glProgramUniform3ui(program, location, v0, v1, v2);
  gl_error_check("glProgramUniform3ui");
}

//------------------------------------------------------------------------------
void program_uniform_4ui(GLuint program, GLint location, GLuint v0, GLuint v1,
                         GLuint v2, GLuint v3) {
  if (verbose) verbose_ostream << "glProgramUniform4ui\n";
  glProgramUniform4ui(program, location, v0, v1, v2, v3);
  gl_error_check("glProgramUniform4ui");
}

//------------------------------------------------------------------------------
void program_uniform_1fv(GLuint program, GLint location, GLsizei count,
                         const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniform1fv\n";
  glProgramUniform1fv(program, location, count, value);
  gl_error_check("glProgramUniform1fv");
}

//------------------------------------------------------------------------------
void program_uniform_2fv(GLuint program, GLint location, GLsizei count,
                         const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniform2fv\n";
  glProgramUniform2fv(program, location, count, value);
  gl_error_check("glProgramUniform2fv");
}

//------------------------------------------------------------------------------
void program_uniform_3fv(GLuint program, GLint location, GLsizei count,
                         const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniform3fv\n";
  glProgramUniform3fv(program, location, count, value);
  gl_error_check("glProgramUniform3fv");
}

//------------------------------------------------------------------------------
void program_uniform_4fv(GLuint program, GLint location, GLsizei count,
                         const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniform4fv\n";
  glProgramUniform4fv(program, location, count, value);
  gl_error_check("glProgramUniform4fv");
}

//------------------------------------------------------------------------------
void program_uniform_1iv(GLuint program, GLint location, GLsizei count,
                         const GLint* value) {
  if (verbose) verbose_ostream << "glProgramUniform1iv\n";
  glProgramUniform1iv(program, location, count, value);
  gl_error_check("glProgramUniform1iv");
}

//------------------------------------------------------------------------------
void program_uniform_2iv(GLuint program, GLint location, GLsizei count,
                         const GLint* value) {
  if (verbose) verbose_ostream << "glProgramUniform2iv\n";
  glProgramUniform2iv(program, location, count, value);
  gl_error_check("glProgramUniform2iv");
}

//------------------------------------------------------------------------------
void program_uniform_3iv(GLuint program, GLint location, GLsizei count,
                         const GLint* value) {
  if (verbose) verbose_ostream << "glProgramUniform3iv\n";
  glProgramUniform3iv(program, location, count, value);
  gl_error_check("glProgramUniform3iv");
}

//------------------------------------------------------------------------------
void program_uniform_4iv(GLuint program, GLint location, GLsizei count,
                         const GLint* value) {
  if (verbose) verbose_ostream << "glProgramUniform4iv\n";
  glProgramUniform4iv(program, location, count, value);
  gl_error_check("glProgramUniform4iv");
}

//------------------------------------------------------------------------------
void program_uniform_1uiv(GLuint program, GLint location, GLsizei count,
                          const GLuint* value) {
  if (verbose) verbose_ostream << "glProgramUniform1uiv\n";
  glProgramUniform1uiv(program, location, count, value);
  gl_error_check("glProgramUniform1uiv");
}

//------------------------------------------------------------------------------
void program_uniform_2uiv(GLuint program, GLint location, GLsizei count,
                          const GLuint* value) {
  if (verbose) verbose_ostream << "glProgramUniform2uiv\n";
  glProgramUniform2uiv(program, location, count, value);
  gl_error_check("glProgramUniform2uiv");
}

//------------------------------------------------------------------------------
void program_uniform_3uiv(GLuint program, GLint location, GLsizei count,
                          const GLuint* value) {
  if (verbose) verbose_ostream << "glProgramUniform3uiv\n";
  glProgramUniform3uiv(program, location, count, value);
  gl_error_check("glProgramUniform3uiv");
}

//------------------------------------------------------------------------------
void program_uniform_4uiv(GLuint program, GLint location, GLsizei count,
                          const GLuint* value) {
  if (verbose) verbose_ostream << "glProgramUniform4uiv\n";
  glProgramUniform4uiv(program, location, count, value);
  gl_error_check("glProgramUniform4uiv");
}

//------------------------------------------------------------------------------
void program_uniform_matrix_2fv(GLuint program, GLint location, GLsizei count,
                                GLboolean transpose, const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniformMatrix2fv\n";
  glProgramUniformMatrix2fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix2fv");
}

//------------------------------------------------------------------------------
void program_uniform_matrix_3fv(GLuint program, GLint location, GLsizei count,
                                GLboolean transpose, const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniformMatrix3fv\n";
  glProgramUniformMatrix3fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix3fv");
}

//------------------------------------------------------------------------------
void program_uniform_matrix_4fv(GLuint program, GLint location, GLsizei count,
                                GLboolean transpose, const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniformMatrix4fv\n";
  glProgramUniformMatrix4fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix4fv");
}

//------------------------------------------------------------------------------
void program_uniform_matrix_2x3fv(GLuint program, GLint location, GLsizei count,
                                  GLboolean transpose, const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniformMatrix2x3fv\n";
  glProgramUniformMatrix2x3fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix2x3fv");
}

//------------------------------------------------------------------------------
void program_uniform_matrix_3x2fv(GLuint program, GLint location, GLsizei count,
                                  GLboolean transpose, const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniformMatrix3x2fv\n";
  glProgramUniformMatrix3x2fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix3x2fv");
}

//------------------------------------------------------------------------------
void program_uniform_matrix_2x4fv(GLuint program, GLint location, GLsizei count,
                                  GLboolean transpose, const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniformMatrix2x4fv\n";
  glProgramUniformMatrix2x4fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix2x4fv");
}

//------------------------------------------------------------------------------
void program_uniform_matrix_4x2fv(GLuint program, GLint location, GLsizei count,
                                  GLboolean transpose, const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniformMatrix4x2fv\n";
  glProgramUniformMatrix4x2fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix4x2fv");
}

//------------------------------------------------------------------------------
void program_uniform_matrix_3x4fv(GLuint program, GLint location, GLsizei count,
                                  GLboolean transpose, const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniformMatrix3x4fv\n";
  glProgramUniformMatrix3x4fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix3x4fv");
}

//------------------------------------------------------------------------------
void program_uniform_matrix_4x3fv(GLuint program, GLint location, GLsizei count,
                                  GLboolean transpose, const GLfloat* value) {
  if (verbose) verbose_ostream << "glProgramUniformMatrix4x3fv\n";
  glProgramUniformMatrix4x3fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix4x3fv");
}

//------------------------------------------------------------------------------
void get_program_iv(GLuint program, GLenum pname, GLint* params) {
  if (verbose) verbose_ostream << "glGetProgramiv\n";
  glGetProgramiv(program, pname, params);
  gl_error_check("glGetProgramiv");
}

//------------------------------------------------------------------------------
void get_program_info_log(GLuint program, GLsizei maxLength, GLsizei* length,
                          GLchar* infoLog) {
  if (verbose) verbose_ostream << "glGetProgramInfoLog\n";
  glGetProgramInfoLog(program, maxLength, length, infoLog);
  gl_error_check("glGetProgramInfoLog");
}

//------------------------------------------------------------------------------
GLint get_uniform_location(GLuint program, const GLchar* name) {
  auto result = glGetUniformLocation(program, name);
  if (verbose)
    verbose_ostream << "glGetUniformLocation(" << program << ", " << name
                     << ") = " << result << '\n';
  gl_error_check("glGetUniformLocation");
  return result;
}

//==============================================================================
// TEXTURE RELATED
//==============================================================================
void create_textures(GLenum target, GLsizei n, GLuint* textures) {
  glCreateTextures(target, n, textures);
  if (verbose) {
    verbose_ostream << "glCreateTextures(" << to_string(target) << ", " << n
                     << ", " << textures << ") = [ ";
    for (GLsizei i = 0; i < n; ++i)
      verbose_ostream << textures[i] << ' ';
    verbose_ostream << "]\n";
  }
  gl_error_check("glCreateTextures");
}

//------------------------------------------------------------------------------
void delete_textures(GLsizei n, GLuint* textures) {
  if (verbose) {
    verbose_ostream << "glDeleteTextures[ ";
    for (GLsizei i = 0; i < n; ++i)
      verbose_ostream << textures[i] << ' ';
    verbose_ostream << "]\n";
  }
  glDeleteTextures(n, textures);
  gl_error_check("glDeleteTextures");
}

//------------------------------------------------------------------------------
void active_texture(GLenum texture) {
  if (verbose)
    verbose_ostream << "glActiveTexture(" << texture - GL_TEXTURE0 << ")\n";
  glActiveTexture(texture);
  gl_error_check("glActiveTexture");
}

//------------------------------------------------------------------------------
void bind_texture(GLenum target, GLuint texture) {
  if (verbose)
    verbose_ostream << "glBindTexture(" << to_string(target) << ", " << texture
                     << ")\n";
  glBindTexture(target, texture);
  gl_error_check("glBindTexture");
}

//------------------------------------------------------------------------------
void bind_image_texture(GLuint unit, GLuint texture, GLint level,
                        GLboolean layered, GLint layer, GLenum access,
                        GLenum format) {
  if (verbose)
    verbose_ostream << "glBindImageTexture(" << unit << ", " << texture << ", "
                     << level << ", " << (layered ? "true" : "false") << ", "
                     << layer << ", " << to_string(access) << ", "
                     << to_string(format) << ")\n";
  glBindImageTexture(unit, texture, level, layered, layer, access, format);
  gl_error_check("glBindImageTexture");
}

//------------------------------------------------------------------------------
void tex_image_1d(GLenum target, GLint level, GLint internal_format,
                  GLsizei width, GLint border, GLenum format, GLenum type,
                  const GLvoid* data) {
  assert(width >= 0);
  // assert(width < GL_MAX_TEXTURE_SIZE);
  if (verbose)
    verbose_ostream << "glTexImage1D(" << to_string(target) << ", " << level
                     << ", " << to_string(internal_format) << ", " << width
                     << ", " << border << ", " << to_string(format) << ", "
                     << to_string(type) << ", " << data << ")\n";
  glTexImage1D(target, level, internal_format, width, border, format, type,
               data);
  gl_error_check("glTexImage1D");
}

//------------------------------------------------------------------------------
void tex_image_2d(GLenum target, GLint level, GLint internal_format,
                  GLsizei width, GLsizei height, GLint border, GLenum format,
                  GLenum type, const GLvoid* data) {
  assert(width >= 0);
  assert(height >= 0);
  // assert(width < GL_MAX_TEXTURE_SIZE);
  // assert(height < GL_MAX_TEXTURE_SIZE);

  if (verbose)
    verbose_ostream << "glTexImage2D(" << to_string(target) << ", " << level
                     << ", " << to_string(internal_format) << ", " << width
                     << ", " << height << ", " << border << ", "
                     << to_string(format) << ", " << to_string(type) << ", "
                     << data << ")\n";
  glTexImage2D(target, level, internal_format, width, height, border, format,
               type, data);
  gl_error_check("glTexImage2D");
}
//------------------------------------------------------------------------------
void tex_sub_image_2d(GLenum target, GLint level, GLint xoffset, GLint yoffset,
                      GLsizei width, GLsizei height, GLenum format, GLenum type,
                      const GLvoid* pixels) {
  if (verbose)
    verbose_ostream << "glTexSubImage2D(" << to_string(target) << ", " << level
                     << ", " << xoffset << ", " << yoffset << ", " << width
                     << ", " << height << ", " << to_string(format) << ", "
                     << to_string(type) << ", " << pixels << ")\n";
  glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type,
                  pixels);
  gl_error_check("glTexSubImage2D");
}
//------------------------------------------------------------------------------
void texture_sub_image_2d(GLuint texture, GLint level, GLint xoffset,
                          GLint yoffset, GLsizei width, GLsizei height,
                          GLenum format, GLenum type, const GLvoid* pixels) {
  if (verbose) {
    verbose_ostream << "glTextureSubImage2D(" << texture << ", " << level
                     << ", " << xoffset << ", " << yoffset << ", " << width
                     << ", " << height << ", " << to_string(format) << ", "
                     << to_string(type) << ", " << pixels << ")\n";
    glTextureSubImage2D(texture, level, xoffset, yoffset, width, height, format,
                        type, pixels);
  }
  gl_error_check("glTextureSubImage2D");
}

//------------------------------------------------------------------------------
void tex_image_3d(GLenum target, GLint level, GLint internal_format,
                  GLsizei width, GLsizei height, GLsizei depth, GLint border,
                  GLenum format, GLenum type, const GLvoid* data) {
  assert(width >= 0);
  assert(height >= 0);
  assert(depth >= 0);
  // assert(width < GL_MAX_TEXTURE_SIZE);
  // assert(height < GL_MAX_TEXTURE_SIZE);
  // assert(depth < GL_MAX_TEXTURE_SIZE);
  if (verbose)
    verbose_ostream << "glTexImage3D(" << to_string(target) << ", " << level
                     << ", " << to_string(internal_format) << ", " << width
                     << ", " << height << ", " << depth << ", " << border
                     << ", " << to_string(format) << ", " << to_string(type)
                     << ", " << data << ")\n";
  glTexImage3D(target, level, internal_format, width, height, depth, border,
               format, type, data);
  gl_error_check("glTexImage3D");
}

//------------------------------------------------------------------------------
void copy_image_sub_data(GLuint srcName, GLenum srcTarget, GLint srcLevel,
                         GLint srcX, GLint srcY, GLint srcZ, GLuint dstName,
                         GLenum dstTarget, GLint dstLevel, GLint dstX,
                         GLint dstY, GLint dstZ, GLsizei srcWidth,
                         GLsizei srcHeight, GLsizei srcDepth) {
  if (verbose)
    verbose_ostream << "glCopyImageSubData(" << srcName << ", "
                     << to_string(srcTarget) << ", " << srcLevel << ", " << srcX
                     << ", " << srcY << ", " << srcZ << ", " << dstName << ", "
                     << to_string(dstTarget) << ", " << dstLevel << ", " << dstX
                     << ", " << dstY << ", " << dstZ << ", " << srcWidth << ", "
                     << srcHeight << ", " << srcDepth << ")\n";
  glCopyImageSubData(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName,
                     dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight,
                     srcDepth);
  gl_error_check("glCopyImageSubData");
}

//------------------------------------------------------------------------------
void get_tex_image(GLenum target, GLint level, GLenum format, GLenum type,
                   GLvoid* pixels) {
  if (verbose) verbose_ostream << "glGetTexImage\n";
  glGetTexImage(target, level, format, type, pixels);
  gl_error_check("glGetTexImage");
}

//------------------------------------------------------------------------------
void get_n_tex_image(GLenum target, GLint level, GLenum format, GLenum type,
                     GLsizei bufSize, void* pixels) {
  if (verbose) verbose_ostream << "glGetnTexImage\n";
  glGetnTexImage(target, level, format, type, bufSize, pixels);
  gl_error_check("glGetnTexImage");
}

//------------------------------------------------------------------------------
void get_texture_image(GLuint texture, GLint level, GLenum format, GLenum type,
                       GLsizei bufSize, void* pixels) {
  if (verbose)
    verbose_ostream << "glGetTextureImage(" << texture << ", " << level << ", "
                     << format << ", " << type << ", " << bufSize << ", "
                     << pixels << ")\n";
  glGetTextureImage(texture, level, format, type, bufSize, pixels);
  gl_error_check("glGetTextureImage");
}
//------------------------------------------------------------------------------
void get_texture_sub_image(GLuint texture, GLint level, GLint xoffset,
                           GLint yoffset, GLint zoffset, GLsizei width,
                           GLsizei height, GLsizei depth, GLenum format,
                           GLenum type, GLsizei bufSize, void* pixels) {
  if (verbose) {
    verbose_ostream << "glGetTextureSubImage(" << texture << ", " << level
                     << ", " << xoffset << ", " << yoffset << ", " << zoffset
                     << ", " << width << ", " << height << ", " << depth << ", "
                     << format << ", " << type << ", " << bufSize << ", "
                     << pixels << ")\n";
  }
  glGetTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height,
                       depth, format, type, bufSize, pixels);
  gl_error_check("glGetTextureSubImage");
}

//------------------------------------------------------------------------------
void tex_parameter_f(GLenum target, GLenum pname, GLfloat param) {
  if (verbose) verbose_ostream << "glTexParameterf\n";
  glTexParameterf(target, pname, param);
  gl_error_check("glTexParameterf");
}

//------------------------------------------------------------------------------
void tex_parameter_i(GLenum target, GLenum pname, GLint param) {
  if (verbose) verbose_ostream << "glTexParameteri\n";
  glTexParameteri(target, pname, param);
  gl_error_check("glTexParameteri");
}

//------------------------------------------------------------------------------
void texture_parameter_f(GLuint texture, GLenum pname, GLfloat param) {
  if (verbose)
    verbose_ostream << "glTextureParameterf(" << texture << ", "
                     << to_string(pname) << ", " << param << ")\n";
  glTextureParameterf(texture, pname, param);
  gl_error_check("glTexParameterf");
}

//------------------------------------------------------------------------------
void texture_parameter_i(GLuint texture, GLenum pname, GLint param) {
  if (verbose)
    verbose_ostream << "glTextureParameteri(" << texture << ", "
                     << to_string(pname) << ", " << texparami_to_string(param)
                     << ")\n";
  glTextureParameteri(texture, pname, param);
  gl_error_check("glTexParameteri");
}

//------------------------------------------------------------------------------
void tex_parameter_fv(GLenum target, GLenum pname, const GLfloat* params) {
  if (verbose)
    verbose_ostream << "glTextureParameterfv(" << to_string(target) << ", "
                     << to_string(pname) << ", " << params << ")\n";
  glTexParameterfv(target, pname, params);
  gl_error_check("glTexParameterfv");
}

//------------------------------------------------------------------------------
void tex_parameter_iv(GLenum target, GLenum pname, const GLint* params) {
  if (verbose)
    verbose_ostream << "glTextureParameteriv(" << to_string(target) << ", "
                     << to_string(pname) << ", " << params << ")\n";
  glTexParameteriv(target, pname, params);
  gl_error_check("glTexParameteriv");
}

//------------------------------------------------------------------------------
void tex_parameter_Iiv(GLenum target, GLenum pname, const GLint* params) {
  if (verbose)
    verbose_ostream << "glTextureParameterIiv(" << to_string(target) << ", "
                     << to_string(pname) << ", " << params << ")\n";
  glTexParameterIiv(target, pname, params);
  gl_error_check("glTexParameterIiv");
}

//------------------------------------------------------------------------------
void tex_parameter_Iuiv(GLenum target, GLenum pname, const GLuint* params) {
  if (verbose)
    verbose_ostream << "glTextureParameterIuiv(" << to_string(target) << ", "
                     << to_string(pname) << ", " << params << ")\n";
  glTexParameterIuiv(target, pname, params);
  gl_error_check("glTexParameterIuiv");
}

//------------------------------------------------------------------------------
void texture_parameter_fv(GLuint texture, GLenum pname,
                          const GLfloat* paramtexture) {
  if (verbose)
    verbose_ostream << "glTextureParameterfv(" << texture << ", "
                     << to_string(pname) << ", " << paramtexture << ")\n";
  glTextureParameterfv(texture, pname, paramtexture);
  gl_error_check("glTextureParameterfv");
}

//------------------------------------------------------------------------------
void texture_parameter_iv(GLuint texture, GLenum pname, const GLint* param) {
  if (verbose)
    verbose_ostream << "glTextureParameterfv(" << texture << ", "
                     << to_string(pname) << ", " << param << ")\n";
  glTextureParameteriv(texture, pname, param);
  gl_error_check("glTextureParameteriv");
}

//------------------------------------------------------------------------------
void texture_parameter_Iiv(GLuint texture, GLenum pname, const GLint* param) {
  if (verbose)
    verbose_ostream << "glTextureParameterIiv(" << texture << ", "
                     << to_string(pname) << ", " << param << ")\n";
  glTextureParameterIiv(texture, pname, param);
  gl_error_check("glTextureParameterIiv");
}

//------------------------------------------------------------------------------
void texture_parameter_Iuiv(GLuint texture, GLenum pname, const GLuint* param) {
  if (verbose)
    verbose_ostream << "glTextureParameterIuiv(" << texture << ", "
                     << to_string(pname) << ", " << param << ")\n";
  glTextureParameterIuiv(texture, pname, param);
  gl_error_check("glTextureParameterIuiv");
}

//------------------------------------------------------------------------------
void clear_tex_image(GLuint texture, GLint level, GLenum format, GLenum type,
                     const void* data) {
  if (verbose)
    verbose_ostream << "glClearTexImage(" << texture << ", " << level << ", "
                     << to_string(format) << ", " << to_string(type) << ", "
                     << data << ")\n";
  glClearTexImage(texture, level, format, type, data);
  gl_error_check("glClearTexImage");
}
//------------------------------------------------------------------------------
auto is_texture(GLuint texture) -> GLboolean {
  if (verbose) verbose_ostream << "glIsTexture(" << texture << ")\n";
  auto const b = glIsTexture(texture);
  gl_error_check("glIsTexture");
  return b;
}
//------------------------------------------------------------------------------
void bind_sampler(GLuint unit, GLuint sampler) {
  if (verbose)
    verbose_ostream << "glBindSampler(" << unit << ", " << sampler << ")\n";
  glBindSampler(unit, sampler);
  gl_error_check("glBindSampler");
}
//==============================================================================
// FRAMEBUFFER RELATED
//==============================================================================
void create_framebuffers(GLsizei n, GLuint* ids) {
  glCreateFramebuffers(n, ids);
  if (verbose) {
    verbose_ostream << "glCreateFramebuffers(" << n << ", " << ids << ") = [ ";
    for (GLsizei i = 0; i < n; ++i)
      verbose_ostream << ids[i] << ' ';
    verbose_ostream << "]\n";
  }
  gl_error_check("glCreateFramebuffers");
}

//------------------------------------------------------------------------------
void delete_framebuffers(GLsizei n, GLuint* ids) {
  if (verbose) {
    verbose_ostream << "glDeleteFramebuffers[ ";
    for (GLsizei i = 0; i < n; ++i)
      verbose_ostream << ids[i] << ' ';
    verbose_ostream << "]\n";
  }
  glDeleteFramebuffers(n, ids);
  gl_error_check("glDeleteFramebuffers");
}

//------------------------------------------------------------------------------
void bind_framebuffer(GLenum target, GLuint framebuffer) {
  if (verbose)
    verbose_ostream << "glBindFramebuffer(" << to_string(target) << ", "
                     << framebuffer << ")\n";
  glBindFramebuffer(target, framebuffer);
  gl_error_check("glBindFramebuffer");
}

//------------------------------------------------------------------------------
void framebuffer_texture(GLenum target, GLenum attachment, GLuint texture,
                         GLint level) {
  if (verbose) verbose_ostream << "glFramebufferTexture\n";
  glFramebufferTexture(target, attachment, texture, level);
  gl_error_check("glFramebufferTexture");
}

//------------------------------------------------------------------------------
void framebuffer_texture_1d(GLenum target, GLenum attachment, GLenum textarget,
                            GLuint texture, GLint level) {
  if (verbose) verbose_ostream << "glFramebufferTexture1D\n";
  glFramebufferTexture1D(target, attachment, textarget, texture, level);
  gl_error_check("glFramebufferTexture1D");
}

//------------------------------------------------------------------------------
void framebuffer_texture_2d(GLenum target, GLenum attachment, GLenum textarget,
                            GLuint texture, GLint level) {
  if (verbose) verbose_ostream << "glFramebufferTexture2D\n";
  glFramebufferTexture2D(target, attachment, textarget, texture, level);
  gl_error_check("glFramebufferTexture2D");
}

//------------------------------------------------------------------------------
void framebuffer_texture_3d(GLenum target, GLenum attachment, GLenum textarget,
                            GLuint texture, GLint level, GLint layer) {
  if (verbose) verbose_ostream << "glFramebufferTexture3D\n";
  glFramebufferTexture3D(target, attachment, textarget, texture, level, layer);
  gl_error_check("glFramebufferTexture3D");
}

//------------------------------------------------------------------------------
void named_framebuffer_texture(GLuint framebuffer, GLenum attachment,
                               GLuint texture, GLint level) {
  if (verbose)
    verbose_ostream << "glNamedFramebufferTexture(" << framebuffer << ", "
                     << to_string(attachment) << ", " << texture << ", "
                     << level << ")\n";
  glNamedFramebufferTexture(framebuffer, attachment, texture, level);
  gl_error_check("glNamedFramebufferTexture");
}

//------------------------------------------------------------------------------
GLenum check_named_framebuffer_status(GLuint framebuffer, GLenum target) {
  if (verbose)
    verbose_ostream << "glCheckNamedFramebufferStatus(" << framebuffer << ", "
                     << to_string(target) << ")\n";
  auto result = glCheckNamedFramebufferStatus(framebuffer, target);
  gl_error_check("glCheckNamedFramebufferStatus");
  return result;
}
//------------------------------------------------------------------------------
void named_framebuffer_draw_buffers(GLuint framebuffer, GLsizei n,
                                    const GLenum* bufs) {
  if (verbose) {
    verbose_ostream << "glNamedFramebufferDrawBuffers(" << framebuffer << ", "
                     << n << ", {";
    for (GLsizei i = 0; i < n - 1; ++i) {
      verbose_ostream << to_string(bufs[i]) << ", ";
    }
    verbose_ostream << to_string(bufs[n - 1]) << "})\n";
  }
  glNamedFramebufferDrawBuffers(framebuffer, n, bufs);
  gl_error_check("glCheckNamedFramebufferStatus");
}
//------------------------------------------------------------------------------
void memory_barrier(GLbitfield barriers) {
  if (verbose) {
    verbose_ostream << "glMemoryBarrier(" << to_string(barriers) << ")\n";
  }
  glMemoryBarrier(barriers);
  gl_error_check("glMemoryBarrier");
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
