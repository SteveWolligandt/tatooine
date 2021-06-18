#include <yavin/glfunctions.h>
#include <cassert>
//==============================================================================
namespace yavin {
//==============================================================================
// MISC
//==============================================================================
void gl::polygon_mode(GLenum face, GLenum mode) {
  if (verbose) {
    *out << "glPolygonMode(" << to_string(face) << ", " << to_string(mode)
         << ")\n";
  }
  glPolygonMode(face, mode);
  gl_error_check("glPolygonMode");
}
//------------------------------------------------------------------------------
void gl::cull_face(GLenum mode) {
  if (verbose) { *out << "glCullFace(" << to_string(mode) << ")\n"; }
  glCullFace(mode);
  gl_error_check("glCullFace");
}
//------------------------------------------------------------------------------
void gl::front_face(GLenum mode) {
  if (verbose) { *out << "glFrontFace(" << to_string(mode) << ")\n"; }
  glFrontFace(mode);
  gl_error_check("glFrontFace");
}
//------------------------------------------------------------------------------
void gl::point_size(GLfloat size) {
  if (verbose) { *out << "glPointSize(" << size << ")\n"; }
  glPointSize(size);
  gl_error_check("glPointSize");
}
//------------------------------------------------------------------------------
void gl::line_width(GLfloat width) {
  if (verbose) { *out << "glLineWidth(" << width << ")\n"; }
  glLineWidth(width);
  gl_error_check("glLineWidth");
}
//------------------------------------------------------------------------------
void gl::get_boolean_v(GLenum pname, GLboolean* params) {
  if (verbose)
    *out << "glGetBooleanv(" << to_string(pname) << ", " << params << ")\n";
  glGetBooleanv(pname, params);
  gl_error_check("glGetBooleanv");
}
//------------------------------------------------------------------------------
void gl::get_double_v(GLenum pname, GLdouble* params) {
  if (verbose)
    *out << "glGetDoublev(" << to_string(pname) << ", " << params << ")\n";
  glGetDoublev(pname, params);
  gl_error_check("glGetDoublev");
}

//------------------------------------------------------------------------------
void gl::get_float_v(GLenum pname, GLfloat* params) {
  if (verbose)
    *out << "glGetFloatv(" << to_string(pname) << ", " << params << ")\n";
  glGetFloatv(pname, params);
  gl_error_check("glGetFloatv");
}

//------------------------------------------------------------------------------
void gl::get_integer_v(GLenum pname, GLint* params) {
  if (verbose)
    *out << "glGetIntegerv(" << to_string(pname) << ", " << params << ")\n";
  glGetIntegerv(pname, params);
  gl_error_check("glGetIntergerv");
}
//------------------------------------------------------------------------------
void gl::get_integer64_v(GLenum pname, GLint64* params) {
  if (verbose) {
    *out << "glGetInteger64v(" << to_string(pname) << ", " << params << ")\n";
  }
  glGetInteger64v(pname, params);
  gl_error_check("glGetInterger64v");
}
//------------------------------------------------------------------------------
void gl::get_booleani_v(GLenum target, GLuint index, GLboolean* data) {
  if (verbose) {
    *out << "glGetBooleani_v(" << to_string(target) << ", " << index << ", "
         << data << ")\n";
  }
  glGetBooleani_v(target, index, data);
  gl_error_check("glGetBooleani_v");
}
//------------------------------------------------------------------------------
void gl::get_integeri_v(GLenum target, GLuint index, GLint* data) {
  if (verbose) {
    *out << "glGetIntegeri_v(" << to_string(target) << ", " << index << ", "
         << data << ")\n";
  }
  glGetIntegeri_v(target, index, data);
  gl_error_check("glGetIntegeri_v");
}
//------------------------------------------------------------------------------
void gl::get_floati_v(GLenum target, GLuint index, GLfloat* data) {
  if (verbose) {
    *out << "glGetFloati_v(" << to_string(target) << ", " << index << ", "
         << data << ")\n";
  }
  glGetFloati_v(target, index, data);
  gl_error_check("glGetFloati_v");
}
//------------------------------------------------------------------------------
void gl::get_doublei_v(GLenum target, GLuint index, GLdouble* data) {
  if (verbose) {
    *out << "glGetDoublei_v(" << to_string(target) << ", " << index << ", "
         << data << ")\n";
  }
  glGetDoublei_v(target, index, data);
  gl_error_check("glGetDoublei_v");
}
//------------------------------------------------------------------------------
void gl::get_integer64i_v(GLenum target, GLuint index, GLint64* data) {
  if (verbose) {
    *out << "glGetInteger64i_v(" << to_string(target) << ", " << index << ", "
         << data << ")\n";
  }
  glGetInteger64i_v(target, index, data);
  gl_error_check("glGetInteger64i_v");
}
//------------------------------------------------------------------------------
void gl::enable(GLenum cap) {
  if (verbose) *out << "glEnable(" << to_string(cap) << ")\n";
  glEnable(cap);
  gl_error_check("glEnable");
}

//------------------------------------------------------------------------------
GLboolean gl::is_enabled(GLenum cap) {
  if (verbose) *out << "glIsEnabled(" << to_string(cap) << ")\n";
  auto result = glIsEnabled(cap);
  gl_error_check("glIsEnabled");
  return result;
}

//------------------------------------------------------------------------------
void gl::disable(GLenum cap) {
  if (verbose) *out << "glDisable(" << to_string(cap) << ")\n";
  glDisable(cap);
  gl_error_check("glDisable");
}

//------------------------------------------------------------------------------
const GLubyte* gl::get_string(GLenum name) {
  if (verbose) *out << "glGetString\n";
  auto result = glGetString(name);
  gl_error_check("glGetString");
  return result;
}
//------------------------------------------------------------------------------
GLenum gl::get_error() {
  //if (verbose) { *out << "glGetError\n"; }
  return glGetError();
}
//------------------------------------------------------------------------------
void gl::depth_func(GLenum func) {
  if (verbose) { *out << "glDepthFunc(" << to_string(func) << ")\n"; }
  glDepthFunc(func);
  gl_error_check("glDepthFunc");
}
//------------------------------------------------------------------------------
void gl::scissor(GLint x, GLint y, GLsizei width, GLsizei height) {
  if (verbose) {
    *out << "glScissor(" << x << ", " << y << ", " << width << ", " << height
         << ")\n";
  }
  glScissor(x, y, width, height);
  gl_error_check("glScissor");
}

//==============================================================================
// BACKBUFFER RELATED
//==============================================================================
void gl::clear(GLbitfield mask) {
  if (verbose) {
    *out << "glClear(";
    bool write_pipe = false;
    if (mask & GL_COLOR_BUFFER_BIT) {
      *out << "GL_COLOR_BUFFER_BIT";
      write_pipe = true;
    }
    if (mask & GL_DEPTH_BUFFER_BIT) {
      if (write_pipe) *out << " | ";
      *out << "GL_DEPTH_BUFFER_BIT";
      write_pipe = true;
    }
    if (mask & GL_STENCIL_BUFFER_BIT) {
      if (write_pipe) *out << " | ";
      *out << "GL_STENCIL_BUFFER_BIT";
      write_pipe = true;
    }
    if (mask & GL_ACCUM_BUFFER_BIT) {
      if (write_pipe) *out << " | ";
      *out << "GL_ACCUM_BUFFER_BIT";
    }
    *out << ")\n";
  }
  glClear(mask);
  gl_error_check("glClear");
}

//------------------------------------------------------------------------------
void gl::clear_color(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
  if (verbose)
    *out << "glClearColor(" << red << ", " << green << ", " << blue << ", "
         << alpha << ")\n";
  glClearColor(red, green, blue, alpha);
  gl_error_check("glClearColor");
}

//------------------------------------------------------------------------------
void gl::viewport(GLint x, GLint y, GLsizei width, GLsizei height) {
  if (verbose) {
    *out << "glViewport(" << x << ", " << y << ", " << width << ", " << height
         << ")\n";
  }
  glViewport(x, y, width, height);
  gl_error_check("glViewport");
}
//------------------------------------------------------------------------------
void gl::flush() {
  if (verbose) *out << "glFlush\n";
  glFlush();
  gl_error_check("glFlush");
}
//------------------------------------------------------------------------------
void gl::depth_mask(GLboolean flag) {
  if (verbose) *out << "glDepthMask(" << (flag ? "true" : "false") << ")\n";
  glDepthMask(flag);
  gl_error_check("glDepthMask");
}

//------------------------------------------------------------------------------
void gl::blend_func(GLenum sfactor, GLenum dfactor) {
  if (verbose) *out << "glBlendFunc(" << sfactor << ", " << dfactor << ")\n";
  glBlendFunc(sfactor, dfactor);
  gl_error_check("glBlendFunc");
}

//------------------------------------------------------------------------------
void gl::blend_func_i(GLuint buf, GLenum sfactor, GLenum dfactor) {
  if (verbose) {
    *out << "glBlendFunci(" << buf << ", " << sfactor << ", " << dfactor
         << ")\n";
  }
  glBlendFunci(buf, sfactor, dfactor);
  gl_error_check("glBlendFunci");
}
//------------------------------------------------------------------------------
void gl::blend_func_separate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha,
                         GLenum dstAlpha) {
  if (verbose) {
    *out << "glBlendFuncSeparate(" << to_string(srcRGB) << ", "
         << to_string(dstRGB) << ", " << to_string(srcAlpha) << ", "
         << to_string(dstAlpha) << ")\n";
  }
  glBlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha);
  gl_error_check("glBlendFuncSeparate");
}
//------------------------------------------------------------------------------
void gl::blend_func_separate_i(GLuint buf, GLenum srcRGB, GLenum dstRGB,
                           GLenum srcAlpha, GLenum dstAlpha) {
  if (verbose) {
    *out << "glBlendFuncSeparatei(" << buf << ", " << to_string(srcRGB) << ", "
         << to_string(dstRGB) << ", " << to_string(srcAlpha) << ", "
         << to_string(dstAlpha) << ")\n";
  }
  glBlendFuncSeparatei(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
  gl_error_check("glBlendFuncSeparatei");
}
//------------------------------------------------------------------------------
void gl::blend_equation(GLenum mode) {
  if (verbose) {
    *out << "glBlendEquation(" << to_string(mode) << ")\n";
  }
  glBlendEquation(mode);
  gl_error_check("glBlendEquation");
}
//------------------------------------------------------------------------------
void gl::blend_equation_i(GLuint buf, GLenum mode) {
  if (verbose) {
    *out << "glBlendEquationi(" << buf << ", " << to_string(mode) << ")\n";
  }
  glBlendEquationi(buf, mode);
  gl_error_check("glBlendEquationi");
}
//------------------------------------------------------------------------------
void gl::blend_equation_separate(GLenum modeRGB, GLenum modeAlpha) {
  if (verbose) {
    *out << "glBlendEquationSeparate(" << to_string(modeRGB) << ", "
         << to_string(modeAlpha) << ")\n";
  }
  glBlendEquationSeparate(modeRGB, modeAlpha);
  gl_error_check("glBlendEquationSeparate");
}
//------------------------------------------------------------------------------
void gl::blend_equation_separate_i(GLuint buf, GLenum modeRGB,
                                   GLenum modeAlpha) {
  if (verbose) {
    *out << "glBlendEquationSeparatei(" << buf << ", " << to_string(modeRGB)
         << ", " << to_string(modeAlpha) << ")\n";
  }
  glBlendEquationSeparatei(buf, modeRGB, modeAlpha);
  gl_error_check("glBlendEquationSeparatei");
}
//==============================================================================
// VERTEXBUFFER RELATED
//==============================================================================
void gl::enable_vertex_attrib_array(GLuint index) {
  if (verbose) *out << "glEnableVertexAttribArray(" << index << ")\n";
  glEnableVertexAttribArray(index);
  gl_error_check("glEnableVertexAttribArray");
}

//------------------------------------------------------------------------------
void gl::disable_vertex_attrib_array(GLuint index) {
  if (verbose) *out << "glDisableVertexAttribArray(" << index << ")\n";
  glDisableVertexAttribArray(index);
  gl_error_check("glDisableVertexAttribArray");
}

//------------------------------------------------------------------------------
void gl::enable_vertex_attrib_array(GLuint vaobj, GLuint index) {
  if (verbose)
    *out << "glEnableVertexArrayAttrib(" << vaobj << ", " << index << ")\n";
  glEnableVertexArrayAttrib(vaobj, index);
  gl_error_check("glEnableVertexArrayAttrib");
}

//------------------------------------------------------------------------------
// gl::void disable_vertex_attrib_array(GLuint vaobj, GLuint index) {
//  if(verbose)*out<<"glDisableVertexArrayAttrib\n";
//  glDisableVertexArrayAttrib(vaobj, index);
//  gl_error_check("glDisableVertexArrayAttrib");
//}

//------------------------------------------------------------------------------
void gl::vertex_attrib_pointer(GLuint index, GLint size, GLenum type,
                               GLboolean normalized, GLsizei stride,
                               const GLvoid* pointer) {
  if (verbose) {
    *out << "glVertexAttribPointer(" << index << ", " << size << ", "
         << to_string(type) << ", " << (normalized ? "true" : "false") << ", "
         << stride << ", " << pointer << ")\n";
  }
  glVertexAttribPointer(index, size, type, normalized, stride, pointer);
  gl_error_check("glVertexAttribPointer");
}

//------------------------------------------------------------------------------
void gl::vertex_attrib_i_pointer(GLuint index, GLint size, GLenum type,
                                 GLsizei stride, const GLvoid* pointer) {
  if (verbose) *out << "glVertexAttribIPointer\n";
  glVertexAttribIPointer(index, size, type, stride, pointer);
  gl_error_check("glVertexAttribIPointer");
}

//------------------------------------------------------------------------------
void gl::vertex_attrib_l_pointer(GLuint index, GLint size, GLenum type,
                                 GLsizei stride, const GLvoid* pointer) {
  if (verbose) *out << "glVertexAttribLPointer\n";
  glVertexAttribLPointer(index, size, type, stride, pointer);
  gl_error_check("glVertexAttribLPointer");
}

//------------------------------------------------------------------------------
void gl::draw_arrays(GLenum mode, GLint first, GLsizei count) {
  if (verbose)
    *out << "glDrawArrays(" << to_string(mode) << ", " << first << ", " << count
         << ")\n";
  glDrawArrays(mode, first, count);
  gl_error_check("glDrawArrays");
}

//==============================================================================
// VERTEXARRAY RELATED
//==============================================================================
void gl::create_vertex_arrays(GLsizei n, GLuint* arr) {
  glCreateVertexArrays(n, arr);
  if (verbose) {
    *out << "glCreateVertexArrays = [" << arr[0];
    for (GLsizei i = 1; i < n; ++i) {
      *out << ", " << arr[i];
    }
    *out << "]\n";
  }
  gl_error_check("glCreateVertexArrays");
}

//------------------------------------------------------------------------------
void gl::delete_vertex_arrays(GLsizei n, GLuint* ids) {
  if (verbose) {
    *out << "glDeleteVertexArrays[" << ids[0];
    for (GLsizei i = 1; i < n; ++i) {
      *out << ", " << ids[i];
    }
    *out << "]\n";
  }
  glDeleteVertexArrays(n, ids);
  gl_error_check("glDeleteVertexArrays");
}

//------------------------------------------------------------------------------
void gl::bind_vertex_array(GLuint array) {
  if (verbose) *out << "glBindVertexArray(" << array << ")\n";
  glBindVertexArray(array);
  gl_error_check("glBindVertexArray");
}

//------------------------------------------------------------------------------
void gl::draw_elements(GLenum mode, GLsizei count, GLenum type,
                       const GLvoid* indices) {
  if (verbose)
    *out << "glDrawElements(" << to_string(mode) << ", " << count << ", "
         << to_string(type) << ", " << indices << ")\n";
  glDrawElements(mode, count, type, indices);
  gl_error_check("glDrawElements");
}

//==============================================================================
// BUFFER RELATED
//==============================================================================
void gl::buffer_data(GLenum target, GLsizeiptr size, const void* data,
                     GLenum usage) {
  if (verbose) {
    *out << "glBufferData(" << to_string(target) << ", " << size << ", " << data
         << ", " << to_string(usage) << ")\n";
  }
  glBufferData(target, size, data, usage);
  gl_error_check("glBufferData");
}
//----------------------------------------------------------------------------
void gl::named_buffer_data(GLuint buffer, GLsizeiptr size, const void* data,
                           GLenum usage) {
  if (verbose) {
    *out << "glNamedBufferData(" << buffer << ", " << size << ", " << data
         << ", " << to_string(usage) << ")\n";
  }
  glNamedBufferData(buffer, size, data, usage);
  gl_error_check("glNamedBufferData");
}
  //----------------------------------------------------------------------------
void gl::bind_buffer(GLenum target, GLuint buffer) {
  if (verbose) {
    *out << "glBindBuffer(" << to_string(target) << ", " << buffer << ")\n";
  }
  glBindBuffer(target, buffer);
  gl_error_check("glBindBuffer");
}

//------------------------------------------------------------------------------
void gl::bind_buffer_base(GLenum target, GLuint index, GLuint buffer) {
  if (verbose)
    *out << "glBindBufferBase(" << to_string(target) << ", " << index << ", "
         << buffer << ")\n";
  glBindBufferBase(target, index, buffer);
  gl_error_check("glBindBufferBase");
}

//------------------------------------------------------------------------------
void gl::create_buffers(GLsizei n, GLuint* ids) {
  glCreateBuffers(n, ids);
  if (verbose) {
    *out << "glCreateBuffers(" << n << ", " << ids << ") = [ ";
    for (GLsizei i = 0; i < n; ++i) *out << ids[i] << ' ';
    *out << "]\n";
  }
  gl_error_check("glCreateBuffers");
}

//------------------------------------------------------------------------------
void gl::delete_buffers(GLsizei n, GLuint* ids) {
  if (verbose) {
    *out << "glDeleteBuffers[ ";
    for (GLsizei i = 0; i < n; ++i) *out << ids[i] << ' ';
    *out << "]\n";
  }
  glDeleteBuffers(n, ids);
  gl_error_check("glDeleteBuffers");
}

//------------------------------------------------------------------------------
void gl::copy_named_buffer_sub_data(GLuint readBuffer, GLuint writeBuffer,
                                    GLintptr readOffset, GLintptr writeOffset,
                                    GLsizei size) {
  if (verbose) *out << "glCopyNamedBufferSubData\n";
  glCopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset,
                           size);
  gl_error_check("glCopyNamedBufferSubData");
}

//------------------------------------------------------------------------------
void* gl::map_named_buffer(GLuint buffer, GLenum access) {
  if (verbose)
    *out << "glMapNamedBuffer(" << buffer << ", " << to_string(access) << ")\n";
  auto result = glMapNamedBuffer(buffer, access);
  gl_error_check("glMapNamedBuffer");
  return result;
}

//------------------------------------------------------------------------------
void* gl::map_named_buffer_range(GLuint buffer, GLintptr offset, GLsizei length,
                                 GLbitfield access) {
  if (verbose)
    *out << "glMapNamedBufferRange(" << buffer << ", " << offset << ", "
         << length << ", " << map_access_to_string(access) << ")\n";
  auto result = glMapNamedBufferRange(buffer, offset, length, access);
  gl_error_check("glMapNamedBufferRange");
  return result;
}


//------------------------------------------------------------------------------
GLboolean gl::unmap_named_buffer(GLuint buffer) {
  if (verbose) *out << "glUnmapNamedBuffer(" << buffer << ")\n";
  auto result = glUnmapNamedBuffer(buffer);
  gl_error_check("glUnmapNamedBuffer");
  return result;
}

//------------------------------------------------------------------------------
void gl::named_buffer_sub_data(GLuint buffer, GLintptr offset, GLsizei size,
                               const void* data) {
  if (verbose)
    *out << "glNamedBufferSubData(" << buffer << ", " << offset << ", " << size
         << ", " << data << ")\n";
  glNamedBufferSubData(buffer, offset, size, data);
  gl_error_check("glNamedBufferSubData");
}
//------------------------------------------------------------------------------
void gl::get_buffer_parameter_iv(GLenum target, GLenum value, GLint* data) {
  assert(target == GL_ARRAY_BUFFER || target == GL_ELEMENT_ARRAY_BUFFER);
  assert(value == GL_BUFFER_SIZE || value == GL_BUFFER_USAGE);
  if (verbose)
    *out << "glGetBufferParameteriv(" << to_string(target) << ", "
         << to_string(value) << ", " << data << ", " << data << ")\n";
  glGetBufferParameteriv(target, value, data);
  gl_error_check("glGetBufferParameteriv");
}
//------------------------------------------------------------------------------
void gl::clear_named_buffer_data(GLuint buffer, GLenum internalformat,
                                 GLenum format, GLenum type, const void* data) {
  if (verbose) {
    *out << "glClearNamedBufferData(" << buffer << ", "
         << to_string(internalformat) << ", " << to_string(format) << ", "
         << to_string(type) << ", " << data << ")\n";
  }
  glClearNamedBufferData(buffer, internalformat, format, type, data);
  gl_error_check("glClearNamedBufferData");
}
//==============================================================================
// SHADER RELATED
//==============================================================================
GLuint gl::create_program() {
  auto            result = glCreateProgram();
  if (verbose) *out << "glCreateProgram() = " << result << "\n";
  gl_error_check("glCreateProgram");
  return result;
}

//------------------------------------------------------------------------------
void gl::attach_shader(GLuint program, GLuint shader) {
  if (verbose) *out << "glAttachShader\n";
  glAttachShader(program, shader);
  gl_error_check("glAttachShader");
}

//------------------------------------------------------------------------------
void gl::link_program(GLuint program) {
  if (verbose) *out << "glLinkProgram(" << program << ")\n";
  glLinkProgram(program);
  gl_error_check("glLinkProgram");
}

//------------------------------------------------------------------------------
void gl::delete_program(GLuint program) {
  if (verbose) *out << "glDeleteProgram(" << program << ")\n";
  glDeleteProgram(program);
  gl_error_check("glDeleteProgram");
}

//------------------------------------------------------------------------------
void gl::use_program(GLuint program) {
  if (verbose) *out << "glUseProgram(" << program << ")\n";
  glUseProgram(program);
  gl_error_check("glUseProgram");
}

//------------------------------------------------------------------------------
GLuint gl::create_shader(GLenum shaderType) {
  auto            result = glCreateShader(shaderType);
  if (verbose) *out << "glCreateShader() = " << result << "\n";
  gl_error_check("glCreateShader");
  return result;
}

//------------------------------------------------------------------------------
void gl::shader_source(GLuint shader, GLsizei count, const GLchar** string,
                       const GLint* length) {
  if (verbose) *out << "glShaderSource\n";
  glShaderSource(shader, count, string, length);
  gl_error_check("glShaderSource");
}

//------------------------------------------------------------------------------
void gl::compile_shader(GLuint shader) {
  if (verbose) *out << "glCompileShader\n";
  glCompileShader(shader);
  gl_error_check("glCompileShader");
}

//------------------------------------------------------------------------------
void gl::delete_shader(GLuint shader) {
  if (verbose) *out << "glDeleteShader(" << shader << ")\n";
  glDeleteShader(shader);
  gl_error_check("glDeleteShader");
}

//------------------------------------------------------------------------------
void gl::dispatch_compute(GLuint num_groups_x, GLuint num_groups_y,
                          GLuint num_groups_z) {
  if (verbose)
    *out << "glDispatchCompute(" << num_groups_x << ", " << num_groups_y << ", "
         << num_groups_z << ")\n";
  glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
  gl_error_check("glDispatchCompute");
}

//------------------------------------------------------------------------------
void gl::get_shader_iv(GLuint shader, GLenum pname, GLint* params) {
  if (verbose) *out << "glGetShaderiv\n";
  glGetShaderiv(shader, pname, params);
  gl_error_check("glGetShaderiv");
}

//------------------------------------------------------------------------------
GLint gl::get_shader_info_log_length(GLuint shader) {
  GLint info_log_length;
  get_shader_iv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
  return info_log_length;
}

//------------------------------------------------------------------------------
void gl::get_shader_info_log(GLuint shader, GLsizei maxLength, GLsizei* length,
                             GLchar* infoLog) {
  if (verbose) *out << "glGetShaderInfoLog\n";
  glGetShaderInfoLog(shader, maxLength, length, infoLog);
  gl_error_check("glGetShaderInfoLog");
}

//------------------------------------------------------------------------------
std::string gl::get_shader_info_log(GLuint shader, GLsizei maxLength) {
  std::vector<char> buffer(maxLength);
  if (verbose) *out << "data\n";
  get_shader_info_log(shader, maxLength, nullptr, buffer.data());
  std::string log(buffer.begin(), buffer.end());
  return log;
}

//------------------------------------------------------------------------------
std::string gl::get_shader_info_log(GLuint shader) {
  return get_shader_info_log(shader, get_shader_info_log_length(shader));
}

//------------------------------------------------------------------------------
void gl::program_uniform_1f(GLuint program, GLint location, GLfloat v0) {
  if (verbose)
    *out << "glProgramUniform1f(" << program << ", " << location << ", " << v0
         << ")\n";
  glProgramUniform1f(program, location, v0);
  gl_error_check("glProgramUniform1f");
}

//------------------------------------------------------------------------------
void gl::program_uniform_2f(GLuint program, GLint location, GLfloat v0,
                            GLfloat v1) {
  if (verbose)
    *out << "glProgramUniform2f(" << program << ", " << location << ", " << v0
         << ", " << v1 << ")\n";
  glProgramUniform2f(program, location, v0, v1);
  gl_error_check("glProgramUniform2f");
}

//------------------------------------------------------------------------------
void gl::program_uniform_3f(GLuint program, GLint location, GLfloat v0,
                            GLfloat v1, GLfloat v2) {
  if (verbose)
    *out << "glProgramUniform3f(" << program << ", " << location << ", " << v0
         << ", " << v1 << ", " << v2 << ")\n";
  glProgramUniform3f(program, location, v0, v1, v2);
  gl_error_check("glProgramUniform3f");
}

//------------------------------------------------------------------------------
void gl::program_uniform_4f(GLuint program, GLint location, GLfloat v0,
                            GLfloat v1, GLfloat v2, GLfloat v3) {
  if (verbose)
    *out << "glProgramUniform4f(" << program << ", " << location << ", " << v0
         << ", " << v1 << ", " << v2 << ", " << v3 << ")\n";
  glProgramUniform4f(program, location, v0, v1, v2, v3);
  gl_error_check("glProgramUniform4f");
}

//------------------------------------------------------------------------------
void gl::program_uniform_1i(GLuint program, GLint location, GLint v0) {
  if (verbose)
    *out << "glProgramUniform1i(" << program << ", " << location << ", " << v0
         << ")\n";
  glProgramUniform1i(program, location, v0);
  gl_error_check("glProgramUniform1i");
}

//------------------------------------------------------------------------------
void gl::program_uniform_2i(GLuint program, GLint location, GLint v0,
                            GLint v1) {
  if (verbose)
    *out << "glProgramUniform2i(" << program << ", " << location << ", " << v0
         << ", " << v1 << ")\n";
  glProgramUniform2i(program, location, v0, v1);
  gl_error_check("glProgramUniform2i(" + std::to_string(program) + ", " +
                 std::to_string(location) + ", " + std::to_string(v0) + ", " +
                 std::to_string(v1) + ")");
}

//------------------------------------------------------------------------------
void gl::program_uniform_3i(GLuint program, GLint location, GLint v0, GLint v1,
                            GLint v2) {
  if (verbose)
    *out << "glProgramUniform3i(" << program << ", " << location << ", " << v0
         << ", " << v1 << ", " << v2 << ")\n";
  glProgramUniform3i(program, location, v0, v1, v2);
  gl_error_check("glProgramUniform3i");
}

//------------------------------------------------------------------------------
void gl::program_uniform_4i(GLuint program, GLint location, GLint v0, GLint v1,
                            GLint v2, GLint v3) {
  if (verbose)
    *out << "glProgramUniform4i(" << program << ", " << location << ", " << v0
         << ", " << v1 << ", " << v2 << ", " << v3 << ")\n";
  glProgramUniform4i(program, location, v0, v1, v2, v3);
  gl_error_check("glProgramUniform4i");
}

//------------------------------------------------------------------------------
void gl::program_uniform_1ui(GLuint program, GLint location, GLuint v0) {
  if (verbose) {
    *out << "glProgramUniform1ui(" << program << ", " << location << ", " << v0
         << ")\n";
  }
  glProgramUniform1ui(program, location, v0);
  gl_error_check("glProgramUniform1ui");
}

//------------------------------------------------------------------------------
void gl::program_uniform_2ui(GLuint program, GLint location, GLuint v0,
                             GLuint v1) {
  if (verbose)
    *out << "glProgramUniform2ui(" << program << ", " << location << ", " << v0
         << ", " << v1 << ")\n";
  glProgramUniform2ui(program, location, v0, v1);
  gl_error_check("glProgramUniform2ui(" + std::to_string(program) + ", " +
                 std::to_string(location) + ", " + std::to_string(v0) + ", " +
                 std::to_string(v1) + ")");
}

//------------------------------------------------------------------------------
void gl::program_uniform_3ui(GLuint program, GLint location, GLuint v0,
                             GLuint v1, GLuint v2) {
  if (verbose) *out << "glProgramUniform3ui\n";
  glProgramUniform3ui(program, location, v0, v1, v2);
  gl_error_check("glProgramUniform3ui");
}

//------------------------------------------------------------------------------
void gl::program_uniform_4ui(GLuint program, GLint location, GLuint v0,
                             GLuint v1, GLuint v2, GLuint v3) {
  if (verbose) *out << "glProgramUniform4ui\n";
  glProgramUniform4ui(program, location, v0, v1, v2, v3);
  gl_error_check("glProgramUniform4ui");
}

//------------------------------------------------------------------------------
void gl::program_uniform_1fv(GLuint program, GLint location, GLsizei count,
                             const GLfloat* value) {
  if (verbose) *out << "glProgramUniform1fv\n";
  glProgramUniform1fv(program, location, count, value);
  gl_error_check("glProgramUniform1fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_2fv(GLuint program, GLint location, GLsizei count,
                             const GLfloat* value) {
  if (verbose) *out << "glProgramUniform2fv\n";
  glProgramUniform2fv(program, location, count, value);
  gl_error_check("glProgramUniform2fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_3fv(GLuint program, GLint location, GLsizei count,
                             const GLfloat* value) {
  if (verbose) *out << "glProgramUniform3fv\n";
  glProgramUniform3fv(program, location, count, value);
  gl_error_check("glProgramUniform3fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_4fv(GLuint program, GLint location, GLsizei count,
                             const GLfloat* value) {
  if (verbose) *out << "glProgramUniform4fv\n";
  glProgramUniform4fv(program, location, count, value);
  gl_error_check("glProgramUniform4fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_1iv(GLuint program, GLint location, GLsizei count,
                             const GLint* value) {
  if (verbose) *out << "glProgramUniform1iv\n";
  glProgramUniform1iv(program, location, count, value);
  gl_error_check("glProgramUniform1iv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_2iv(GLuint program, GLint location, GLsizei count,
                             const GLint* value) {
  if (verbose) *out << "glProgramUniform2iv\n";
  glProgramUniform2iv(program, location, count, value);
  gl_error_check("glProgramUniform2iv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_3iv(GLuint program, GLint location, GLsizei count,
                             const GLint* value) {
  if (verbose) *out << "glProgramUniform3iv\n";
  glProgramUniform3iv(program, location, count, value);
  gl_error_check("glProgramUniform3iv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_4iv(GLuint program, GLint location, GLsizei count,
                             const GLint* value) {
  if (verbose) *out << "glProgramUniform4iv\n";
  glProgramUniform4iv(program, location, count, value);
  gl_error_check("glProgramUniform4iv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_1uiv(GLuint program, GLint location, GLsizei count,
                              const GLuint* value) {
  if (verbose) *out << "glProgramUniform1uiv\n";
  glProgramUniform1uiv(program, location, count, value);
  gl_error_check("glProgramUniform1uiv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_2uiv(GLuint program, GLint location, GLsizei count,
                              const GLuint* value) {
  if (verbose) *out << "glProgramUniform2uiv\n";
  glProgramUniform2uiv(program, location, count, value);
  gl_error_check("glProgramUniform2uiv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_3uiv(GLuint program, GLint location, GLsizei count,
                              const GLuint* value) {
  if (verbose) *out << "glProgramUniform3uiv\n";
  glProgramUniform3uiv(program, location, count, value);
  gl_error_check("glProgramUniform3uiv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_4uiv(GLuint program, GLint location, GLsizei count,
                              const GLuint* value) {
  if (verbose) *out << "glProgramUniform4uiv\n";
  glProgramUniform4uiv(program, location, count, value);
  gl_error_check("glProgramUniform4uiv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_matrix_2fv(GLuint program, GLint location,
                                    GLsizei count, GLboolean transpose,
                                    const GLfloat* value) {
  if (verbose) *out << "glProgramUniformMatrix2fv\n";
  glProgramUniformMatrix2fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix2fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_matrix_3fv(GLuint program, GLint location,
                                    GLsizei count, GLboolean transpose,
                                    const GLfloat* value) {
  if (verbose) *out << "glProgramUniformMatrix3fv\n";
  glProgramUniformMatrix3fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix3fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_matrix_4fv(GLuint program, GLint location,
                                    GLsizei count, GLboolean transpose,
                                    const GLfloat* value) {
  if (verbose) *out << "glProgramUniformMatrix4fv\n";
  glProgramUniformMatrix4fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix4fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_matrix_2x3fv(GLuint program, GLint location,
                                      GLsizei count, GLboolean transpose,
                                      const GLfloat* value) {
  if (verbose) *out << "glProgramUniformMatrix2x3fv\n";
  glProgramUniformMatrix2x3fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix2x3fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_matrix_3x2fv(GLuint program, GLint location,
                                      GLsizei count, GLboolean transpose,
                                      const GLfloat* value) {
  if (verbose) *out << "glProgramUniformMatrix3x2fv\n";
  glProgramUniformMatrix3x2fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix3x2fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_matrix_2x4fv(GLuint program, GLint location,
                                      GLsizei count, GLboolean transpose,
                                      const GLfloat* value) {
  if (verbose) *out << "glProgramUniformMatrix2x4fv\n";
  glProgramUniformMatrix2x4fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix2x4fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_matrix_4x2fv(GLuint program, GLint location,
                                      GLsizei count, GLboolean transpose,
                                      const GLfloat* value) {
  if (verbose) *out << "glProgramUniformMatrix4x2fv\n";
  glProgramUniformMatrix4x2fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix4x2fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_matrix_3x4fv(GLuint program, GLint location,
                                      GLsizei count, GLboolean transpose,
                                      const GLfloat* value) {
  if (verbose) *out << "glProgramUniformMatrix3x4fv\n";
  glProgramUniformMatrix3x4fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix3x4fv");
}

//------------------------------------------------------------------------------
void gl::program_uniform_matrix_4x3fv(GLuint program, GLint location,
                                      GLsizei count, GLboolean transpose,
                                      const GLfloat* value) {
  if (verbose) *out << "glProgramUniformMatrix4x3fv\n";
  glProgramUniformMatrix4x3fv(program, location, count, transpose, value);
  gl_error_check("glProgramUniformMatrix4x3fv");
}

//------------------------------------------------------------------------------
void gl::get_program_iv(GLuint program, GLenum pname, GLint* params) {
  if (verbose) *out << "glGetProgramiv\n";
  glGetProgramiv(program, pname, params);
  gl_error_check("glGetProgramiv");
}

//------------------------------------------------------------------------------
void gl::get_program_info_log(GLuint program, GLsizei maxLength,
                              GLsizei* length, GLchar* infoLog) {
  if (verbose) *out << "glGetProgramInfoLog\n";
  glGetProgramInfoLog(program, maxLength, length, infoLog);
  gl_error_check("glGetProgramInfoLog");
}

//------------------------------------------------------------------------------
GLint gl::get_uniform_location(GLuint program, const GLchar* name) {
  auto            result = glGetUniformLocation(program, name);
  if (verbose)
    *out << "glGetUniformLocation(" << program << ", " << name
         << ") = " << result << '\n';
  gl_error_check("glGetUniformLocation");
  return result;
}

//==============================================================================
// TEXTURE RELATED
//==============================================================================
void gl::create_textures(GLenum target, GLsizei n, GLuint* textures) {
  glCreateTextures(target, n, textures);
  if (verbose) {
    *out << "glCreateTextures(" << to_string(target) << ", " << n << ", "
         << textures << ") = [ ";
    for (GLsizei i = 0; i < n; ++i) *out << textures[i] << ' ';
    *out << "]\n";
  }
  gl_error_check("glCreateTextures");
}

//------------------------------------------------------------------------------
void gl::delete_textures(GLsizei n, GLuint* textures) {
  if (verbose) {
    *out << "glDeleteTextures[ ";
    for (GLsizei i = 0; i < n; ++i) *out << textures[i] << ' ';
    *out << "]\n";
  }
  glDeleteTextures(n, textures);
  gl_error_check("glDeleteTextures");
}

//------------------------------------------------------------------------------
void gl::active_texture(GLenum texture) {
  if (verbose) *out << "glActiveTexture(" << texture - GL_TEXTURE0 << ")\n";
  glActiveTexture(texture);
  gl_error_check("glActiveTexture");
}

//------------------------------------------------------------------------------
void gl::bind_texture(GLenum target, GLuint texture) {
  if (verbose)
    *out << "glBindTexture(" << to_string(target) << ", " << texture << ")\n";
  glBindTexture(target, texture);
  gl_error_check("glBindTexture");
}

//------------------------------------------------------------------------------
void gl::bind_image_texture(GLuint unit, GLuint texture, GLint level,
                            GLboolean layered, GLint layer, GLenum access,
                            GLenum format) {
  if (verbose)
    *out << "glBindImageTexture(" << unit << ", " << texture << ", " << level
         << ", " << (layered ? "true" : "false") << ", " << layer << ", "
         << to_string(access) << ", " << to_string(format) << ")\n";
  glBindImageTexture(unit, texture, level, layered, layer, access, format);
  gl_error_check("glBindImageTexture");
}

//------------------------------------------------------------------------------
void gl::tex_image_1d(GLenum target, GLint level, GLint internal_format,
                      GLsizei width, GLint border, GLenum format, GLenum type,
                      const GLvoid* data) {
  assert(width >= 0);
  //assert(width < GL_MAX_TEXTURE_SIZE);
  if (verbose)
    *out << "glTexImage1D(" << to_string(target) << ", " << level << ", "
         << to_string(internal_format) << ", " << width << ", " << border
         << ", " << to_string(format) << ", " << to_string(type) << ", " << data
         << ")\n";
  glTexImage1D(target, level, internal_format, width, border, format, type,
               data);
  gl_error_check("glTexImage1D");
}

//------------------------------------------------------------------------------
void gl::tex_image_2d(GLenum target, GLint level, GLint internal_format,
                      GLsizei width, GLsizei height, GLint border,
                      GLenum format, GLenum type, const GLvoid* data) {
  assert(width >= 0);
  assert(height >= 0);
  //assert(width < GL_MAX_TEXTURE_SIZE);
  //assert(height < GL_MAX_TEXTURE_SIZE);

  if (verbose)
    *out << "glTexImage2D(" << to_string(target) << ", " << level << ", "
         << to_string(internal_format) << ", " << width << ", " << height
         << ", " << border << ", " << to_string(format) << ", "
         << to_string(type) << ", " << data << ")\n";
  glTexImage2D(target, level, internal_format, width, height, border, format,
               type, data);
  gl_error_check("glTexImage2D");
}
//------------------------------------------------------------------------------
void gl::tex_sub_image_2d(GLenum target, GLint level, GLint xoffset,
                          GLint yoffset, GLsizei width, GLsizei height,
                          GLenum format, GLenum type, const GLvoid* pixels) {
  if (verbose)
    *out << "glTexSubImage2D(" << to_string(target) << ", " << level << ", "
         << xoffset << ", " << yoffset << ", " << width << ", " << height
         << ", " << to_string(format) << ", " << to_string(type) << ", "
         << pixels << ")\n";
  glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type,
                  pixels);
  gl_error_check("glTexSubImage2D");
}
//------------------------------------------------------------------------------
void gl::texture_sub_image_2d(GLuint texture, GLint level, GLint xoffset,
                              GLint yoffset, GLsizei width, GLsizei height,
                              GLenum format, GLenum type,
                              const GLvoid* pixels) {
  if (verbose) {
    *out << "glTextureSubImage2D(" << texture << ", " << level << ", "
         << xoffset << ", " << yoffset << ", " << width << ", " << height
         << ", " << to_string(format) << ", " << to_string(type) << ", "
         << pixels << ")\n";
    glTextureSubImage2D(texture, level, xoffset, yoffset, width, height, format,
                        type, pixels);
  }
  gl_error_check("glTextureSubImage2D");
}

//------------------------------------------------------------------------------
void gl::tex_image_3d(GLenum target, GLint level, GLint internal_format,
                      GLsizei width, GLsizei height, GLsizei depth,
                      GLint border, GLenum format, GLenum type,
                      const GLvoid* data) {
  assert(width >= 0);
  assert(height >= 0);
  assert(depth >= 0);
  //assert(width < GL_MAX_TEXTURE_SIZE);
  //assert(height < GL_MAX_TEXTURE_SIZE);
  //assert(depth < GL_MAX_TEXTURE_SIZE);
  if (verbose)
    *out << "glTexImage3D(" << to_string(target) << ", " << level << ", "
         << to_string(internal_format) << ", " << width << ", " << height
         << ", " << depth << ", " << border << ", " << to_string(format) << ", "
         << to_string(type) << ", " << data << ")\n";
  glTexImage3D(target, level, internal_format, width, height, depth, border,
               format, type, data);
  gl_error_check("glTexImage3D");
}

//------------------------------------------------------------------------------
void gl::copy_image_sub_data(GLuint srcName, GLenum srcTarget, GLint srcLevel,
                             GLint srcX, GLint srcY, GLint srcZ, GLuint dstName,
                             GLenum dstTarget, GLint dstLevel, GLint dstX,
                             GLint dstY, GLint dstZ, GLsizei srcWidth,
                             GLsizei srcHeight, GLsizei srcDepth) {
  if (verbose)
    *out << "glCopyImageSubData(" << srcName << ", " << to_string(srcTarget)
         << ", " << srcLevel << ", " << srcX << ", " << srcY << ", " << srcZ
         << ", " << dstName << ", " << to_string(dstTarget) << ", " << dstLevel
         << ", " << dstX << ", " << dstY << ", " << dstZ << ", " << srcWidth
         << ", " << srcHeight << ", " << srcDepth << ")\n";
  glCopyImageSubData(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName,
                     dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight,
                     srcDepth);
  gl_error_check("glCopyImageSubData");
}

//------------------------------------------------------------------------------
void gl::get_tex_image(GLenum target, GLint level, GLenum format, GLenum type,
                       GLvoid* pixels) {
  if (verbose) *out << "glGetTexImage\n";
  glGetTexImage(target, level, format, type, pixels);
  gl_error_check("glGetTexImage");
}

//------------------------------------------------------------------------------
void gl::get_n_tex_image(GLenum target, GLint level, GLenum format, GLenum type,
                         GLsizei bufSize, void* pixels) {
  if (verbose) *out << "glGetnTexImage\n";
  glGetnTexImage(target, level, format, type, bufSize, pixels);
  gl_error_check("glGetnTexImage");
}

//------------------------------------------------------------------------------
void gl::get_texture_image(GLuint texture, GLint level, GLenum format,
                           GLenum type, GLsizei bufSize, void* pixels) {
  if (verbose)
    *out << "glGetTextureImage(" << texture << ", " << level << ", " << format
         << ", " << type << ", " << bufSize << ", " << pixels << ")\n";
  glGetTextureImage(texture, level, format, type, bufSize, pixels);
  gl_error_check("glGetTextureImage");
}
//------------------------------------------------------------------------------
void gl::get_texture_sub_image(GLuint texture, GLint level, GLint xoffset,
                               GLint yoffset, GLint zoffset, GLsizei width,
                               GLsizei height, GLsizei depth, GLenum format,
                               GLenum type, GLsizei bufSize, void* pixels) {
  if (verbose) {
    *out << "glGetTextureSubImage(" << texture << ", " << level << ", "
         << xoffset << ", " << yoffset << ", " << zoffset << ", " << width
         << ", " << height << ", " << depth << ", " << format << ", " << type
         << ", " << bufSize << ", " << pixels << ")\n";
  }
  glGetTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height,
                       depth, format, type, bufSize, pixels);
  gl_error_check("glGetTextureSubImage");
}

//------------------------------------------------------------------------------
void gl::tex_parameter_f(GLenum target, GLenum pname, GLfloat param) {
  if (verbose) *out << "glTexParameterf\n";
  glTexParameterf(target, pname, param);
  gl_error_check("glTexParameterf");
}

//------------------------------------------------------------------------------
void gl::tex_parameter_i(GLenum target, GLenum pname, GLint param) {
  if (verbose) *out << "glTexParameteri\n";
  glTexParameteri(target, pname, param);
  gl_error_check("glTexParameteri");
}

//------------------------------------------------------------------------------
void gl::texture_parameter_f(GLuint texture, GLenum pname, GLfloat param) {
  if (verbose)
    *out << "glTextureParameterf(" << texture << ", " << to_string(pname)
         << ", " << param << ")\n";
  glTextureParameterf(texture, pname, param);
  gl_error_check("glTexParameterf");
}

//------------------------------------------------------------------------------
void gl::texture_parameter_i(GLuint texture, GLenum pname, GLint param) {
  if (verbose)
    *out << "glTextureParameteri(" << texture << ", " << to_string(pname)
         << ", " << texparami_to_string(param) << ")\n";
  glTextureParameteri(texture, pname, param);
  gl_error_check("glTexParameteri");
}

//------------------------------------------------------------------------------
void gl::tex_parameter_fv(GLenum target, GLenum pname, const GLfloat* params) {
  if (verbose)
    *out << "glTextureParameterfv(" << to_string(target) << ", "
         << to_string(pname) << ", " << params << ")\n";
  glTexParameterfv(target, pname, params);
  gl_error_check("glTexParameterfv");
}

//------------------------------------------------------------------------------
void gl::tex_parameter_iv(GLenum target, GLenum pname, const GLint* params) {
  if (verbose)
    *out << "glTextureParameteriv(" << to_string(target) << ", "
         << to_string(pname) << ", " << params << ")\n";
  glTexParameteriv(target, pname, params);
  gl_error_check("glTexParameteriv");
}

//------------------------------------------------------------------------------
void gl::tex_parameter_Iiv(GLenum target, GLenum pname, const GLint* params) {
  if (verbose)
    *out << "glTextureParameterIiv(" << to_string(target) << ", "
         << to_string(pname) << ", " << params << ")\n";
  glTexParameterIiv(target, pname, params);
  gl_error_check("glTexParameterIiv");
}

//------------------------------------------------------------------------------
void gl::tex_parameter_Iuiv(GLenum target, GLenum pname, const GLuint* params) {
  if (verbose)
    *out << "glTextureParameterIuiv(" << to_string(target) << ", "
         << to_string(pname) << ", " << params << ")\n";
  glTexParameterIuiv(target, pname, params);
  gl_error_check("glTexParameterIuiv");
}

//------------------------------------------------------------------------------
void gl::texture_parameter_fv(GLuint texture, GLenum pname,
                              const GLfloat* paramtexture) {
  if (verbose)
    *out << "glTextureParameterfv(" << texture << ", " << to_string(pname)
         << ", " << paramtexture << ")\n";
  glTextureParameterfv(texture, pname, paramtexture);
  gl_error_check("glTextureParameterfv");
}

//------------------------------------------------------------------------------
void gl::texture_parameter_iv(GLuint texture, GLenum pname,
                              const GLint* param) {
  if (verbose)
    *out << "glTextureParameterfv(" << texture << ", " << to_string(pname)
         << ", " << param << ")\n";
  glTextureParameteriv(texture, pname, param);
  gl_error_check("glTextureParameteriv");
}

//------------------------------------------------------------------------------
void gl::texture_parameter_Iiv(GLuint texture, GLenum pname,
                               const GLint* param) {
  if (verbose)
    *out << "glTextureParameterIiv(" << texture << ", " << to_string(pname)
         << ", " << param << ")\n";
  glTextureParameterIiv(texture, pname, param);
  gl_error_check("glTextureParameterIiv");
}

//------------------------------------------------------------------------------
void gl::texture_parameter_Iuiv(GLuint texture, GLenum pname,
                                const GLuint* param) {
  if (verbose)
    *out << "glTextureParameterIuiv(" << texture << ", " << to_string(pname)
         << ", " << param << ")\n";
  glTextureParameterIuiv(texture, pname, param);
  gl_error_check("glTextureParameterIuiv");
}

//------------------------------------------------------------------------------
void gl::clear_tex_image(GLuint texture, GLint level, GLenum format,
                         GLenum type, const void* data) {
  if (verbose)
    *out << "glClearTexImage(" << texture << ", " << level << ", "
         << to_string(format) << ", " << to_string(type) << ", " << data
         << ")\n";
  glClearTexImage(texture, level, format, type, data);
  gl_error_check("glClearTexImage");
}
//------------------------------------------------------------------------------
auto gl::is_texture(GLuint texture) -> GLboolean {
  if (verbose) *out << "glIsTexture(" << texture << ")\n";
  auto const b = glIsTexture(texture);
  gl_error_check("glIsTexture");
  return b;
}
//------------------------------------------------------------------------------
void gl::bind_sampler(GLuint unit, GLuint sampler) {
  if (verbose) *out << "glBindSampler(" << unit << ", " << sampler << ")\n";
  glBindSampler(unit, sampler);
  gl_error_check("glBindSampler");
}
//==============================================================================
// FRAMEBUFFER RELATED
//==============================================================================
void gl::create_framebuffers(GLsizei n, GLuint* ids) {
  glCreateFramebuffers(n, ids);
  if (verbose) {
    *out << "glCreateFramebuffers(" << n << ", " << ids << ") = [ ";
    for (GLsizei i = 0; i < n; ++i) *out << ids[i] << ' ';
    *out << "]\n";
  }
  gl_error_check("glCreateFramebuffers");
}

//------------------------------------------------------------------------------
void gl::delete_framebuffers(GLsizei n, GLuint* ids) {
  if (verbose) {
    *out << "glDeleteFramebuffers[ ";
    for (GLsizei i = 0; i < n; ++i) *out << ids[i] << ' ';
    *out << "]\n";
  }
  glDeleteFramebuffers(n, ids);
  gl_error_check("glDeleteFramebuffers");
}

//------------------------------------------------------------------------------
void gl::bind_framebuffer(GLenum target, GLuint framebuffer) {
  if (verbose)
    *out << "glBindFramebuffer(" << to_string(target) << ", " << framebuffer
         << ")\n";
  glBindFramebuffer(target, framebuffer);
  gl_error_check("glBindFramebuffer");
}

//------------------------------------------------------------------------------
void gl::framebuffer_texture(GLenum target, GLenum attachment, GLuint texture,
                             GLint level) {
  if (verbose) *out << "glFramebufferTexture\n";
  glFramebufferTexture(target, attachment, texture, level);
  gl_error_check("glFramebufferTexture");
}

//------------------------------------------------------------------------------
void gl::framebuffer_texture_1d(GLenum target, GLenum attachment,
                                GLenum textarget, GLuint texture, GLint level) {
  if (verbose) *out << "glFramebufferTexture1D\n";
  glFramebufferTexture1D(target, attachment, textarget, texture, level);
  gl_error_check("glFramebufferTexture1D");
}

//------------------------------------------------------------------------------
void gl::framebuffer_texture_2d(GLenum target, GLenum attachment,
                                GLenum textarget, GLuint texture, GLint level) {
  if (verbose) *out << "glFramebufferTexture2D\n";
  glFramebufferTexture2D(target, attachment, textarget, texture, level);
  gl_error_check("glFramebufferTexture2D");
}

//------------------------------------------------------------------------------
void gl::framebuffer_texture_3d(GLenum target, GLenum attachment,
                                GLenum textarget, GLuint texture, GLint level,
                                GLint layer) {
  if (verbose) *out << "glFramebufferTexture3D\n";
  glFramebufferTexture3D(target, attachment, textarget, texture, level, layer);
  gl_error_check("glFramebufferTexture3D");
}

//------------------------------------------------------------------------------
void gl::named_framebuffer_texture(GLuint framebuffer, GLenum attachment,
                                   GLuint texture, GLint level) {
  if (verbose)
    *out << "glNamedFramebufferTexture(" << framebuffer << ", "
         << to_string(attachment) << ", " << texture << ", " << level << ")\n";
  glNamedFramebufferTexture(framebuffer, attachment, texture, level);
  gl_error_check("glNamedFramebufferTexture");
}

//------------------------------------------------------------------------------
GLenum gl::check_named_framebuffer_status(GLuint framebuffer, GLenum target) {
  if (verbose)
    *out << "glCheckNamedFramebufferStatus(" << framebuffer << ", "
         << to_string(target) << ")\n";
  auto result = glCheckNamedFramebufferStatus(framebuffer, target);
  gl_error_check("glCheckNamedFramebufferStatus");
  return result;
}
//------------------------------------------------------------------------------
void gl::named_framebuffer_draw_buffers(GLuint framebuffer, GLsizei n,
                                               const GLenum* bufs) {
  if (verbose) {
    *out << "glNamedFramebufferDrawBuffers(" << framebuffer << ", " << n
         << ", {";
    for (GLsizei i = 0; i < n - 1; ++i) { *out << to_string(bufs[i]) << ", "; }
    *out << to_string(bufs[n - 1]) << "})\n";
  }
  glNamedFramebufferDrawBuffers(framebuffer, n, bufs);
  gl_error_check("glCheckNamedFramebufferStatus");
}
//------------------------------------------------------------------------------
void gl::memory_barrier(GLbitfield barriers) {
  if (verbose) { *out << "glMemoryBarrier(" << to_string(barriers) << ")\n"; }
  glMemoryBarrier(barriers);
  gl_error_check("glMemoryBarrier");
}
//==============================================================================
}  // namespace yavin
//==============================================================================
