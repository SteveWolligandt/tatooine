#include <tatooine/gl/glfunctions.h>
#include <tatooine/gl/glwrapper.h>
#include <tatooine/gl/shader.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
void shader::create() {
  delete_shader();
  set_id(gl::create_program());

  for (auto& stage : m_shader_stages) {
    stage.compile();
    gl::attach_shader(id(), stage.id());
    for (const auto& var : stage.glsl_vars()) {
      if (var.modifier == GLSLVar::UNIFORM)
        m_uniform_var_names.insert(var.name);
      else if (stage.stage_type() == GL_VERTEX_SHADER &&
               var.modifier == GLSLVar::IN)
        m_attribute_var_names.insert(var.name);
    }
    stage.delete_stage();
  }

  gl::link_program(id());
  if (auto log = info_log(); log) {
    throw std::runtime_error{std::move(*log)};
  }
  bind();
  for (const auto& var : m_attribute_var_names) {
    add_attribute(var);
  }
  for (const auto& var : m_uniform_var_names) {
    add_uniform(var);
  }
}
//------------------------------------------------------------------------------
void shader::delete_shader() {
  if (bound_program() == id()) {
    unbind();
  }
  if (id() != 0) gl::delete_program(id());
  set_id(0);
}
//------------------------------------------------------------------------------
void shader::bind() const { gl::use_program(id()); }
//------------------------------------------------------------------------------
void shader::unbind() const { gl::use_program(0); }
//------------------------------------------------------------------------------
void shader::add_uniform(const std::string& uniform_var_name) {
  m_uniform_locations.insert(std::pair<std::string, GLint>(
      uniform_var_name, gl::get_uniform_location(id(), uniform_var_name.c_str())));
}
//------------------------------------------------------------------------------
void shader::add_attribute(const std::string& attribute_var_name) {
  m_attribute_locations.insert(std::pair<std::string, GLint>(
      attribute_var_name,
      gl::get_uniform_location(id(), attribute_var_name.c_str())));
}
//------------------------------------------------------------------------------
GLint shader::uniform(const std::string& uniform_var_name) {
  auto it = m_uniform_locations.find(uniform_var_name);
  if (it == end(m_uniform_locations)) {
    throw std::runtime_error{"There is no uniform variable with name " +
                             uniform_var_name};
  }
  return it->second;
}
//------------------------------------------------------------------------------
GLint shader::attribute(const std::string& attribute_var_name) {
  return m_attribute_locations[attribute_var_name];
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLboolean value) {
  gl::program_uniform_1i(id(), uniform(name), value);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLfloat value) {
  gl::program_uniform_1f(id(), uniform(name), value);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLfloat x, GLfloat y) {
  gl::program_uniform_2f(id(), uniform(name), x, y);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLfloat x, GLfloat y,
                         GLfloat z) {
  gl::program_uniform_3f(id(), uniform(name), x, y, z);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLfloat x, GLfloat y,
                         GLfloat z, GLfloat w) {
  gl::program_uniform_4f(id(), uniform(name), x, y, z, w);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string&            name,
                         std::array<GLfloat, 2> const& data) {
  gl::program_uniform_2fv(id(), uniform(name), 1, data.data());
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string&            name,
                         std::array<GLfloat, 3> const& data) {
  gl::program_uniform_3fv(id(), uniform(name), 1, data.data());
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string&            name,
                         std::array<GLfloat, 4> const& data) {
  gl::program_uniform_4fv(id(), uniform(name), 1, data.data());
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLint value) {
  gl::program_uniform_1i(id(), uniform(name), value);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLint x, GLint y) {
  gl::program_uniform_2i(id(), uniform(name), x, y);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLint x, GLint y, GLint z) {
  gl::program_uniform_3i(id(), uniform(name), x, y, z);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLint x, GLint y, GLint z,
                         GLint w) {
  gl::program_uniform_4i(id(), uniform(name), x, y, z, w);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string&          name,
                         std::array<GLint, 2> const& data) {
  gl::program_uniform_2iv(id(), uniform(name), 1, data.data());
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string&          name,
                         std::array<GLint, 3> const& data) {
  gl::program_uniform_3iv(id(), uniform(name), 1, data.data());
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string&          name,
                         std::array<GLint, 4> const& data) {
  gl::program_uniform_4iv(id(), uniform(name), 1, data.data());
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLuint value) {
  gl::program_uniform_1ui(id(), uniform(name), value);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLuint x, GLuint y) {
  gl::program_uniform_2ui(id(), uniform(name), x, y);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLuint x, GLuint y,
                         GLuint z) {
  gl::program_uniform_3ui(id(), uniform(name), x, y, z);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, GLuint x, GLuint y, GLuint z,
                         GLuint w) {
  gl::program_uniform_4ui(id(), uniform(name), x, y, z, w);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string&           name,
                         std::array<GLuint, 2> const& data) {
  gl::program_uniform_2uiv(id(), uniform(name), 1, data.data());
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string&           name,
                         std::array<GLuint, 3> const& data) {
  gl::program_uniform_3uiv(id(), uniform(name), 1, data.data());
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string&           name,
                         std::array<GLuint, 4> const& data) {
  gl::program_uniform_4uiv(id(), uniform(name), 1, data.data());
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, Vec2<GLfloat> const& data) {
  gl::program_uniform_2fv(id(), uniform(name), 1, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_vec2(const std::string& name, GLfloat const* data) {
  gl::program_uniform_2fv(id(), uniform(name), 1, data);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform(const std::string& name, Vec2<GLint> const& data) {
  gl::program_uniform_2iv(id(), uniform(name), 1, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_vec2(const std::string& name, GLint const* data) {
  gl::program_uniform_2iv(id(), uniform(name), 1, data);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform(const std::string& name, Vec2<GLuint> const& data) {
  gl::program_uniform_2uiv(id(), uniform(name), 1, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_vec2(const std::string& name, GLuint const* data) {
  gl::program_uniform_2uiv(id(), uniform(name), 1, data);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, Vec3<GLfloat> const& data) {
  gl::program_uniform_3fv(id(), uniform(name), 1, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_vec3(const std::string& name, GLfloat const* data) {
  gl::program_uniform_3fv(id(), uniform(name), 1, data);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform(const std::string& name, Vec3<GLint> const& data) {
  gl::program_uniform_3iv(id(), uniform(name), 1, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_vec3(const std::string& name, GLint const* data) {
  gl::program_uniform_3iv(id(), uniform(name), 1, data);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform(const std::string& name, Vec3<GLuint> const& data) {
  gl::program_uniform_3uiv(id(), uniform(name), 1, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_vec3(const std::string& name, GLuint const* data) {
  gl::program_uniform_3uiv(id(), uniform(name), 1, data);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, Vec4<GLfloat> const& data) {
  gl::program_uniform_4fv(id(), uniform(name), 1, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_vec4(const std::string& name, GLfloat const* data) {
  gl::program_uniform_4fv(id(), uniform(name), 1, data);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform(const std::string& name, Vec4<GLint> const& data) {
  gl::program_uniform_4iv(id(), uniform(name), 1, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_vec4(const std::string& name, GLint const* data) {
  gl::program_uniform_4iv(id(), uniform(name), 1, data);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform(const std::string& name, Vec4<GLuint> const& data) {
  gl::program_uniform_4uiv(id(), uniform(name), 1, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_vec4(const std::string& name, GLuint const* data) {
  gl::program_uniform_4uiv(id(), uniform(name), 1, data);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, Mat2<GLfloat> const& data) {
  gl::program_uniform_matrix_2fv(id(), uniform(name), 1, false,
                                 data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_mat2(const std::string& name, GLfloat const* data) {
  gl::program_uniform_matrix_2fv(id(), uniform(name), 1, false,
                                 data);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, Mat3<GLfloat> const& data) {
  gl::program_uniform_matrix_3fv(id(), uniform(name), 1, false,
                                 data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_mat3(const std::string& name, GLfloat const* data) {
  gl::program_uniform_matrix_3fv(id(), uniform(name), 1, false,
                                 data);
}
//------------------------------------------------------------------------------
void shader::set_uniform(const std::string& name, Mat4<GLfloat> const& data) {
  set_uniform_mat4(name, data.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void shader::set_uniform_mat4(const std::string& name, GLfloat const* data) {
  gl::program_uniform_matrix_4fv(id(), uniform(name), 1, false,
                                 data);
}
//------------------------------------------------------------------------------
std::optional<std::string> shader::info_log() {
  GLint   info_log_length = 0;
  GLsizei chars_written   = 0;

  gl::get_program_iv(id(), GL_INFO_LOG_LENGTH, &info_log_length);
  if (info_log_length == 0) {
    return {};
  }
  auto l = std::string(info_log_length, ' ');
  gl::get_program_info_log(id(), info_log_length, &chars_written, l.data());

  return l;
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
