#ifndef YAVIN_SHADER_H
#define YAVIN_SHADER_H
//==============================================================================
#include <string>
#include <array>
#include <optional>

#include <map>
#include <type_traits>
#include <unordered_set>
#include "computeshader.h"
#include "fragmentshader.h"
#include "geometryshader.h"
#include "tesselationcontrolshader.h"
#include "tesselationevaluationshader.h"
#include "vertexshader.h"
#include "dllexport.h"

#include "windowsundefines.h"
//==============================================================================
namespace yavin {
//==============================================================================
class shader : public id_holder<GLuint> {
 public:
  shader() = default;
  ~shader() {
    if (m_delete) { delete_shader(); }
  }

  template <typename T, typename... Args>
  void add_stage(Args&&... args) {
    m_shader_stages.push_back(T{std::forward<Args>(args)...});
  }

  DLL_API void create();
  DLL_API void delete_shader();

  DLL_API virtual void bind() const;
  DLL_API virtual void unbind() const;
  DLL_API void         add_uniform(const std::string& uniformVarName);
  DLL_API void         add_attribute(const std::string& attributeVarName);
  DLL_API GLint uniform(const std::string& uniformVarName);
  DLL_API GLint attribute(const std::string& attributeVarName);

  DLL_API void set_uniform(const std::string&, GLfloat);
  DLL_API void set_uniform(const std::string&, GLint);
  DLL_API void set_uniform(const std::string&, GLuint);
  DLL_API void set_uniform(const std::string&, GLboolean);

  DLL_API void set_uniform(const std::string&, GLfloat, GLfloat);
  DLL_API void set_uniform(const std::string&, GLfloat, GLfloat, GLfloat);
  DLL_API void set_uniform(const std::string&, GLfloat, GLfloat, GLfloat, GLfloat);

  DLL_API void set_uniform(const std::string&,
                           std::array<GLfloat, 2> const& data);
  DLL_API void set_uniform(const std::string&,
                           std::array<GLfloat, 3> const& data);
  DLL_API void set_uniform(const std::string&,
                           std::array<GLfloat, 4> const& data);

  DLL_API void set_uniform(const std::string&, GLint, GLint);
  DLL_API void set_uniform(const std::string&, GLint, GLint, GLint);
  DLL_API void set_uniform(const std::string&, GLint, GLint, GLint,
                           GLint);

  DLL_API void set_uniform(const std::string&,
                           std::array<GLint, 2> const& data);
  DLL_API void set_uniform(const std::string&,
                           std::array<GLint, 3> const& data);
  DLL_API void set_uniform(const std::string&,
                           std::array<GLint, 4> const& data);

  DLL_API void set_uniform(const std::string&, GLuint, GLuint);
  DLL_API void set_uniform(const std::string&, GLuint, GLuint, GLuint);
  DLL_API void set_uniform(const std::string&, GLuint, GLuint, GLuint,
                           GLuint);

  DLL_API void set_uniform(const std::string&,
                           std::array<GLuint, 2> const& data);
  DLL_API void set_uniform(const std::string&,
                           std::array<GLuint, 3> const& data);
  DLL_API void set_uniform(const std::string&,
                           std::array<GLuint, 4> const& data);

  DLL_API void set_uniform_vec2(const std::string&, GLfloat const *);
  DLL_API void set_uniform_vec2(const std::string&, GLint const*);
  DLL_API void set_uniform_vec2(const std::string&, GLuint const*);

  DLL_API void set_uniform_vec3(const std::string&, GLfloat const*);
  DLL_API void set_uniform_vec3(const std::string&, GLint const*);
  DLL_API void set_uniform_vec3(const std::string&, GLuint const*);

  DLL_API void set_uniform_vec4(const std::string&, GLfloat const*);
  DLL_API void set_uniform_vec4(const std::string&, GLint const*);
  DLL_API void set_uniform_vec4(const std::string&, GLuint const*);

  DLL_API void set_uniform_mat2(const std::string&, GLfloat const *);
  DLL_API void set_uniform_mat3(const std::string&, GLfloat const *);
  DLL_API void set_uniform_mat4(const std::string&, GLfloat const *);

  DLL_API std::optional<std::string> info_log();

 private:
  std::map<std::string, GLint>    m_uniform_locations;
  std::map<std::string, GLint>    m_attribute_locations;
  std::vector<shaderstage>        m_shader_stages;
  std::unordered_set<std::string> m_uniform_var_names;
  std::unordered_set<std::string> m_attribute_var_names;

  bool m_delete = true;
};

//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
