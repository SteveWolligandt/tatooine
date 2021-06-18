#ifndef YAVIN_SHADERSTAGE_H
#define YAVIN_SHADERSTAGE_H
//==============================================================================
#include <filesystem>
#include <variant>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <yavin/idholder.h>
#include <yavin/dllexport.h>
#include <yavin/errorcheck.h>
#include <yavin/glincludes.h>
#include <yavin/shaderstageparser.h>
//==============================================================================
namespace yavin {
//==============================================================================
class shaderstage : public id_holder<GLuint> {
  using path = std::filesystem::path;
 private:
  bool m_delete = true;

  GLenum                                            m_shader_type;
  std::variant<shadersource, std::filesystem::path> m_source;

  std::vector<GLSLVar> m_glsl_vars;
  include_tree         m_include_tree;

  static std::regex const regex_nvidia_compiler_error;
  static std::regex const regex_mesa_compiler_error;

 public:
  DLL_API static auto type_to_string(GLenum shader_type) -> std::string_view;
  //----------------------------------------------------------------------------
  DLL_API shaderstage(GLenum shader_type, shadersource const& shaderfilepath);
  //----------------------------------------------------------------------------
  DLL_API shaderstage(GLenum                       shader_type,
                      std::filesystem::path const& shaderfilepath);
  //----------------------------------------------------------------------------
  DLL_API shaderstage(shaderstage&& other);
  //----------------------------------------------------------------------------
  DLL_API ~shaderstage();
  //----------------------------------------------------------------------------
  DLL_API auto compile(bool use_ansi_color = true) -> void;
  DLL_API auto delete_stage() -> void;
  //----------------------------------------------------------------------------
  auto glsl_vars() const -> auto const& { return m_glsl_vars; }
  //----------------------------------------------------------------------------
  auto stage_name() const { return type_to_string(m_shader_type); }
  auto stage_type() const { return m_shader_type; }
  auto is_created() const { return id() != 0; }

 protected:
  DLL_API auto info_log(bool use_ansi_color = true) -> void;
  DLL_API auto parse_compile_error(std::smatch& match, std::ostream& os,
                                   bool use_ansi_color = true) -> void;
  DLL_API auto print_line(std::filesystem::path const& filename,
                          size_t line_number, std::ostream& os) -> void;
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
