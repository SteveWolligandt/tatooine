#include <tatooine/gl/ansiformat.h>
#include <tatooine/gl/glfunctions.h>
#include <tatooine/gl/shaderstage.h>

#include <memory>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
std::regex const shaderstage::regex_nvidia_compiler_error(
    R"(\d+\((\d+)\)\s*:\s*(error|warning)\s*\w*:\s*(.*))");
//------------------------------------------------------------------------------
std::regex const shaderstage::regex_mesa_compiler_error(
    R"(\d+:(\d+)\(\d+\)\s*:\s*(error|warning)\s*\w*:\s*(.*))");
//==============================================================================
shaderstage::shaderstage(GLenum              shader_type,
                         shadersource const &source)
    : m_shader_type{shader_type}, m_source{source} {}
//------------------------------------------------------------------------------
shaderstage::shaderstage(GLenum shader_type, std::filesystem::path const& sourcepath)
    : m_shader_type{shader_type},
      m_source{sourcepath} {}
//------------------------------------------------------------------------------
shaderstage::shaderstage(shaderstage &&other)
    : id_holder<GLuint>{std::move(other)},
      m_shader_type{other.m_shader_type},
      m_source{std::move(other.m_source)},
      m_glsl_vars{std::move(other.m_glsl_vars)},
      m_include_tree{std::move(other.m_include_tree)} {
  other.m_delete = false;
}
//------------------------------------------------------------------------------
shaderstage::~shaderstage() {
  if (m_delete) {
    delete_stage();
  }
}
//------------------------------------------------------------------------------
auto shaderstage::type_to_string(GLenum shader_type) -> std::string_view {
  switch (shader_type) {
    case GL_VERTEX_SHADER:
      return "Vertex";
    case GL_FRAGMENT_SHADER:
      return "Fragment";
    case GL_GEOMETRY_SHADER:
      return "Geometry";
    case GL_TESS_EVALUATION_SHADER:
      return "Tesselation Evaluation";
    case GL_TESS_CONTROL_SHADER:
      return "Tesselation Control";
    case GL_COMPUTE_SHADER:
      return "Compute";
    default:
      return "unknown";
  }
}
//------------------------------------------------------------------------------
auto shaderstage::compile(bool use_ansi_color) -> void {
  delete_stage();
  set_id(gl::create_shader(m_shader_type));
  auto source = std::visit(
      [this](auto const &src) -> decltype(auto) {
        return shaderstageparser::parse(src, m_glsl_vars, m_include_tree);
      },
      m_source);
  auto source_c = source.string().c_str();
  gl::shader_source(id(), 1, &source_c, nullptr);
  gl::compile_shader(id());
  info_log(use_ansi_color);
}
//------------------------------------------------------------------------------
auto shaderstage::delete_stage() -> void {
  if (id()) gl::delete_shader(id());
  set_id(0);
}
//------------------------------------------------------------------------------
auto shaderstage::info_log(bool use_ansi_color) -> void {
  auto info_log_length = gl::get_shader_info_log_length(id());
  if (info_log_length > 0) {
    auto info_log = gl::get_shader_info_log(id(), info_log_length);
    std::istringstream is(info_log);
    std::ostringstream os;

    std::string line;
    while (std::getline(is, line)) {
      std::smatch match;

      std::regex_match(line, match, regex_nvidia_compiler_error);
      if (!match.str(0).empty()) {
        parse_compile_error(match, os, use_ansi_color);
        os << '\n';
        continue;
      }
      std::regex_match(line, match, regex_mesa_compiler_error);
      if (!match.str(0).empty()) {
        parse_compile_error(match, os, use_ansi_color);
        os << '\n';
        continue;
      }
      os << line << '\n';
    }
    throw std::runtime_error(os.str());
  }
}
//------------------------------------------------------------------------------
auto shaderstage::parse_compile_error(std::smatch &match, std::ostream &os,
                                      bool use_ansi_color) -> void {
  size_t const line_number        = stoul(match.str(1));
  auto [include_tree, error_line] = m_include_tree.parse_line(line_number - 1);

  // print file and include hierarchy
  if (use_ansi_color) {
    os << ansi::red << ansi::bold;
  }
  os << "[GLSL " << stage_name() << " Shader " << match.str(2) << "]\n";
  if (use_ansi_color) {
    os << ansi::reset;
  }

  os << "in file ";
  if (use_ansi_color) os << ansi::cyan << ansi::bold;
  os << include_tree.path() << ":";

  if (use_ansi_color) os << ansi::yellow << ansi::bold;
  os << error_line + 1;
  if (use_ansi_color) os << ansi::reset;
  os << ": " << match.str(3) << '\n';

  auto const *hierarchy = &include_tree;
  while (hierarchy->has_parent()) {
    os << "    included from ";
    if (use_ansi_color) os << ansi::bold;
    os << hierarchy->parent().path();

    if (use_ansi_color) os << ansi::reset;
    os << ":";
    if (use_ansi_color) os << ansi::bold;
    os << hierarchy->line_number() << '\n';
    if (use_ansi_color) os << ansi::reset;
    hierarchy = &hierarchy->parent();
  }

  print_line(include_tree.path(), error_line, os);
}
//------------------------------------------------------------------------------
auto shaderstage::print_line(std::filesystem::path const &path,
                             size_t line_number, std::ostream &os) -> void {
  std::ifstream file{path};

  if (file.is_open()) {
    std::string line;
    size_t      line_cnt = 0;
    while (std::getline(file, line)) {
      if (line_cnt == line_number) {
        os << "  " << line << "\n  ";
        os << ansi::bold;
        for (size_t i = 0; i < line.size(); ++i)
          os << '~';
        os << ansi::reset;
        break;
      }
      ++line_cnt;
    }
    file.close();
  }
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
