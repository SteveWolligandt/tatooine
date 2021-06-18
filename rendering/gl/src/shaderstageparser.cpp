#include <yavin/shaderstageparser.h>

#include <sstream>
//==============================================================================
namespace yavin {
//==============================================================================
std::regex const shaderstageparser::regex_var{
    R"((layout\s*\(location\s*=\s*\d+\s*\))?\s*(in|out|uniform)\s*(float|double|int|uint|bool|sampler\dD|mat\d|vec\d|ivec\d|uvec\d|bvec\d|dvec\d)\s*(.*)[;])"};
//------------------------------------------------------------------------------
std::regex const shaderstageparser::regex_include{R"(#include\s+\"(.*)\")"};
//==============================================================================
auto shaderstageparser::parse(std::filesystem::path const& path,
                              std::vector<GLSLVar>& vars, include_tree& it)
    -> shadersource {
  it.path()            = path;
  auto          folder = path.parent_path();
  std::ifstream file{path};

  if (!file.is_open()) {
    throw std::runtime_error{"ERROR: Unable to open file " + path.string()};
  }
  return parse_stream(file, vars, it, folder);
}
//------------------------------------------------------------------------------
auto shaderstageparser::parse(shadersource const&   source,
                              std::vector<GLSLVar>& vars, include_tree& it)
    -> shadersource {
  it.path() = "from string";
  std::stringstream stream(source.string());
  return parse_stream(stream, vars, it);
}
//------------------------------------------------------------------------------
auto shaderstageparser::parse_varname(std::string const& line)
    -> std::optional<GLSLVar> {
  std::smatch match;
  std::regex_match(line, match, regex_var);
  if (!match.str(4).empty()) {
    return GLSLVar{[](std::string const& mod_name) {
                     if (mod_name == "uniform") {
                       return GLSLVar::UNIFORM;
                     } else if (mod_name == "in") {
                       return GLSLVar::IN;
                     } else if (mod_name == "out") {
                       return GLSLVar::OUT;
                     } else {
                       return GLSLVar::UNKNOWN;
                     }
                   }(match.str(2)),
                   match.str(3), match.str(4)};
  }
  return {};
}
//------------------------------------------------------------------------------
auto shaderstageparser::parse_include(std::string const& line)
    -> std::optional<std::string> {
  std::smatch match;

  std::regex_match(line, match, regex_include);

  if (match.str(1).size() > 0) {
    return match.str(1);
  }
  return {};
}
//==============================================================================
}  // namespace yavin
//==============================================================================
