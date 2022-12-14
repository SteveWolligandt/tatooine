#ifndef TATOOINE_GL_SHADERSTAGEPARSER_H
#define TATOOINE_GL_SHADERSTAGEPARSER_H
//==============================================================================
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>

#include <tatooine/gl/dllexport.h>
#include <tatooine/gl/glslvar.h>
#include <tatooine/gl/includetree.h>
#include <tatooine/gl/shadersource.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class shaderstageparser {
 private:
  static std::regex const regex_var;
  static std::regex const regex_include;
 public:
  //------------------------------------------------------------------------------
  DLL_API static auto parse(std::filesystem::path const& path,
                            std::vector<GLSLVar>& vars, include_tree& it)
      -> shadersource;
  DLL_API static auto parse(shadersource const&   source,
                            std::vector<GLSLVar>& vars, include_tree& it)
      -> shadersource;
  DLL_API static auto parse_varname(std::string const& line)
      -> std::optional<GLSLVar>;
  DLL_API static auto parse_include(std::string const& line)
      -> std::optional<std::string>;

 private:
  template <typename Stream>
  static auto parse_stream(Stream& stream, std::vector<GLSLVar>& vars,
                           include_tree& it, std::string const& folder = "")
      -> shadersource {
    std::string line;
    shadersource content;

    auto line_number = std::size_t{};

    while (std::getline(stream, line)) {
      if (auto parsed_var = parse_varname(line); parsed_var)
        vars.push_back(parsed_var.value());

      if (auto parsed_include = parse_include(line); parsed_include) {
        it.nested_include_trees().emplace_back(line_number, 0,
                                               parsed_include.value(), it);
        content.string() += parse(folder + parsed_include.value(), vars,
                         it.nested_include_trees().back()).string();
      } else
        content.string() += line + '\n';

      ++line_number;
    }
    it.num_lines() = line_number;

    return content;
  }
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
