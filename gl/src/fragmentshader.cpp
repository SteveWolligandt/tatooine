#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/fragmentshader.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
fragmentshader::fragmentshader(std::filesystem::path const& sourcepath)
    : shaderstage{GL_FRAGMENT_SHADER, sourcepath} {}
//------------------------------------------------------------------------------
fragmentshader::fragmentshader(shadersource const& source)
    : shaderstage{GL_FRAGMENT_SHADER, source} {}
//------------------------------------------------------------------------------
fragmentshader::fragmentshader(fragmentshader&& other)
    : shaderstage(std::move(other)) {}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
