#include <tatooine/rendering/gl/glincludes.h>
#include <tatooine/rendering/gl/fragmentshader.h>
//==============================================================================
namespace tatooine::rendering::gl {
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
}  // namespace tatooine::rendering::gl
//==============================================================================
