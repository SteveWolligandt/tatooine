#include <tatooine/rendering/gl/geometryshader.h>

#include <tatooine/rendering/gl/glincludes.h>
//==============================================================================
namespace tatooine::rendering::gl {
//==============================================================================
geometryshader::geometryshader(std::filesystem::path const& sourcepath)
    : shaderstage{GL_GEOMETRY_SHADER, sourcepath} {}
//------------------------------------------------------------------------------
geometryshader::geometryshader(shadersource const& source)
    : shaderstage{GL_GEOMETRY_SHADER, source} {}
//------------------------------------------------------------------------------
geometryshader::geometryshader(geometryshader&& other)
    : shaderstage{std::move(other)} {}
//==============================================================================
}  // namespace tatooine::rendering::gl
//==============================================================================
