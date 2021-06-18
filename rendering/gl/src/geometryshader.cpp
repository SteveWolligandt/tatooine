#include <yavin/geometryshader.h>

#include <yavin/glincludes.h>
//==============================================================================
namespace yavin {
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
}  // namespace yavin
//==============================================================================
