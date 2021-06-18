#include <yavin/vertexshader.h>
//==============================================================================
namespace yavin {
//==============================================================================
vertexshader::vertexshader(std::filesystem::path const& sourcepath)
    : shaderstage{GL_VERTEX_SHADER, sourcepath} {}
//------------------------------------------------------------------------------
vertexshader::vertexshader(shadersource const& source)
    : shaderstage{GL_VERTEX_SHADER, source} {}
//------------------------------------------------------------------------------
vertexshader::vertexshader(vertexshader&& other)
    : shaderstage{std::move(other)} {}
//==============================================================================
}  // namespace yavin
//==============================================================================
