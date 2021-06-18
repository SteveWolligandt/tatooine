#include <yavin/glincludes.h>
#include <yavin/fragmentshader.h>
//==============================================================================
namespace yavin {
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
}  // namespace yavin
//==============================================================================
