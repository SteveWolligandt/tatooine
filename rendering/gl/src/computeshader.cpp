#include <yavin/glincludes.h>
#include <yavin/computeshader.h>
//==============================================================================
namespace yavin {
//==============================================================================
computeshader::computeshader(std::filesystem::path const& sourcepath)
    : shaderstage{GL_COMPUTE_SHADER, sourcepath} {}
//------------------------------------------------------------------------------
computeshader::computeshader(shadersource const& source)
    : shaderstage{GL_COMPUTE_SHADER, source} {}
//------------------------------------------------------------------------------
computeshader::computeshader(computeshader&& other)
    : shaderstage(std::move(other)) {}
//==============================================================================
}  // namespace yavin
//==============================================================================
