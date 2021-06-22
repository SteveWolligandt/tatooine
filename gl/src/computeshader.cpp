#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/computeshader.h>
//==============================================================================
namespace tatooine::gl {
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
}  // namespace tatooine::gl
//==============================================================================
