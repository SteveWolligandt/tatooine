#include <tatooine/rendering/gl/glincludes.h>
#include <tatooine/rendering/gl/computeshader.h>
//==============================================================================
namespace tatooine::rendering::gl {
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
}  // namespace tatooine::rendering::gl
//==============================================================================
