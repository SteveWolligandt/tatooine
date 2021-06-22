#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/tesselationcontrolshader.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
tesselationcontrolshader::tesselationcontrolshader(
    std::filesystem::path const& sourcepath)
    : shaderstage{GL_TESS_CONTROL_SHADER, sourcepath} {}
//------------------------------------------------------------------------------
tesselationcontrolshader::tesselationcontrolshader(shadersource const& source)
    : shaderstage{GL_TESS_CONTROL_SHADER, source} {}
//------------------------------------------------------------------------------
tesselationcontrolshader::tesselationcontrolshader(
    tesselationcontrolshader&& other)
    : shaderstage{std::move(other)} {}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
