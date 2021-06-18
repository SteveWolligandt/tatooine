#include <tatooine/rendering/gl/glincludes.h>
#include <tatooine/rendering/gl/tesselationcontrolshader.h>
//==============================================================================
namespace tatooine::rendering::gl {
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
}  // namespace tatooine::rendering::gl
//==============================================================================
