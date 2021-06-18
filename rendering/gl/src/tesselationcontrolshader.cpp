#include <yavin/glincludes.h>
#include <yavin/tesselationcontrolshader.h>
//==============================================================================
namespace yavin {
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
}  // namespace yavin
//==============================================================================
