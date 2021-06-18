#include <yavin/glincludes.h>
#include <yavin/tesselationevaluationshader.h>
//==============================================================================
namespace yavin {
//==============================================================================
tesselationevaluationshader::tesselationevaluationshader(
    std::filesystem::path const& sourcepath)
    : shaderstage{GL_TESS_EVALUATION_SHADER, sourcepath} {}
//------------------------------------------------------------------------------
tesselationevaluationshader::tesselationevaluationshader(
    shadersource const& source)
    : shaderstage{GL_TESS_EVALUATION_SHADER, source} {}
//------------------------------------------------------------------------------
tesselationevaluationshader::tesselationevaluationshader(
    tesselationevaluationshader&& other)
    : shaderstage(std::move(other)) {}
//==============================================================================
}  // namespace yavin
//==============================================================================
