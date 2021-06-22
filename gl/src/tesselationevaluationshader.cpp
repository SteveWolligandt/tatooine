#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/tesselationevaluationshader.h>
//==============================================================================
namespace tatooine::gl {
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
}  // namespace tatooine::gl
//==============================================================================
