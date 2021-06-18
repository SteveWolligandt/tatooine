#include <tatooine/rendering/gl/glincludes.h>
#include <tatooine/rendering/gl/tesselationevaluationshader.h>
//==============================================================================
namespace tatooine::rendering::gl {
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
}  // namespace tatooine::rendering::gl
//==============================================================================
