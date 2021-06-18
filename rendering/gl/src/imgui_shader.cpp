#include <yavin/imgui_shader.h>
//==============================================================================
namespace yavin {
//==============================================================================
const std::string_view imgui_shader::vert_source =
    "#version 430\n"
    "layout (location = 0) in vec2 pos;\n"
    "layout (location = 1) in vec2 uv;\n"
    "layout (location = 2) in vec4 col;\n"
    "uniform mat4 projection_matrix;\n"
    "out vec2 frag_uv;\n"
    "out vec4 frag_col;\n"
    "void main() {\n"
    "  frag_uv = uv;\n"
    "  frag_col = col;\n"
    "  gl_Position = projection_matrix * vec4(pos.xy,0,1);\n"
    "}";
//------------------------------------------------------------------------------
const std::string_view imgui_shader::frag_source =
    "#version 430\n"
    "in vec2 frag_uv;\n"
    "in vec4 frag_col;\n"
    "uniform sampler2D tex;\n"
    "layout (location = 0) out vec4 out_color;\n"
    "void main() {\n"
    "  out_color = frag_col * texture(tex, frag_uv.st);\n"
    "}";
//==============================================================================
imgui_shader::imgui_shader() {
  add_stage<vertexshader>(shadersource{vert_source});
  add_stage<fragmentshader>(shadersource{frag_source});
  create();
}
//------------------------------------------------------------------------------
void imgui_shader::set_projection_matrix(std::array<GLfloat, 16> const& p) {
  set_uniform_mat4("projection_matrix", p.data());
}
//------------------------------------------------------------------------------
void imgui_shader::set_texture_slot(int s) {
  set_uniform("tex", s);
}
//==============================================================================
}  // namespace yavin
//==============================================================================
