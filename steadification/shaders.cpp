#include "shaders.h"

//==============================================================================
using namespace yavin;

//==============================================================================
VertFragShader::VertFragShader(const std::string& vert,
                               const std::string& frag) {
  add_stage<vertexshader>(vert);
  add_stage<fragmentshader>(frag);
  create();
  set_projection(glm::mat4{1.0f});
}

//------------------------------------------------------------------------------
void VertFragShader::set_projection(const glm::mat4& projection) {
  set_uniform("projection", projection);
}

//==============================================================================
CompShader::CompShader(const std::string& comp) {
  add_stage<computeshader>(comp);
  create();
}

//------------------------------------------------------------------------------
void CompShader::dispatch2d(GLuint w, GLuint h) {
  bind();
  gl::dispatch_compute(w, h, 1);
}

//==============================================================================
StreamsurfaceBaseShader::StreamsurfaceBaseShader(const std::string& frag)
    : VertFragShader("streamsurface.vert", frag) {
  set_u_range(0, 1);
}

//------------------------------------------------------------------------------
void StreamsurfaceBaseShader::set_backward_tau_range(GLfloat min, GLfloat max) {
  set_uniform("backward_tau_range", min, max);
}

//------------------------------------------------------------------------------
void StreamsurfaceBaseShader::set_forward_tau_range(GLfloat min, GLfloat max) {
  set_uniform("forward_tau_range", min, max);
}

//------------------------------------------------------------------------------
void StreamsurfaceBaseShader::set_u_range(GLfloat min, GLfloat max) {
  set_uniform("u_range", min, max);
}

//==============================================================================
StreamsurfaceToLinkedListShader::StreamsurfaceToLinkedListShader(GLuint s)
    : StreamsurfaceBaseShader("streamsurface_to_linked_list.frag") {
  set_size(s);
}

//------------------------------------------------------------------------------
void StreamsurfaceToLinkedListShader::set_size(GLuint s) {
  set_uniform("size", s);
}

//==============================================================================
StreamsurfaceTauShader::StreamsurfaceTauShader()
    : StreamsurfaceBaseShader("streamsurface_tau.frag") {
  set_color_scale_slot(0);
}

//------------------------------------------------------------------------------
void StreamsurfaceTauShader::set_tau_range(const glm::vec2& tau_range) {
  set_uniform("tau_range", tau_range);
}

//------------------------------------------------------------------------------
void StreamsurfaceTauShader::set_color_scale_slot(int color_scale_slot) {
  set_uniform("color_scale", color_scale_slot);
}

//==============================================================================
StreamsurfaceVectorfieldShader::StreamsurfaceVectorfieldShader()
    : StreamsurfaceBaseShader("streamsurface_vectorfield.frag") {}

//==============================================================================
LinkedListToHeadVectorsShader::LinkedListToHeadVectorsShader()
    : CompShader("linked_list_to_head_vectors.comp") {}

//==============================================================================
LICShader::LICShader(GLuint num_steps, GLfloat min_x, GLfloat min_y,
                     GLfloat max_x, GLfloat max_y)
    : CompShader("lic.comp") {
  set_uniform("vector_tex", 0);
  set_uniform("noise_tex", 1);
  set_num_steps(num_steps);
  set_bounding_min(min_x, min_y);
  set_bounding_max(max_x, max_y);
}

//==============================================================================
void LICShader::set_num_steps(GLuint num_steps) {
  set_uniform("num_steps", num_steps);
}

//------------------------------------------------------------------------------
void LICShader::set_bounding_min(GLfloat x, GLfloat y) {
  set_uniform("bounding_min", x, y);
}

//------------------------------------------------------------------------------
void LICShader::set_bounding_max(GLfloat x, GLfloat y) {
  set_uniform("bounding_max", x, y);
}

//==============================================================================
WeightShader::WeightShader() : CompShader("weight.comp") {}

//------------------------------------------------------------------------------
void WeightShader::set_bw_tau(GLfloat tau) {
  set_uniform("bw_tau", tau);
}

//------------------------------------------------------------------------------
void WeightShader::set_fw_tau(GLfloat tau) {
  set_uniform("fw_tau", tau);
}

//==============================================================================
SpatialCoverageShader::SpatialCoverageShader()
    : CompShader("spatial_coverage.comp") {}

//==============================================================================
MeshViewerShader::MeshViewerShader()
    : VertFragShader("mesh_viewer.vert", "mesh_viewer.frag") {
  set_tex_slot(0);
}

//------------------------------------------------------------------------------
void MeshViewerShader::set_tex_slot(const int slot) {
  set_uniform("tex", slot);
}

//------------------------------------------------------------------------------
void MeshViewerShader::set_modelview(const glm::mat4& modelview) {
  set_uniform("modelview", modelview);
}

//==============================================================================
ScreenSpaceBaseShader::ScreenSpaceBaseShader(GLuint w, GLuint h,
                                             const std::string& frag)
    : VertFragShader("viewport_show.vert", frag) {
  resize(w, h);
}

//------------------------------------------------------------------------------
void ScreenSpaceBaseShader::resize(GLuint w, GLuint h) {
  glm::vec4          viewport{0, 0, w, h};
  orthographiccamera cam(0, 1, 0, 1, -1, 1, viewport);
  set_projection(cam.projection_matrix());
}

//==============================================================================
WeightShowShader::WeightShowShader(GLuint w, GLuint h)
    : ScreenSpaceBaseShader(w, h, "viewport_show.frag") {}

//------------------------------------------------------------------------------
void WeightShowShader::set_t0(GLfloat t0) {
  set_uniform("t0", t0);
}

//------------------------------------------------------------------------------
void WeightShowShader::set_bw_tau(GLfloat tau) {
  set_uniform("bw_tau", tau);
}

//------------------------------------------------------------------------------
void WeightShowShader::set_fw_tau(GLfloat tau) {
  set_uniform("fw_tau", tau);
}

//==============================================================================
ScreenSpaceShader::ScreenSpaceShader(GLuint w, GLuint h)
    : ScreenSpaceBaseShader(w, h, "screenspacetex.frag") {
  set_uniform("tex", int(0));
  set_resolution(w, h);
}

//------------------------------------------------------------------------------
void ScreenSpaceShader::set_resolution(GLuint w, GLuint h) {
  set_uniform("resolution", w, h);
}

//------------------------------------------------------------------------------
void ScreenSpaceShader::resize(GLuint w, GLuint h) {
  ScreenSpaceBaseShader::resize(w, h);
  set_resolution(w, h);
}

//==============================================================================
LineShader::LineShader(GLfloat r, GLfloat g, GLfloat b, GLfloat a)
    : VertFragShader("line.vert", "line.frag") {
  set_color(r, g, b, a);
}

//------------------------------------------------------------------------------
void LineShader::set_color(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
  set_uniform("color", r, g, b, a);
}

