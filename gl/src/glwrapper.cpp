#include <tatooine/gl/glfunctions.h>
#include <tatooine/gl/glwrapper.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
auto clear_color_buffer() -> void { gl::clear(GL_COLOR_BUFFER_BIT); }
auto clear_depth_buffer() -> void { gl::clear(GL_DEPTH_BUFFER_BIT); }
auto clear_color_depth_buffer() -> void {
  gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
//==============================================================================
auto enable_depth_test() -> void { gl::enable(GL_DEPTH_TEST); }
auto disable_depth_test() -> void { gl::disable(GL_DEPTH_TEST); }
//==============================================================================
auto enable_depth_write() -> void { gl::depth_mask(GL_TRUE); }
auto disable_depth_write() -> void { gl::depth_mask(GL_FALSE); }
//==============================================================================
auto depth_func_never() -> void { gl::depth_func(GL_NEVER); }
auto depth_func_less() -> void { gl::depth_func(GL_LESS); }
auto depth_func_equal() -> void { gl::depth_func(GL_EQUAL); }
auto depth_func_lequal() -> void { gl::depth_func(GL_LEQUAL); }
auto depth_func_greater() -> void { gl::depth_func(GL_GREATER); }
auto depth_func_notequal() -> void { gl::depth_func(GL_NOTEQUAL); }
auto depth_func_gequal() -> void { gl::depth_func(GL_GEQUAL); }
auto depth_func_always() -> void { gl::depth_func(GL_ALWAYS); }
//==============================================================================
auto enable_blending() -> void { gl::enable(GL_BLEND); }
auto disable_blending() -> void { gl::disable(GL_BLEND); }
//==============================================================================
auto enable_scissor_test() -> void { gl::enable(GL_SCISSOR_TEST); }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto disable_scissor_test() -> void { gl::disable(GL_SCISSOR_TEST); }
//==============================================================================
auto enable_face_culling() -> void { gl::enable(GL_CULL_FACE); }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto disable_face_culling() -> void { gl::disable(GL_CULL_FACE); }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto set_front_face_culling() -> void { gl::cull_face(GL_FRONT); }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto set_back_face_culling() -> void { gl::cull_face(GL_BACK); }
//------------------------------------------------------------------------------
auto face_culling_enabled() -> GLboolean { return glIsEnabled(GL_CULL_FACE); }
//==============================================================================
auto enable_multisampling() -> void { gl::enable(GL_MULTISAMPLE); }
auto disable_multisampling() -> void { gl::disable(GL_MULTISAMPLE); }
auto multisampling_enabled() -> bool { return gl::is_enabled(GL_MULTISAMPLE); }
//==============================================================================
auto blend_func_additive() -> void { gl::blend_func(GL_ONE, GL_ONE); }
auto blend_func_multiplicative() -> void { gl::blend_func(GL_ONE, GL_ONE); }
auto blend_func_subtractive() -> void { gl::blend_func(GL_ONE, GL_ONE); }
auto blend_func_alpha() -> void {
  gl::blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}
//------------------------------------------------------------------------------
auto get_total_available_memory() -> GLint {
  constexpr auto GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX = 0x9048;
  auto           tam                                     = GLint{};
  gl::get_integer_v(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &tam);
  return tam;
}
//------------------------------------------------------------------------------
auto get_current_available_memory() -> GLint {
  constexpr auto GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX = 0x9049;
  auto           tam                                       = GLint{};
  gl::get_integer_v(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &tam);
  return tam;
}
//==============================================================================
auto blending_enabled() -> GLboolean { return glIsEnabled(GL_BLEND); }
//------------------------------------------------------------------------------
auto depth_test_enabled() -> GLboolean { return glIsEnabled(GL_DEPTH_TEST); }
//------------------------------------------------------------------------------
auto scissor_test_enabled() -> GLboolean {
  return glIsEnabled(GL_SCISSOR_TEST);
}
//------------------------------------------------------------------------------
auto current_viewport() -> std::array<GLint, 4> {
  auto cur = std::array<GLint, 4>{};
  gl::get_integer_v(GL_VIEWPORT, cur.data());
  return cur;
}
//------------------------------------------------------------------------------
auto current_scissor_box() -> std::array<GLint, 4> {
  auto cur = std::array<GLint, 4>{};
  gl::get_integer_v(GL_SCISSOR_BOX, cur.data());
  return cur;
}
//------------------------------------------------------------------------------
auto current_blend_src_rgb() -> GLenum {
  auto cur = GLint{};
  gl::get_integer_v(GL_BLEND_SRC_RGB, &cur);
  return static_cast<GLenum>(cur);
}
//------------------------------------------------------------------------------
auto current_blend_dst_rgb() -> GLenum {
  auto cur = GLint{};
  gl::get_integer_v(GL_BLEND_DST_RGB, &cur);
  return static_cast<GLenum>(cur);
}
//------------------------------------------------------------------------------
auto current_blend_src_alpha() -> GLenum {
  auto cur = GLint{};
  gl::get_integer_v(GL_BLEND_SRC_ALPHA, &cur);
  return static_cast<GLenum>(cur);
}
//------------------------------------------------------------------------------
auto current_blend_dst_alpha() -> GLenum {
  auto cur = GLint{};
  gl::get_integer_v(GL_BLEND_DST_ALPHA, &cur);
  return static_cast<GLenum>(cur);
}
//------------------------------------------------------------------------------
auto current_blend_equation_rgb() -> GLenum {
  auto cur = GLint{};
  gl::get_integer_v(GL_BLEND_EQUATION_RGB, &cur);
  return static_cast<GLenum>(cur);
}
//------------------------------------------------------------------------------
auto current_blend_equation_alpha() -> GLenum {
  auto cur = GLint{};
  gl::get_integer_v(GL_BLEND_EQUATION_ALPHA, &cur);
  return static_cast<GLenum>(cur);
}
//------------------------------------------------------------------------------
auto current_polygon_mode() -> std::array<GLenum, 2> {
  std::array<GLint, 2> cur;
  gl::get_integer_v(GL_POLYGON_MODE, cur.data());
  return {static_cast<GLenum>(cur[0]), static_cast<GLenum>(cur[1])};
}
//------------------------------------------------------------------------------
auto bound_program() -> GLuint {
  auto cur = GLint{};
  gl::get_integer_v(GL_CURRENT_PROGRAM, &cur);
  return static_cast<GLuint>(cur);
}
//------------------------------------------------------------------------------
auto bound_vertexbuffer() -> GLuint {
  auto cur = GLint{};
  gl::get_integer_v(GL_ARRAY_BUFFER_BINDING, &cur);
  return static_cast<GLuint>(cur);
}
//------------------------------------------------------------------------------
auto bound_vertexarray() -> GLuint {
  auto cur = GLint{};
  gl::get_integer_v(GL_VERTEX_ARRAY_BINDING, &cur);
  return static_cast<GLuint>(cur);
}
//------------------------------------------------------------------------------
auto current_active_texture() -> GLenum {
  auto cur = GLint{};
  gl::get_integer_v(GL_ACTIVE_TEXTURE, &cur);
  return static_cast<GLenum>(cur);
}
//------------------------------------------------------------------------------
auto current_clip_origin() -> GLenum {
  auto cur = GLint{};
  gl::get_integer_v(GL_CLIP_ORIGIN, &cur);
  return static_cast<GLenum>(cur);
}
//------------------------------------------------------------------------------
auto bound_sampler() -> GLuint {
  auto cur = GLint{};
  gl::get_integer_v(GL_SAMPLER_BINDING, &cur);
  return static_cast<GLuint>(cur);
}
//------------------------------------------------------------------------------
auto bound_texture(GLenum binding) -> GLuint {
  auto tex = GLint{};
  gl::get_integer_v(binding, &tex);
  return static_cast<GLuint>(tex);
}
//------------------------------------------------------------------------------
auto bound_texture1d() -> GLuint {
  return bound_texture(GL_TEXTURE_BINDING_1D);
}
//------------------------------------------------------------------------------
auto bound_texture2d() -> GLuint {
  return bound_texture(GL_TEXTURE_BINDING_2D);
}
//------------------------------------------------------------------------------
auto bound_texture3d() -> GLuint {
  return bound_texture(GL_TEXTURE_BINDING_3D);
}
//==============================================================================
auto max_compute_shared_memory_size() -> GLint {
  auto s = GLint{};
  gl::get_integer_v(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &s);
  return s;
}
//------------------------------------------------------------------------------
auto barrier() -> void { gl::memory_barrier(GL_ALL_BARRIER_BITS); }
//------------------------------------------------------------------------------
auto shader_storage_barrier() -> void {
  gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
//------------------------------------------------------------------------------
auto shader_image_access_barrier() -> void {
  gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}
//------------------------------------------------------------------------------
auto atomic_counter_barrier() -> void {
  gl::memory_barrier(GL_ATOMIC_COUNTER_BARRIER_BIT);
}
//------------------------------------------------------------------------------
auto max_compute_work_group_count() -> std::array<GLint, 3> {
  auto work_grp_cnt = std::array<GLint, 3>{};
  gl::get_integeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_cnt[0]);
  gl::get_integeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_cnt[1]);
  gl::get_integeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_cnt[2]);
  return work_grp_cnt;
}
//------------------------------------------------------------------------------
auto max_compute_work_group_size() -> std::array<GLint, 3> {
  auto work_grp_size = std::array<GLint, 3>{};
  gl::get_integeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
  gl::get_integeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
  gl::get_integeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);
  return work_grp_size;
}
//------------------------------------------------------------------------------
auto max_compute_work_group_invocations() -> GLint {
  auto val = GLint{};
  gl::get_integer_v(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &val);
  return val;
}
//------------------------------------------------------------------------------
auto max_compute_image_uniforms() -> GLint {
  auto val = GLint{};
  gl::get_integer_v(GL_MAX_COMPUTE_IMAGE_UNIFORMS, &val);
  return val;
}
//==============================================================================
auto opengl_version() -> std::pair<GLint, GLint> {
  auto version = std::pair<GLint, GLint>{};
  gl::get_integer_v(GL_MAJOR_VERSION, &version.first);
  gl::get_integer_v(GL_MINOR_VERSION, &version.second);
  return version;
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
