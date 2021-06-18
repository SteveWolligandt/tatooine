#ifndef YAVIN_GL_WRAPPER_H
#define YAVIN_GL_WRAPPER_H
//==============================================================================
#include <array>

#include <yavin/dllexport.h>
#include <yavin/errorcheck.h>
#include <yavin/glincludes.h>
//==============================================================================
namespace yavin {
//==============================================================================
DLL_API auto clear_color_buffer() -> void;
DLL_API auto clear_depth_buffer() -> void;
DLL_API auto clear_color_depth_buffer() -> void;
//==============================================================================
DLL_API auto enable_depth_test() -> void;
DLL_API auto disable_depth_test() -> void;
DLL_API auto depth_test_enabled() -> GLboolean;
DLL_API auto enable_depth_write() -> void;
DLL_API auto disable_depth_write() -> void;
//==============================================================================
DLL_API auto depth_func_never() -> void;
DLL_API auto depth_func_less() -> void;
DLL_API auto depth_func_equal() -> void;
DLL_API auto depth_func_lequal() -> void;
DLL_API auto depth_func_greater() -> void;
DLL_API auto depth_func_notequal() -> void;
DLL_API auto depth_func_gequal() -> void;
DLL_API auto depth_func_always() -> void;
//==============================================================================
DLL_API auto enable_multisampling() -> void;
DLL_API auto disable_multisampling() -> void;
DLL_API auto multisampling_enabled() -> bool;
//==============================================================================
DLL_API auto enable_blending() -> void;
DLL_API auto disable_blending() -> void;
//==============================================================================
DLL_API auto enable_scissor_test() -> void;
DLL_API auto disable_scissor_test() -> void;
//==============================================================================
DLL_API auto enable_face_culling() -> void;
DLL_API auto disable_face_culling() -> void;
DLL_API auto set_front_face_culling() -> void;
DLL_API auto set_back_face_culling() -> void;
DLL_API auto face_culling_enabled() -> GLboolean;
//==============================================================================
DLL_API auto blending_enabled() -> GLboolean;
DLL_API auto blend_func_additive() -> void;
DLL_API auto blend_func_multiplicative() -> void;
DLL_API auto blend_func_subtractive() -> void;
DLL_API auto blend_func_alpha() -> void;
//==============================================================================
DLL_API auto get_total_available_memory() -> GLint;
DLL_API auto get_current_available_memory() -> GLint;
//==============================================================================
DLL_API auto scissor_test_enabled() -> GLboolean;
DLL_API auto current_scissor_box() -> std::array<GLint, 4>;
DLL_API auto current_blend_src_rgb() -> GLenum;
DLL_API auto current_blend_dst_rgb() -> GLenum;
DLL_API auto current_blend_src_alpha() -> GLenum;
DLL_API auto current_blend_dst_alpha() -> GLenum;
DLL_API auto current_blend_equation_rgb() -> GLenum;
DLL_API auto current_blend_equation_alpha() -> GLenum;
DLL_API auto current_active_texture() -> GLenum;
DLL_API auto current_clip_origin() -> GLenum;
DLL_API auto current_polygon_mode() -> std::array<GLenum, 2>;
DLL_API auto current_viewport() -> std::array<GLint, 4>;
DLL_API auto bound_sampler() -> GLuint;
DLL_API auto bound_program() -> GLuint;
DLL_API auto bound_vertexbuffer() -> GLuint;
DLL_API auto bound_vertexarray() -> GLuint;
DLL_API auto bound_texture(GLenum binding) -> GLuint;
DLL_API auto bound_texture1d() -> GLuint;
DLL_API auto bound_texture2d() -> GLuint;
DLL_API auto bound_texture3d() -> GLuint;
//==============================================================================
DLL_API auto max_compute_shared_memory_size() -> GLint;
DLL_API auto barrier() -> void;
DLL_API auto shader_storage_barrier() -> void;
DLL_API auto shader_image_access_barrier() -> void;
DLL_API auto atomic_counter_barrier() -> void;
DLL_API auto max_compute_work_group_count() -> std::array<GLint, 3>;
DLL_API auto max_compute_work_group_size() -> std::array<GLint, 3>;
DLL_API auto max_compute_work_group_invocations() -> GLint;
DLL_API auto max_compute_image_uniforms() -> GLint;
//==============================================================================
DLL_API auto opengl_version() -> std::pair<GLint, GLint>;
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
