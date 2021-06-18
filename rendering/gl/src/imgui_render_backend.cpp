#include <yavin/imgui_render_backend.h>

#include <iostream>
//==============================================================================
namespace yavin {
//==============================================================================
imgui_render_backend::imgui_render_backend() {
  // Query for GL version
  auto const [major, minor] = opengl_version();
  m_gl_version              = major * 1000 + minor;

  // Setup back-end capabilities flags
  ImGuiIO& io            = ImGui::GetIO();
  io.BackendRendererName = "yavin";

  if (m_gl_version >= 3200) {
    io.BackendFlags |=
        ImGuiBackendFlags_RendererHasVtxOffset;  // We can honor the
                                                 // ImDrawCmd::VtxOffset field,
                                                 // allowing for large meshes.
  }

  // Store GLSL version string so we can refer to it later in case we recreate
  // shaders. Note: GLSL version is NOT the same as GL version. Leave this to
  // nullptr if unsure.
  const char* glsl_version = "#version 410";
  IM_ASSERT((int)strlen(glsl_version) + 2 <
            IM_ARRAYSIZE(m_glsl_version_string));
  strcpy(m_glsl_version_string, glsl_version);
  strcat(m_glsl_version_string, "\n");
  create_device_objects();
}
//------------------------------------------------------------------------------
void imgui_render_backend::setup_render_state(ImDrawData* draw_data,
                                              int fb_width, int fb_height,
                                              vertexarray& vao) {
  // Setup render state: alpha-blending enabled, no face culling, no depth
  // testing, scissor enabled, polygon fill
  enable_blending();
  gl::blend_equation(GL_FUNC_ADD);
  gl::blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  disable_face_culling();
  disable_depth_test();
  enable_scissor_test();
#ifdef GL_POLYGON_MODE
  gl::polygon_mode(GL_FRONT_AND_BACK, GL_FILL);
#endif

  // Setup viewport, orthographic projection matrix
  // Our visible imgui space lies from draw_data->DisplayPos (top left) to
  // draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos
  // is (0,0) for single viewport apps.
  gl::viewport(0, 0, (GLsizei)fb_width, (GLsizei)fb_height);
  float L = draw_data->DisplayPos.x;
  float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
  float T = draw_data->DisplayPos.y;
  float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
  const std::array<GLfloat, 16> ortho_projection = {2.0f / (R - L),
                                                    0.0f,
                                                    0.0f,
                                                    0.0f,
                                                    0.0f,
                                                    2.0f / (T - B),
                                                    0.0f,
                                                    0.0f,
                                                    0.0f,
                                                    0.0f,
                                                    -1.0f,
                                                    0.0f,
                                                    (R + L) / (L - R),
                                                    (T + B) / (B - T),
                                                    0.0f,
                                                    1.0f};

  m_shader.bind();
  m_shader.set_texture_slot(0);
  m_shader.set_projection_matrix(ortho_projection);
#ifdef GL_SAMPLER_BINDING
  gl::bind_sampler(0, 0);  // We use combined texture/sampler state.
                           // Applications using GL 3.3 may set that otherwise.
#endif

  vao.bind();
  m_vbo.bind();
  m_ibo.bind();
  m_vbo.activate_attributes(GL_FALSE, GL_FALSE, GL_TRUE);
}
//------------------------------------------------------------------------------
void imgui_render_backend::render_draw_data(ImDrawData* draw_data) {
  // Avoid rendering when minimized, scale coordinates for retina displays
  // (screen coordinates != framebuffer coordinates)
  int fb_width =
      (int)(draw_data->DisplaySize.x * draw_data->FramebufferScale.x);
  int fb_height =
      (int)(draw_data->DisplaySize.y * draw_data->FramebufferScale.y);
  if (fb_width <= 0 || fb_height <= 0) {
    return;
  }

  // Backup GL state
  auto last_active_texture = current_active_texture();
  auto last_program        = bound_program();
  auto last_texture        = bound_texture2d();
  gl::active_texture(GL_TEXTURE0);
#ifdef GL_SAMPLER_BINDING
  auto last_sampler = bound_sampler();
#endif
  auto last_array_buffer        = bound_vertexbuffer();
  auto last_vertex_array_object = bound_vertexarray();
#ifdef GL_POLYGON_MODE
  auto last_polygon_mode = current_polygon_mode();
#endif
  auto last_viewport             = current_viewport();
  auto last_scissor_box          = current_scissor_box();
  auto last_blend_src_rgb        = current_blend_src_rgb();
  auto last_blend_dst_rgb        = current_blend_dst_rgb();
  auto last_blend_src_alpha      = current_blend_src_alpha();
  auto last_blend_dst_alpha      = current_blend_dst_alpha();
  auto last_blend_equation_rgb   = current_blend_equation_rgb();
  auto last_blend_equation_alpha = current_blend_equation_alpha();
  auto last_enable_blend         = blending_enabled();
  auto last_enable_cull_face     = face_culling_enabled();
  auto last_enable_depth_test    = depth_test_enabled();
  auto last_enable_scissor_test  = scissor_test_enabled();
  bool clip_origin_lower_left    = true;
#if defined(GL_CLIP_ORIGIN) && !defined(__APPLE__)
  auto last_clip_origin = current_clip_origin();
  if (last_clip_origin == GL_UPPER_LEFT) {
    clip_origin_lower_left = false;
  }
#endif

  // Setup desired GL state
  // Recreate the VAO every time (this is to easily allow multiple GL contexts
  // to be rendered to. VAO are not shared among GL contexts) The renderer
  // would actually work without any VAO bound, but then our VertexAttrib
  // calls would overwrite the default one currently bound.
  vertexarray vao;
  setup_render_state(draw_data, fb_width, fb_height, vao);

  // Will project scissor/clipping rectangles into framebuffer space
  ImVec2 clip_off =
      draw_data->DisplayPos;  // (0,0) unless using multi-viewports
  ImVec2 clip_scale =
      draw_data->FramebufferScale;  // (1,1) unless using retina display which
                                    // are often (2,2)

  // Render command lists
  for (int n = 0; n < draw_data->CmdListsCount; n++) {
    const ImDrawList* cmd_list = draw_data->CmdLists[n];

    // Upload vertex/index buffers
    gl::buffer_data(GL_ARRAY_BUFFER,
                    (GLsizeiptr)cmd_list->VtxBuffer.Size * sizeof(ImDrawVert),
                    (const GLvoid*)cmd_list->VtxBuffer.Data, GL_STREAM_DRAW);
    gl::buffer_data(GL_ELEMENT_ARRAY_BUFFER,
                    (GLsizeiptr)cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx),
                    (const GLvoid*)cmd_list->IdxBuffer.Data, GL_STREAM_DRAW);

    for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
      const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
      if (pcmd->UserCallback != nullptr) {
        // User callback, registered via ImDrawList::AddCallback()
        // (ImDrawCallback_ResetRenderState is a special callback value used
        // by the user to request the renderer to reset render state.)
        if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
          setup_render_state(draw_data, fb_width, fb_height, vao);
        else
          pcmd->UserCallback(cmd_list, pcmd);
      } else {
        // Project scissor/clipping rectangles into framebuffer space
        ImVec4 clip_rect;
        clip_rect.x = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
        clip_rect.y = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
        clip_rect.z = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
        clip_rect.w = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;

        if (clip_rect.x < fb_width && clip_rect.y < fb_height &&
            clip_rect.z >= 0.0f && clip_rect.w >= 0.0f) {
          // Apply scissor/clipping rectangle
          if (clip_origin_lower_left) {
            gl::scissor((int)clip_rect.x, (int)(fb_height - clip_rect.w),
                        (int)(clip_rect.z - clip_rect.x),
                        (int)(clip_rect.w - clip_rect.y));
          } else {
            gl::scissor((int)clip_rect.x, (int)clip_rect.y, (int)clip_rect.z,
                        (int)clip_rect.w);  // Support for GL 4.5 rarely used
                                            // glClipControl(GL_UPPER_LEFT)
          }

          // Bind texture, Draw
          gl::bind_texture(GL_TEXTURE_2D, (GLuint)(intptr_t)pcmd->TextureId);
#if IMGUI_IMPL_OPENGL_MAY_HAVE_VTX_OFFSET
          if (m_gl_version >= 3200) {
            gl::draw_elements_base_bertex(
                GL_TRIANGLES, (GLsizei)pcmd->ElemCount,
                sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
                (void*)(intptr_t)(pcmd->IdxOffset * sizeof(ImDrawIdx)),
                (GLint)pcmd->VtxOffset);
          } else {
#else
          {
#endif
            gl::draw_elements(
                GL_TRIANGLES, (GLsizei)pcmd->ElemCount,
                sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
                (void*)(intptr_t)(pcmd->IdxOffset * sizeof(ImDrawIdx)));
          }
        }
      }
    }
  }

  // Restore modified GL state
  gl::use_program(last_program);

  if (gl::is_texture(last_texture)) {
    gl::bind_texture(GL_TEXTURE_2D, last_texture);
  }
#ifdef GL_SAMPLER_BINDING
  gl::bind_sampler(0, last_sampler);
#endif
  gl::active_texture(last_active_texture);
#ifndef IMGUI_IMPL_OPENGL_ES2
  gl::bind_vertex_array(last_vertex_array_object);
#endif
  gl::bind_buffer(GL_ARRAY_BUFFER, last_array_buffer);
  gl::blend_equation_separate(last_blend_equation_rgb,
                              last_blend_equation_alpha);
  gl::blend_func_separate(last_blend_src_rgb, last_blend_dst_rgb,
                          last_blend_src_alpha, last_blend_dst_alpha);
  if (last_enable_blend) {
    enable_blending();
  } else {
    disable_blending();
  }
  if (last_enable_cull_face) {
    enable_face_culling();
  } else {
    disable_face_culling();
  }
  if (last_enable_depth_test) {
    enable_depth_test();
  } else {
    disable_depth_test();
  }
  if (last_enable_scissor_test) {
    enable_scissor_test();
  } else {
    disable_scissor_test();
  }
#ifdef GL_POLYGON_MODE
  gl::polygon_mode(GL_FRONT_AND_BACK, last_polygon_mode[0]);
#endif
  gl::viewport(last_viewport[0], last_viewport[1],
               static_cast<GLsizei>(last_viewport[2]),
               static_cast<GLsizei>(last_viewport[3]));
  gl::scissor(last_scissor_box[0], last_scissor_box[1],
              static_cast<GLsizei>(last_scissor_box[2]),
              static_cast<GLsizei>(last_scissor_box[3]));
}
//------------------------------------------------------------------------------
bool imgui_render_backend::create_fonts_texture() {
  // Build texture atlas
  ImGuiIO&      io = ImGui::GetIO();
  std::uint8_t* pixels;
  int           width, height;
  io.Fonts->GetTexDataAsRGBA32(
      &pixels, &width,
      &height);  // Load as RGBA 32-bit (75% of the memory is wasted, but
                 // default font is so small) because it is more likely to be
                 // compatible with user's existing shaders. If your
                 // ImTextureId represent a higher-level concept than just a
                 // GL texture id, consider calling GetTexDataAsAlpha8()
                 // instead to save on GPU memory.

  const auto err = gl::get_error();
  if (err != GL_NO_ERROR) {
    const auto err_str = gl_error_to_string(err);
    throw gl_error(std::string{"imgui"}, err_str);
  }
  // Upload texture to graphics system
  m_font_texture.upload_data(pixels, width, height);
  // Store our identifier
  io.Fonts->TexID = (ImTextureID)(intptr_t)m_font_texture.id();
  return true;
}
//------------------------------------------------------------------------------
bool imgui_render_backend::check_shader(GLuint handle, const char* desc) {
  GLint status = 0, log_length = 0;
  gl::get_shader_iv(handle, GL_COMPILE_STATUS, &status);
  gl::get_shader_iv(handle, GL_INFO_LOG_LENGTH, &log_length);
  if ((GLboolean)status == GL_FALSE)
    std::cerr << "ERROR: imgui_render_backend::create_device_objects: failed "
                 "to compile "
              << desc << "!\n";
  if (log_length > 1) {
    ImVector<char> buf;
    buf.resize((int)(log_length + 1));
    gl::get_shader_info_log(handle, log_length, nullptr, (GLchar*)buf.begin());
    std::cerr << buf.begin() << '\n';
  }
  return (GLboolean)status == GL_TRUE;
}
//------------------------------------------------------------------------------
bool imgui_render_backend::check_program(GLuint handle, const char* desc) {
  GLint status = 0, log_length = 0;
  gl::get_program_iv(handle, GL_LINK_STATUS, &status);
  gl::get_program_iv(handle, GL_INFO_LOG_LENGTH, &log_length);
  if ((GLboolean)status == GL_FALSE)
    std::cerr << "ERROR: ImGui_ImplOpenGL3_CreateDeviceObjects: failed to link "
              << desc
              << "! "
                 "(with GLSL \'"
              << m_glsl_version_string << "\')\n";

  if (log_length > 1) {
    ImVector<char> buf;
    buf.resize((int)(log_length + 1));
    gl::get_program_info_log(handle, log_length, nullptr, (GLchar*)buf.begin());
    std::cerr << buf.begin() << '\n';
  }
  return (GLboolean)status == GL_TRUE;
}
//------------------------------------------------------------------------------
bool imgui_render_backend::create_device_objects() {
  // Backup GL state
  auto last_texture      = bound_texture2d();
  auto last_array_buffer = bound_vertexbuffer();
  auto last_vertex_array = bound_vertexarray();

  // Create buffers
  create_fonts_texture();

  // Restore modified GL state
  gl::bind_texture(GL_TEXTURE_2D, last_texture);
  gl::bind_buffer(GL_ARRAY_BUFFER, last_array_buffer);
  gl::bind_vertex_array(last_vertex_array);

  return true;
}
//==============================================================================
}  // namespace yavin
//==============================================================================
