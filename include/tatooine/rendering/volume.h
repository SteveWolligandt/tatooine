#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/field_operations.h>
#include <tatooine/gl/framebuffer.h>
#include <tatooine/gl/texture.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/color_scales/magma.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gpu/upload.h>
#include <tatooine/grid.h>
#include <tatooine/rendering/first_person_window.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename DimX, typename DimY, typename DimZ, floating_point ValueType,
          bool HasNonConstReference>
auto interactive(
    typed_grid_vertex_property_interface<grid<DimX, DimY, DimZ>, ValueType,
                                         HasNonConstReference> const& prop)
    -> void {
  auto       win = rendering::first_person_window{};
  auto const gpu_tex   = gpu::upload(prop);
  auto const color_scale_tex   = color_scales::viridis<float>{}.to_gpu_tex();
  auto       cube_data = gl::indexeddata<vec3f>{};
  cube_data.vertexbuffer().resize(8);
  cube_data.indexbuffer().resize(36);
  {
    size_t i    = 0;
    auto   data = cube_data.vertexbuffer().wmap();
    data[i++]   = vec3f{0, 0, 0};
    data[i++]   = vec3f{1, 0, 0};
    data[i++]   = vec3f{0, 1, 0};
    data[i++]   = vec3f{1, 1, 0};
    data[i++]   = vec3f{0, 0, 1};
    data[i++]   = vec3f{1, 0, 1};
    data[i++]   = vec3f{0, 1, 1};
    data[i++]   = vec3f{1, 1, 1};
  }
  {
    size_t i    = 0;
    auto   data = cube_data.indexbuffer().wmap();
    // front
    data[i++] = 0;
    data[i++] = 1;
    data[i++] = 2;
    data[i++] = 1;
    data[i++] = 3;
    data[i++] = 2;

    // back
    data[i++] = 4;
    data[i++] = 7;
    data[i++] = 5;
    data[i++] = 4;
    data[i++] = 6;
    data[i++] = 7;

    // right
    data[i++] = 1;
    data[i++] = 5;
    data[i++] = 3;
    data[i++] = 5;
    data[i++] = 7;
    data[i++] = 3;

    // left
    data[i++] = 0;
    data[i++] = 2;
    data[i++] = 4;
    data[i++] = 2;
    data[i++] = 6;
    data[i++] = 4;

    // top
    data[i++] = 2;
    data[i++] = 3;
    data[i++] = 6;
    data[i++] = 3;
    data[i++] = 7;
    data[i++] = 6;

    // bottom
    data[i++] = 0;
    data[i++] = 4;
    data[i++] = 1;
    data[i++] = 1;
    data[i++] = 4;
    data[i++] = 5;
  }
  auto       screenspace_quad_data = gl::indexeddata<vec2f>{};
  screenspace_quad_data.vertexbuffer().resize(4);
  screenspace_quad_data.indexbuffer().resize(6);
  {
    size_t i    = 0;
    auto   data = screenspace_quad_data.vertexbuffer().wmap();
    data[i++]   = vec2f{-1, -1};
    data[i++]   = vec2f{1, -1};
    data[i++]   = vec2f{-1, 1};
    data[i++]   = vec2f{1, 1};
  }
  {
    size_t i    = 0;
    auto   data = screenspace_quad_data.indexbuffer().wmap();
    data[i++] = 0;
    data[i++] = 1;
    data[i++] = 2;
    data[i++] = 1;
    data[i++] = 3;
    data[i++] = 2;
  }

  struct position_shader_t : gl::shader {
    position_shader_t() {
      add_stage<gl::vertexshader>(gl::shadersource{
          "#version 330 core\n"
          "layout(location = 0) in vec3 pos;\n"
          "out vec3 frag_pos;\n"
          "uniform mat4 projection_matrix;\n"
          "uniform mat4 modelview_matrix;\n"
          "//"
          "------------------------------------------------------------------\n"
          "void main() {\n"
          "  gl_Position = projection_matrix * modelview_matrix * vec4(pos, "
          "1);\n"
          "  frag_pos = pos;\n"
          "}\n"});
      add_stage<gl::fragmentshader>(gl::shadersource{
          "#version 330 core\n"
          "in vec3 frag_pos;\n"
          "out vec4 frag_out;\n"
          "//"
          "------------------------------------------------------------------\n"
          "void main() {\n"
          "  frag_out = vec4(frag_pos, 1);\n"
          "}\n"});
      create();
    }
    auto set_modelview_matrix(mat4f const& MV) -> void {
      set_uniform_mat4("modelview_matrix", MV.data_ptr());
    }
    auto set_projection_matrix(mat4f const& P) -> void {
      set_uniform_mat4("projection_matrix", P.data_ptr());
    }
  } position_shader;
  struct dvr_shader_t : gl::shader {
    dvr_shader_t() {
      add_stage<gl::vertexshader>(gl::shadersource{
          "#version 330 core\n"
          "layout(location = 0) in vec2 pos;\n"
          "//----------------------------------------------------------------\n"
          "void main() {\n"
          "  gl_Position =  vec4(pos, 0, 1);\n"
          "}\n"});
      add_stage<gl::fragmentshader>(gl::shadersource{
          "#version 430 core\n"
          "uniform sampler3D volume_data;\n"
          "uniform sampler1D color_scale;\n"
          "uniform int mode;\n"
          "uniform float min;\n"
          "uniform float max;\n"
          "uniform float ray_offset;\n"
          "layout(rgba32f, binding = 0) readonly uniform image2D front_cube;\n"
          "layout(rgba32f, binding = 1) readonly uniform image2D back_cube;\n"
          "out vec4 frag_out;\n"
          "//----------------------------------------------------------------\n"
          "void main() {\n"
          "  vec4 front = imageLoad(front_cube, ivec2(gl_FragCoord.xy));\n"
          "  vec4 back  = imageLoad(back_cube,  ivec2(gl_FragCoord.xy));\n"
          "  if (front.w == 0 || back.w == 0) {frag_out = vec4(1,1,1,1);return;}"
          "  if (mode == 0) {\n"
          "    vec3 direction = back.xyz - front.xyz;\n"
          "    float distance = length(direction);\n"
          "    direction = normalize(direction);\n"
          "    int num_steps = int(distance / ray_offset);\n"
          "    float actual_ray_offset = distance / num_steps;\n"
          "    vec4 accumulator = vec4(0,0,0,0);\n"
          "    for (int i = 0; i < num_steps; ++i) {\n"
          "      vec3 cur_pos            = front.xyz + direction * i * actual_ray_offset;"
          "      float s                 = texture(volume_data, cur_pos).x;\n"
          "      float normalized_sample = clamp((s - min) / (max - min), 0, 1);\n"
          "      float cur_alpha         = clamp(normalized_sample, 0, 1);\n"
          "      vec3 color              = texture(color_scale, normalized_sample).xyz;\n"
          "      accumulator.rgb += (1 - accumulator.a) * cur_alpha * color;\n"
          "      accumulator.a += (1 - accumulator.a) * cur_alpha;\n"
          "      if (accumulator.a >= 0.95) { break; }\n"
          "    }\n"
          "    frag_out.xyz = vec3(1 - accumulator.a) + accumulator.xyz*accumulator.a  ;\n"
          "  } else if (mode == 1) {\n"
          "    frag_out = front;\n"
          "  } else if (mode == 2) {\n"
          "    frag_out = back;\n"
          "  } else if (mode == 3) {\n"
          "    float s = texture(volume_data, front.xyz).x;\n"
          "    float normalized_sample = (s-min) / (max-min);\n"
          "    //vec3 color = vec3(normalized_sample);\n"
          "    vec3 color = texture(color_scale, normalized_sample).xyz;\n"
          "    frag_out = vec4(color, 1);\n"
          "  }\n"
          "}\n"});
      create();
      set_mode(0);
    }
    auto set_volume_data_sampler_unit(int unit) -> void {
      set_uniform("volume_data", unit);
    }
    auto set_color_scale_sampler_unit(int unit) -> void {
      set_uniform("color_scale", unit);
    }
    auto set_mode(int mode) -> void { set_uniform("mode", mode); }
    auto set_min(float min) -> void { set_uniform("min", min); }
    auto set_max(float max) -> void { set_uniform("max", max); }
    auto set_ray_offset(float ray_offset) -> void {
      set_uniform("ray_offset", ray_offset);
    }
  } dvr_shader;

  auto  front_cube_tex = gl::tex2rgba32f{32, 32};
  auto  back_cube_tex  = gl::tex2rgba32f{32, 32};
  float min            = 0.0f;
  float max            = 1.0f;
  float ray_offset     = 0.001f;
  dvr_shader.set_volume_data_sampler_unit(0);
  dvr_shader.set_color_scale_sampler_unit(1);
  dvr_shader.set_min(min);
  dvr_shader.set_max(max);
  dvr_shader.set_ray_offset(ray_offset);

  struct listener_t : gl::window_listener {
    gl::tex2rgba32f& front_cube_tex;
    gl::tex2rgba32f& back_cube_tex;
    listener_t(gl::tex2rgba32f& f, gl::tex2rgba32f& b)
        : front_cube_tex{f}, back_cube_tex{b} {}
    auto on_resize(int width, int height) -> void override {
      front_cube_tex.resize(width, height);
      back_cube_tex.resize(width, height);
    }
  } listener{front_cube_tex, back_cube_tex};
  win.add_listener(listener);

  auto viridis_tex2 = color_scales::viridis<float>{}.to_gpu_tex2d();
  auto magma_tex2 = color_scales::magma<float>{}.to_gpu_tex2d();
  win.render_loop([&](auto const /*dt*/) {
    front_cube_tex.clear(1, 1, 1, 0);
    back_cube_tex.clear(1, 1, 1, 0);
    gl::clear_color(1, 1, 1, 1);
    gl::clear_color_depth_buffer();

    gl::enable_face_culling();
    gl::disable_depth_test();
    position_shader.bind();
    position_shader.set_modelview_matrix(
        win.camera_controller().view_matrix() *
        rendering::translation_matrix(prop.grid().template front<0>(),
                                      prop.grid().template front<1>(),
                                      prop.grid().template front<2>()) *
        rendering::scale_matrix(prop.grid().template extent<0>(),
                                prop.grid().template extent<1>(),
                                prop.grid().template extent<2>()));
    position_shader.set_projection_matrix(
        win.camera_controller().projection_matrix());

    auto front_cube_framebuffer = gl::framebuffer{front_cube_tex};
    front_cube_framebuffer.bind();
    gl::set_front_face_culling();
    cube_data.draw_triangles();

    auto back_cube_framebuffer = gl::framebuffer{back_cube_tex};
    back_cube_framebuffer.bind();
    gl::set_back_face_culling();
    cube_data.draw_triangles();

    gl::framebuffer::unbind();
    gl::disable_face_culling();
    gl::enable_depth_test();

    dvr_shader.bind();

    front_cube_tex.bind_image_texture(0);
    back_cube_tex.bind_image_texture(1);
    gpu_tex.bind(0);
    color_scale_tex.bind(1);
    screenspace_quad_data.draw_triangles();
    if (ImGui::Button("Volume")) {
      dvr_shader.set_mode(0);
    }
    ImGui::SameLine();
    if (ImGui::Button("Front")) {
      dvr_shader.set_mode(1);
    }
    ImGui::SameLine();
    if (ImGui::Button("Back")) {
      dvr_shader.set_mode(2);
    }
    ImGui::SameLine();
    if (ImGui::Button("Map")) {
      dvr_shader.set_mode(3);
    }
    if (ImGui::DragFloat("Ray Offset", &ray_offset, 0.001f, 0.0001f, 0.1f)) {
      dvr_shader.set_ray_offset(ray_offset);
    }
    if (ImGui::DragFloat("Min", &min, 0.001f,
                         -std::numeric_limits<float>::max(), max)) {
      dvr_shader.set_min(min);
    }
    if (ImGui::DragFloat("Max", &max, 0.001f, min,
                         std::numeric_limits<float>::max())) {
      dvr_shader.set_max(max);
    }
    ImGui::Begin("OpenGL Texture Text");
    auto items = std::array{std::pair{"Viridis", std::ref(viridis_tex2)},
                            std::pair{"Magma", std::ref(magma_tex2)}};
    static decltype(items)::value_type const* current_item = &items.front();

    ImGuiStyle& style     = ImGui::GetStyle();
    float       w         = ImGui::CalcItemWidth();
    float       spacing   = style.ItemInnerSpacing.x;
    float       button_sz = ImGui::GetFrameHeight();
    ImVec2 combo_pos = ImGui::GetCursorScreenPos();
    ImGui::PushItemWidth(w - spacing * 2.0f - button_sz * 2.0f);
    if (ImGui::BeginCombo("##custom combo", "")) {
      size_t i = 0;
      for (auto const& item : items) {
        bool is_selected = (current_item == &item);
        ImGui::PushID(i++);
        if (ImGui::Selectable(item.first, is_selected)) {
          current_item = &item;
        }
        ImGui::PopID();
        ImGui::SameLine();
        ImGui::Image((void*)(intptr_t)item.second.get().id(), ImVec2(100, 10));
        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine(0, style.ItemInnerSpacing.x);
    ImVec2      backup_pos = ImGui::GetCursorScreenPos();
    ImGui::SetCursorScreenPos(
        ImVec2(combo_pos.x + style.FramePadding.x, combo_pos.y));
    ImGui::Image((void*)(intptr_t)current_item->second.get().id(), ImVec2(100, 10));
    ImGui::SetCursorScreenPos(backup_pos);
    ImGui::SameLine();
    ImGui::Text(current_item->first);
    ImGui::End();
  });
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
