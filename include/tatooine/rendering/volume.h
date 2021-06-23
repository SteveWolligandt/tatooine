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
  auto const color_scales = std::array{
      std::tuple{"Viridis", color_scales::viridis<float>{}.to_gpu_tex(),
                 color_scales::viridis<float>{}.to_gpu_tex2d()},
      std::tuple{"Magma", color_scales::magma<float>{}.to_gpu_tex(),
                 color_scales::magma<float>{}.to_gpu_tex2d()}};
  decltype(color_scales)::value_type const* current_color_scale =
      &color_scales.front();
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
          "  gl_Position = vec4(pos, 0, 1);\n"
          "}\n"});
      add_stage<gl::fragmentshader>(gl::shadersource{
          "#version 430 core\n"
          "uniform float shininess;\n"
          "uniform vec3 specular_color;\n"
          "uniform mat4 model_matrix;\n"
          "uniform mat4 modelview_matrix;\n"
          "uniform vec3 eps;\n"
          "uniform sampler3D volume_data;\n"
          "uniform sampler1D color_scale;\n"
          "uniform sampler1D alpha;\n"
          "uniform int mode;\n"
          "uniform float range_min;\n"
          "uniform float range_max;\n"
          "uniform float ray_offset;\n"
          "layout(rgba32f, binding = 0) readonly uniform image2D front_cube;\n"
          "layout(rgba32f, binding = 1) readonly uniform image2D back_cube;\n"
          "out vec4 frag_out;\n"
          "//----------------------------------------------------------------\n"
          "vec3 phong_brdf(vec3 light_dir, vec3 view_dir,\n"
          "                vec3 normal, vec3 diffuse_color) {\n"
          "  vec3 color       = diffuse_color;\n"
          "  vec3 reflect_dir = reflect(-light_dir, normal);\n"
          "  float spec_dot   = abs(dot(reflect_dir, view_dir));\n"
          "  color           += pow(spec_dot, shininess) * specular_color;\n"
          "  return color;\n"
          "}\n"
          "//----------------------------------------------------------------\n"
          "void main() {\n"
          "  vec4 front = imageLoad(front_cube, ivec2(gl_FragCoord.xy));\n"
          "  vec4 back  = imageLoad(back_cube,  ivec2(gl_FragCoord.xy));\n"
          "  if (front.w == 0) {\n"
          "    frag_out = vec4(1, 1, 1, 1);\n"
          "    return;\n"
          "  }\n"
          "  if (mode == 0) {\n"
          "    vec3 direction           = back.xyz - front.xyz;\n"
          "    float distance           = length(direction);\n"
          "    direction                = normalize(direction);\n"
          "    vec3 modelview_direction =\n"
          "      normalize((modelview_matrix * back).xyz -\n"
          "                (modelview_matrix * front).xyz);\n"
          "    int num_steps = int(distance / ray_offset);\n"
          "    float actual_ray_offset = distance / num_steps;\n"
          "    vec4 accumulator  = vec4(0, 0, 0, 0);\n"
          "    vec3 physical_eps = vec3(\n"
          "      length((model_matrix * vec4(2 * eps.x, 0, 0, 1)).xyz),\n"
          "      length((model_matrix * vec4(0, 2 * eps.y, 0, 1)).xyz),\n"
          "      length((model_matrix * vec4(0, 0, 2 * eps.z, 1)).xyz)\n"
          "      );\n"
          "    for (int i = 0; i < num_steps; ++i) {\n"
          "      vec3 cur_pos  = front.xyz + direction * i * actual_ray_offset;"
          "      float s       = texture(volume_data, cur_pos).x;\n"
          "      vec3 gradient = vec3(\n"
          "        (texture(volume_data, cur_pos + vec3(eps.x, 0, 0)).x -\n"
          "         texture(volume_data, cur_pos - vec3(eps.x, 0, 0)).x),\n"
          "        (texture(volume_data, cur_pos + vec3(0, eps.y, 0)).x -\n"
          "         texture(volume_data, cur_pos - vec3(0, eps.y, 0)).x),\n"
          "        (texture(volume_data, cur_pos + vec3(0, 0, eps.z)).x -\n"
          "         texture(volume_data, cur_pos - vec3(0, 0, eps.z)).x)) / physical_eps;\n"
          "      float normalized_sample = (s-range_min) / (range_max-range_min);\n"
          "      vec3 normal = normalize(gradient);\n"
          "      vec3 albedo = texture(color_scale, normalized_sample).rgb;\n"
          "      vec3 luminance = albedo * 0.1;\n"
          "      float illuminance = abs(dot(modelview_direction, normal));\n"
          "      luminance += phong_brdf(modelview_direction, modelview_direction,\n"
          "                              normal, albedo) * illuminance;\n"
          "      float cur_alpha = texture(alpha, normalized_sample).r;\n"
          "      accumulator.rgb += (1 - accumulator.a) * cur_alpha * luminance;\n"
          "      accumulator.a += (1 - accumulator.a) * cur_alpha;\n"
          "      if (accumulator.a >= 0.95) { break; }\n"
          "    }\n"
          "    frag_out.xyz = vec3(1 - accumulator.a) + accumulator.xyz * accumulator.a  ;\n"
          "  } else if (mode == 1) {\n"
          "    frag_out = front;\n"
          "  } else if (mode == 2) {\n"
          "    frag_out = back;\n"
          "  } else if (mode == 3) {\n"
          "    float s = texture(volume_data, front.xyz).x;\n"
          "    float normalized_sample = (s-range_min) / (range_max-range_min);\n"
          "    //vec3 color = vec3(normalized_sample);\n"
          "    vec3 color = texture(color_scale, normalized_sample).xyz;\n"
          "    frag_out = vec4(color, 1);\n"
          "  }\n"
          "}\n"});
      create();
      set_mode(0);
    }
    auto set_shininess(float shininess) -> void {
      set_uniform("shininess", shininess);
    }
    auto set_volume_data_sampler_unit(int unit) -> void {
      set_uniform("volume_data", unit);
    }
    auto set_color_scale_sampler_unit(int unit) -> void {
      set_uniform("color_scale", unit);
    }
    auto set_alpha_sampler_unit(int unit) -> void {
      set_uniform("alpha", unit);
    }
    auto set_model_matrix(mat4f const& M) -> void {
      set_uniform_mat4("model_matrix", M.data_ptr());
    }
    auto set_modelview_matrix(mat4f const& MV) -> void {
      set_uniform_mat4("modelview_matrix", MV.data_ptr());
    }
    auto set_mode(int mode) -> void { set_uniform("mode", mode); }
    auto set_range_min(float range_min) -> void { set_uniform("range_min", range_min); }
    auto set_specular_color(vec3f const& specular_color) -> void {
      set_uniform_vec3("specular_color", specular_color.data_ptr());
    }
    auto set_eps(vec3f const& eps) -> void {
      set_uniform_vec3("eps", eps.data_ptr());
    }
    auto set_range_max(float range_max) -> void { set_uniform("range_max", range_max); }
    auto set_ray_offset(float ray_offset) -> void {
      set_uniform("ray_offset", ray_offset);
    }
  } dvr_shader;

  auto  front_cube_tex  = gl::tex2rgba32f{32, 32};
  auto  back_cube_tex   = gl::tex2rgba32f{32, 32};
  auto  specular_color = vec3f{0.1f, 0.1f, 0.1f};
  float range_min       = 0.0f;
  float range_max       = 1.0f;
  float ray_offset      = 0.001f;
  float shininess       = 100.0f;
  size_t const       num_alpha_samples = 100;
  auto               alpha_tex         = gl::tex1r32f{num_alpha_samples};
  float v0[] = {0.0f, 0.0f, 0.5f, 0.0f};
  float v1[] = {1.0f, 1.0f, 1.0f, 0.5f};
  alpha_tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  std::vector<float> alpha_data(num_alpha_samples);
  dvr_shader.set_volume_data_sampler_unit(0);
  dvr_shader.set_color_scale_sampler_unit(1);
  dvr_shader.set_alpha_sampler_unit(2);
  dvr_shader.set_range_min(range_min);
  dvr_shader.set_range_max(range_max);
  dvr_shader.set_shininess(shininess);
  dvr_shader.set_specular_color(specular_color);
  dvr_shader.set_ray_offset(ray_offset);
  dvr_shader.set_eps(vec3f{1.0f / prop.grid().template size<0>(),
                           1.0f / prop.grid().template size<1>(),
                           1.0f / prop.grid().template size<2>()});

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

  win.render_loop([&](auto const /*dt*/) {
    front_cube_tex.clear(1, 1, 1, 0);
    back_cube_tex.clear(1, 1, 1, 0);
    gl::clear_color(1, 1, 1, 1);
    gl::clear_color_depth_buffer();

    gl::enable_face_culling();
    gl::disable_depth_test();
    position_shader.bind();
    auto const M =
        rendering::translation_matrix(prop.grid().template front<0>(),
                                      prop.grid().template front<1>(),
                                      prop.grid().template front<2>()) *
        rendering::scale_matrix(prop.grid().template extent<0>(),
                                prop.grid().template extent<1>(),
                                prop.grid().template extent<2>());
    auto const MV = win.camera_controller().view_matrix() * M;
    position_shader.set_modelview_matrix(MV);
    dvr_shader.set_model_matrix(M);
    dvr_shader.set_modelview_matrix(MV);
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
    std::get<1>(*current_color_scale).bind(1);

    for (size_t i = 0; i < num_alpha_samples; ++i) {
      float const pos = i / (float)(num_alpha_samples - 1);
      alpha_data[i]   = ImGui::BezierValue(pos, v0, v1);
    }
    alpha_tex.upload_data(alpha_data, num_alpha_samples);
    alpha_tex.bind(2);
    screenspace_quad_data.draw_triangles();

    ImGui::Begin("Settings");
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
    if (ImGui::DragFloat("Min", &range_min, 0.001f,
                         -std::numeric_limits<float>::max(), range_max)) {
      dvr_shader.set_range_min(range_min);
    }
    if (ImGui::DragFloat("Max", &range_max, 0.001f, range_min,
                         std::numeric_limits<float>::max())) {
      dvr_shader.set_range_max(range_max);
    }
    if (ImGui::DragFloat("Shininess", &shininess, 5.0f)) {
      dvr_shader.set_shininess(shininess);
    }
    if (ImGui::ColorEdit3("Specular Color", specular_color.data_ptr())) {
      dvr_shader.set_specular_color(specular_color);
    }
    ImGui::Bezier("alpha", v0, v1);

    ImGuiStyle& style     = ImGui::GetStyle();
    float       w         = ImGui::CalcItemWidth();
    float       spacing   = style.ItemInnerSpacing.x;
    float       button_sz = ImGui::GetFrameHeight();
    ImVec2 combo_pos = ImGui::GetCursorScreenPos();
    ImGui::PushItemWidth(w - spacing * 2.0f - button_sz * 2.0f);
    if (ImGui::BeginCombo("##custom combo", "")) {
      size_t i = 0;
      for (auto const& color_scale : color_scales) {
        bool is_selected = (current_color_scale == &color_scale);
        ImGui::PushID(i++);
        if (ImGui::Selectable("##foo", is_selected)) {
          current_color_scale = &color_scale;
        }
        ImGui::PopID();
        ImGui::SameLine();
        ImGui::Image((void*)(intptr_t)std::get<2>(color_scale).id(), ImVec2(100, 10));
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetColumnWidth() -
                             ImGui::CalcTextSize(std::get<0>(color_scale)).x -
                             ImGui::GetScrollX() -
                             2 * ImGui::GetStyle().ItemSpacing.x);
        ImGui::TextUnformatted(std::get<0>(color_scale));
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
    ImGui::Image((void*)(intptr_t)std::get<2>(*current_color_scale).id(), ImVec2(100, 10));
    ImGui::SetCursorScreenPos(backup_pos);
    ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetColumnWidth() -
                             ImGui::CalcTextSize(std::get<0>(*current_color_scale)).x -
                             ImGui::GetScrollX() -
                             2 * ImGui::GetStyle().ItemSpacing.x);
    ImGui::TextUnformatted(std::get<0>(*current_color_scale));
    ImGui::End();
  });
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
