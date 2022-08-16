#ifndef TATOOINE_RENDERING_INTERACTIVE_COOK_TORRANCE_BRDF_SHADER_H
#define TATOOINE_RENDERING_INTERACTIVE_COOK_TORRANCE_BRDF_SHADER_H
//==============================================================================
#include <tatooine/gl/shader.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
struct cook_torrance_brdf_shader : gl::shader {
  using this_type = cook_torrance_brdf_shader;
  static constexpr std::string_view vertex_shader_source =
      "#version 330 core\n"
      "in vec3 position; // input vertex position from mesh\n"
      "in vec3 normal;   // input vertex normal from mesh\n"
      "in float scalar;   // scalar per vertex\n"
      "\n"
      "uniform mat4 view_matrix; //camera look at matrix\n"
      "uniform mat4 projection_matrix; //camera projection matrix\n"
      "uniform mat4 model_matrix; // mesh transformation\n"
      "uniform mat4 normal_matrix; // transposed inverse of model_matrix\n"
      "\n"
      "out vec3 wfn; // output fragment normal of vertex in world space\n"
      "out vec3 view_vert_pos; // output 3D position in view space\n"
      "out float interp_scalar;\n"
      "\n"
      "void main(){\n"
      "  wfn = (view_matrix * normal_matrix * vec4(normal, 0)).xyz;\n"
      "  vec4 view_vert_pos4 = view_matrix * model_matrix * vec4(position, 1.0);\n"
      "  view_vert_pos = vec3(view_vert_pos4) / view_vert_pos4.w;\n"
      "  interp_scalar = scalar;\n"
      "  gl_Position = projection_matrix * view_vert_pos4;\n"
      "}\n";
  //============================================================================
  static constexpr std::string_view fragment_shader_source =
      "#version 330 core\n"
      "out vec4 out_color;\n"
      "\n"
      "in vec3 wfn; // fragment normal of pixel in world space (interpolated)\n"
      "in vec3 view_vert_pos; // fragment vertex position in view space\n"
      "                  // (interpolated)\n"
      "in float interp_scalar;\n"
      "\n"
      "uniform int lighting_enabled;\n"
      "uniform vec3 solid_base_color; // albedo for dielectrics or F0 for\n"
      "                               // metals\n"
      "uniform float roughness;\n"
      "uniform float metallic; // metallic parameter, 0.0 for dielectrics,\n"
      "                        //                     1.0 for metals\n"
      "uniform float reflectance; // Fresnel reflectance for dielectrics in\n"
      "                           // the range [0.0, 1.0]\n"
      "uniform vec4 light_color; // color of light\n"
      "uniform float irradi_perp; // irradiance in perpendicular direction\n"
      "uniform sampler1D color_scale;\n"
      "uniform float min_scalar;\n"
      "uniform float max_scalar;\n"
      "uniform int invert_scale;\n"
      "uniform int use_solid_base_color;\n"
      "\n"
      "vec3 rgb2lin(vec3 rgb) { // sRGB to linear approximation\n"
      "  return pow(rgb, vec3(2.2));\n"
      "}\n"
      "\n"
      "vec3 lin2rgb(vec3 lin) { // linear to sRGB approximation\n"
      "  return pow(lin, vec3(1.0 / 2.2));\n"
      "}\n"
      "\n"
      "#define RECIPROCAL_PI 0.3183098861837907\n"
      "#define RECIPROCAL_2PI 0.15915494309189535\n"
      "\n"
      "float roughness_to_shininess( const in float roughness ) {\n"
      "  return pow(1000.0, 1.0-roughness);\n"
      "}\n"
      "\n"
      "vec3 fresnel_schlick(float cos_theta, vec3 F0) {\n"
      "  return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0);\n"
      "}\n"
      "\n"
      "float fresnel_schlick90(float cos_theta, float F0, float F90) {\n"
      "  return F0 + (F90 - F0) * pow(1.0 - cos_theta, 5.0);\n"
      "} \n"
      "\n"
      "float D_GGX(float NoH, float roughness) {\n"
      "  float alpha = roughness * roughness;\n"
      "  float alpha2 = alpha * alpha;\n"
      "  float NoH2 = NoH * NoH;\n"
      "  float b = (NoH2 * (alpha2 - 1.0) + 1.0);\n"
      "  return alpha2 * RECIPROCAL_PI / (b * b);\n"
      "}\n"
      "\n"
      "float G1_GGX_Schlick(float NoV, float roughness) {\n"
      "  float alpha = roughness * roughness;\n"
      "  float k = alpha / 2.0;\n"
      "  return max(NoV, 0.001) / (NoV * (1.0 - k) + k);\n"
      "}\n"
      "\n"
      "float G_Smith(float NoV, float NoL, float roughness) {\n"
      "  return G1_GGX_Schlick(NoL, roughness) *\n"
      "         G1_GGX_Schlick(NoV, roughness);\n"
      "}\n"
      "\n"
      "float disney_diffuse_factor(float NoV, float NoL,\n"
      "                            float VoH, float roughness) {\n"
      "  float alpha = roughness * roughness;\n"
      "  float F90 = 0.5 + 2.0 * alpha * VoH * VoH;\n"
      "  float F_in = fresnel_schlick90(NoL, 1.0, F90);\n"
      "  float F_out = fresnel_schlick90(NoV, 1.0, F90);\n"
      "  return F_in * F_out;\n"
      "}\n"
      "\n"
      "vec3 brdf_microfacet(in vec3 L, in vec3 V, in vec3 N,\n"
      "                    in float metallic, in float roughness,\n"
      "                    in vec3 base_color, in float reflectance) {\n"
      "  vec3 H = normalize(V + L);\n"
      "  \n"
      "  float NoV = clamp(dot(N, V), 0.0, 1.0);\n"
      "  float NoL = clamp(abs(dot(N, L)), 0.0, 1.0);\n"
      "  float NoH = clamp(dot(N, H), 0.0, 1.0);\n"
      "  float VoH = clamp(dot(V, H), 0.0, 1.0);\n"
      "  \n"
      "  vec3 f0 = vec3(0.16 * (reflectance * reflectance));\n"
      "  f0 = mix(f0, base_color, metallic);\n"
      "  \n"
      "  vec3 F = fresnel_schlick(VoH, f0);\n"
      "  float D = D_GGX(NoH, roughness);\n"
      "  float G = G_Smith(NoV, NoL, roughness);\n"
      "  \n"
      "  vec3 spec = (F * D * G) / (4.0 * max(NoV, 0.001) * max(NoL, 0.001));\n"
      "  \n"
      "  vec3 rhoD = base_color;\n"
      "  \n"
      "  // optionally\n"
      "  rhoD *= vec3(1.0) - F;\n"
      "  //rhoD *= disney_diffuse_factor(NoV, NoL, VoH, roughness);\n"
      "  \n"
      "  rhoD *= (1.0 - metallic);\n"
      "  \n"
      "  vec3 diff = rhoD * RECIPROCAL_PI;\n"
      "  \n"
      "  return diff + spec;\n"
      "}\n"
      "\n"
      "void main() {\n"
      "  vec3 base_color = vec3(0,0,0);\n"
      "  if (use_solid_base_color == 1) {\n"
      "    base_color = solid_base_color;\n"
      "  } else {\n"
      "    if (isnan(interp_scalar)) {\n"
      "      base_color = vec3(1,0,0);\n"
      "    } else {\n"
      "      float norm_scalar = clamp((interp_scalar - min_scalar) / (max_scalar - min_scalar), 0, 1);\n"
      "      if (invert_scale == 1) {\n"
      "        norm_scalar = 1 - norm_scalar;\n"
      "      }\n"
      "      base_color = texture(color_scale, norm_scalar).rgb;\n"
      "    }\n"
      "  }\n"
      "  if (lighting_enabled == 0) {\n"
      "    out_color.rgb = base_color;\n"
      "    return;\n"
      "  }"
      "  vec3 light_dir = normalize(- view_vert_pos); // towards light\n"
      "  vec3 view_dir = normalize(-view_vert_pos);\n"
      "  vec3 n = normalize(wfn);\n"
      "  \n"
      "  //vec3 radiance = rgb2lin(emission.rgb);\n"
      "  vec3 radiance = rgb2lin(vec3(0,0,0));\n"
      "  \n"
      "  //float irradiance = max(dot(light_dir, n), 0) * irradi_perp; \n"
      "  float irradiance = abs(dot(light_dir, n)) * irradi_perp; \n"
      "  if(irradiance > 0.0) { // if receives light\n"
      "    vec3 brdf = brdf_microfacet(light_dir, view_dir, n, metallic,\n"
      "                                roughness, base_color, reflectance);\n"
      "    // irradiance contribution from directional light\n"
      "    radiance += brdf * irradiance * light_color.rgb;\n"
      "   }\n"
      "\n"
      "  out_color.rgb = lin2rgb(radiance);\n"
      "  out_color.a = 1.0;\n"
      "}\n";
  //============================================================================
  private:cook_torrance_brdf_shader() {
    add_stage<gl::vertexshader>(gl::shadersource{vertex_shader_source});
    add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader_source});
    create();
    set_view_matrix(Mat4<GLfloat>::eye());
    set_projection_matrix(Mat4<GLfloat>::eye());
    set_model_matrix(Mat4<GLfloat>::eye());
    set_solid_base_color(Vec3<GLfloat>::ones());
    set_reflectance(0.5);
    set_roughness(0.5);
    set_metallic(0);
    set_light_color(Vec4<GLfloat>::ones());
    set_irradi_perp(10);
    set_uniform("color_scale", 0);
    set_min(0);
    set_max(1);
    invert_scale(false);
    enable_lighting(true);
   }

  public:
   //--------------------------------------------------------------------------
   static auto get() -> auto& {
     static auto s = this_type{};
     return s;
   }
   auto set_view_matrix(Mat4<GLfloat> const& V) -> void {
     set_uniform("view_matrix", V);
   }
   auto set_projection_matrix(Mat4<GLfloat> const& P) -> void {
     set_uniform("projection_matrix", P);
   }
   auto set_model_matrix(Mat4<GLfloat> const& M) -> void {
     set_uniform("model_matrix", M);
     set_uniform("normal_matrix", Mat4<GLfloat>(transposed(*inv(M))));
   }
   auto set_solid_base_color(Vec3<GLfloat> const& b) -> void {
     set_uniform("solid_base_color", b);
   }
   auto set_roughness(GLfloat const r) -> void { set_uniform("roughness", r); }
   auto set_metallic(GLfloat const m) -> void { set_uniform("metallic", m); }
   auto set_reflectance(GLfloat const r) -> void {
     set_uniform("reflectance", r);
   }
   auto set_light_color(Vec4<GLfloat> const& l) -> void {
     set_uniform("light_color", l);
   }
   auto set_irradi_perp(GLfloat const i) -> void {
     set_uniform("irradi_perp", i);
   }
   //--------------------------------------------------------------------------
   auto set_min(GLfloat const min) -> void { set_uniform("min_scalar", min); }
   auto set_max(GLfloat const max) -> void { set_uniform("max_scalar", max); }
   //--------------------------------------------------------------------------
   auto invert_scale(bool const invert) -> void {
     set_uniform("invert_scale", invert ? 1 : 0);
   }
   //--------------------------------------------------------------------------
   auto use_solid_base_color(bool const use) -> void {
     set_uniform("use_solid_base_color", use ? 1 : 0);
   }
   //--------------------------------------------------------------------------
   auto enable_lighting(bool const en) -> void {
     set_uniform("lighting_enabled", en ? 1 : 0);
   }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
