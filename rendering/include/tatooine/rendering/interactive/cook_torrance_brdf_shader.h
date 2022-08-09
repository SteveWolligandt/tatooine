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
      "#version 300 core\n"
      "//precision highp float;\n"
      "in vec3 position; // input vertex position from mesh\n"
      "in vec3 normal;   // input vertex normal from mesh\n"
      "\n"
      "uniform mat4 view_matrix; //camera look at matrix\n"
      "uniform mat4 projection_matrix; //camera projection matrix\n"
      "uniform mat4 model_matrix; // mesh transformation\n"
      "uniform mat4 inv_model_matrix; // transposed inverse of model_matrix\n"
      "\n"
      "out vec3 wfn; // output fragment normal of vertex in world space\n"
      "out vec3 vert_pos; // output 3D position in world space\n"
      "\n"
      "void main(){\n"
      "  wfn = vec3(inv_model_matrix * vec4(normal, 0.0));\n"
      "  vec4 vert_pos4 = model_matrix * vec4(position, 1.0);\n"
      "  vert_pos = vec3(vert_pos4) / vert_pos4.w;\n"
      "  gl_Position = projection_matrix * view_matrix * vert_pos4;\n"
      "}\n";
  //============================================================================
  static constexpr std::string_view fragment_shader_source =
      "#version 300 core\n"
      "//precision highp float;\n"
      "out vec4 out_color;\n"
      "\n"
      "in vec3 wfn; // fragment normal of pixel in world space (interpolated)\n"
      "in vec3 vert_pos; // fragment vertex position in world space (interpolated)\n"
      "\n"
      "uniform vec3 base_color;// albedo for dielectrics or F0 for metals\n"
      "uniform float roughness; //roughness texture\n"
      "uniform float metallic; // metallic parameter, 0.0 for dielectrics, 1.0 for metals\n"
      "uniform float reflectance; // Fresnel reflectance for dielectrics in the range [0.0, 1.0]\n"
      "uniform vec4 light_color; // color of light\n"
      "uniform float irradi_perp; // irradiance in perpendicular direction\n"
      "uniform vec3 light_direction; // light direction in world space\n"
      "uniform vec3 camera_position; // camera position in world space\n"
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
      "vec3 modifiedPhongBRDF(vec3 light_dir, vec3 view_dir, vec3 normal,\n"
      "                       vec3 diffuse_color, vec3 specular_color,\n"
      "                       float shininess) {\n"
      "  vec3  color         = diffuse_color * RECIPROCAL_PI;\n"
      "  vec3  reflect_dir    = reflect(-light_dir, normal);\n"
      "  float spec_dot       = max(dot(reflect_dir, view_dir), 0.001);\n"
      "  float normalization = (shininess + 2.0) * RECIPROCAL_2PI;\n"
      "  color += pow(spec_dot, shininess) * normalization * specular_color;\n"
      "  return color;\n"
      "}\n"
      "\n"
      "float roughness_to_shininess( const in float roughness ) {\n"
      "  return pow(1000.0, 1.0-roughness);\n"
      "}\n"
      "\n"
      "// from http://www.thetenthplanet.de/archives/1180\n"
      "mat3 cotangent_frame(in vec3 N, in vec3 p, in vec2 uv) {\n"
      "  // get edge vectors of the pixel triangle\n"
      "  vec3 dp1 = dFdx( p );\n"
      "  vec3 dp2 = dFdy( p );\n"
      "  vec2 duv1 = dFdx( uv );\n"
      "  vec2 duv2 = dFdy( uv );\n"
      "\n"
      "  // solve the linear system\n"
      "  vec3 dp2perp = cross( dp2, N );\n"
      "  vec3 dp1perp = cross( N, dp1 );\n"
      "  vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;\n"
      "  vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;\n"
      "\n"
      "  // construct a scale-invariant frame \n"
      "  float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );\n"
      "  return mat3( T * invmax, B * invmax, N );\n"
      " }\n"
      "\n"
      "vec3 apply_normal_map(in vec3 normal, in vec3 viewVec,\n"
      "                    in vec2 texcoord) {\n"
      "  vec3 highResNormal = texture(normalTexture, texcoord).xyz;\n"
      "  highResNormal = normalize(highResNormal * 2.0 - 1.0);\n"
      "  mat3 TBN = cotangent_frame(normal, -viewVec, texcoord);\n"
      "  return normalize(TBN * highResNormal);\n"
      "}\n"
      "\n"
      "vec3 fresnel_schlick(float cosTheta, vec3 F0) {\n"
      "  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);\n"
      "}\n"
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
      "  return G1_GGX_Schlick(NoL, roughness) * G1_GGX_Schlick(NoV, roughness);\n"
      "}\n"
      "\n"
      "float fresnel_schlick90(float cosTheta, float F0, float F90) {\n"
      "  return F0 + (F90 - F0) * pow(1.0 - cosTheta, 5.0);\n"
      "} \n"
      "\n"
      "float disney_diffuse_factor(float NoV, float NoL, float VoH, float roughness) {\n"
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
      "  float NoL = clamp(dot(N, L), 0.0, 1.0);\n"
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
      "  // rhoD *= disney_diffuse_factor(NoV, NoL, VoH, roughness);\n"
      "  \n"
      "  rhoD *= (1.0 - metallic);\n"
      "  \n"
      "  vec3 diff = rhoD * RECIPROCAL_PI;\n"
      "  \n"
      "  return diff + spec;\n"
      "}\n"
      "\n"
      "void main() {\n"
      "  vec3 light_dir = normalize(-light_direction); // towards light\n"
      "  vec3 view_dir = normalize(camera_position - vert_pos);\n"
      "  vec3 n = normalize(wfn);\n"
      "\n"
      "  vec3 radiance = rgb2lin(emission.rgb);\n"
      "  \n"
      "  float irradiance = max(dot(light_dir, n), 0.0) * irradi_perp; \n"
      "  if(irradiance > 0.0) { // if receives light\n"
      "    vec3 brdf = brdf_microfacet(light_dir, view_dir, n, metallic, roughness, base_color, reflectance);\n"
      "    // irradiance contribution from directional light\n"
      "    radiance += brdf * irradiance * light_color.rgb;\n"
      "   }\n"
      "\n"
      "  out_color.rgb = lin2rgb(radiance);\n"
      "  out_color.a = 1.0;\n"
      "}\n";
  //============================================================================
  private:cook_torrance_brdf_shader() {
    add_stage<gl::vertexshader>(vertex_shader_source);
    add_stage<gl::vertexshader>(fragment_shader_source);
    create();
    set_view_matrix(Mat4<GLfloat>::eye());
    set_projection_matrix(Mat4<GLfloat>::eye());
    set_model_matrix(Mat4<GLfloat>::eye());
    set_base_color(Vec3<GLfloat>::ones());
    set_roughness(0);
    set_metallic(0);
    set_reflectance(0.5);
    set_light_color(Vec4<GLfloat>::ones());
    set_irradi_perp(10);
    set_light_direction(Vec3<GLfloat>{0, -1, 0});
    set_camera_position(Vec3<GLfloat>{0, 0, 0});
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
     set_uniform("inv_model_matrix", *inv(M));
   }
   auto set_base_color(Vec3<GLfloat> const& b) -> void {
     set_uniform("base_color", b);
   }
   auto set_roughness(float const r) -> void { set_uniform("roughness", r); }
   auto set_metallic(float const m) -> void { set_uniform("metallic", m); }
   auto set_reflectance(float const r) -> void {
     set_uniform("reflectance", r);
   }
   auto set_light_color(Vec4<GLfloat> const& l) -> void {
     set_uniform("light_color", l);
   }
   auto set_irradi_perp(float const i) -> void {
     set_uniform("irradi_perp", i);
   }
   auto set_light_direction(Vec3<GLfloat> const& l) -> void {
     set_uniform("light_direction", l);
   }
   auto set_camera_position(Vec3<GLfloat> const& c) -> void {
     set_uniform("camera_position", c);
   }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
