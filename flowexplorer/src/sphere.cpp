#include <tatooine/flowexplorer/scene.h>
//
#include <tatooine/flowexplorer/nodes/sphere.h>
#include <tatooine/rendering/yavin_interop.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
sphere_shader::sphere_shader() {
  add_stage<yavin::vertexshader>(vertex_shader_path);
  add_stage<yavin::fragmentshader>(fragment_shader_path);
  create();
}
//------------------------------------------------------------------------------
auto sphere_shader::set_color(std::array<GLfloat, 4> const& col) -> void {
  set_uniform("color", col[0], col[1], col[2], col[3]);
}
//------------------------------------------------------------------------------
auto sphere_shader::set_projection_matrix(mat4f const& P) -> void {
  set_uniform_mat4("projection", P.data_ptr());
}
//------------------------------------------------------------------------------
auto sphere_shader::set_modelview_matrix(mat4f const& MV) -> void {
  set_uniform_mat4("modelview", MV.data_ptr());
}
//==============================================================================
auto sphere::shader() -> sphere_shader& {
  static sphere_shader s;
  return s;
}
//------------------------------------------------------------------------------
sphere::sphere(flowexplorer::scene& s)
    : renderable<sphere>{"Sphere", s,
                         *dynamic_cast<geometry::sphere<real_t, 3>*>(this)} {
  insert_input_pin_property_link(insert_input_pin<real_t>(""), radius());
  insert_input_pin_property_link(insert_input_pin<vec3>(""), center());

  auto d = discretize(*this, 3);
  for (auto v : d.vertices()) {
    m_gpu_data.vertexbuffer().push_back(vec3f{static_cast<float>(d[v](0)),
                                              static_cast<float>(d[v](1)),
                                              static_cast<float>(d[v](2))});
  }
  for (auto f : d.faces()) {
    auto const& [v0, v1, v2] = d[f];
    m_gpu_data.indexbuffer().push_back(v0.i);
    m_gpu_data.indexbuffer().push_back(v1.i);
    m_gpu_data.indexbuffer().push_back(v2.i);
  }
}
//------------------------------------------------------------------------------
auto sphere::render(mat4f const& P, mat4f const& V) -> void {
  auto MV = V;
  float r  = radius();
  MV       = MV * mat4f{
                  {1.0f, 0.0f, 0.0f, (float)center()(0)},
                  {0.0f, 1.0f, 0.0f, (float)center()(1)},
                  {0.0f, 0.0f, 1.0f, (float)center()(2)},
                  {0.0f, 0.0f, 0.0f, 1.0f}};
  MV       = MV * diag(vec4f{r, r, r, 1});
  shader().set_projection_matrix(P);
  shader().set_modelview_matrix(MV);
  shader().set_color(m_color);
  shader().bind();
  m_gpu_data.draw_triangles();
}
//------------------------------------------------------------------------------
auto sphere::is_transparent() const -> bool { return m_color[3] < 1; }
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
