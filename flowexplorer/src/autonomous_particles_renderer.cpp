#include <tatooine/flowexplorer/nodes/autonomous_particles_renderer.h>
#include <tatooine/netcdf.h>
#include <tatooine/rendering/yavin_interop.h>
#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/window.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
autonomous_particles_renderer::shader::shader() {
  add_stage<yavin::vertexshader>(vertex_shader_path);
  add_stage<yavin::geometryshader>(geometry_shader_path);
  add_stage<yavin::fragmentshader>(fragment_shader_path);
  create();
}
//------------------------------------------------------------------------------
void autonomous_particles_renderer::shader::set_view_projection_matrix(
    mat4f const& A) {
  set_uniform_mat4("view_projection_matrix", A.data_ptr());
}
//----------------------------------------------------------------------------
autonomous_particles_renderer::autonomous_particles_renderer(
    flowexplorer::scene& s)
    : renderable<autonomous_particles_renderer>{"Autonomous Particles Renderer",
                                                s} {}
//----------------------------------------------------------------------------
void autonomous_particles_renderer::render(mat4f const& P, mat4f const& V) {
  if (m_currently_reading) {
    return;
  }
  if (!m_gpu_Ss.empty()) {
    m_shader.bind();
    m_shader.set_view_projection_matrix(P * V);
    yavin::vertexarray vao;
    vao.bind();
    m_gpu_Ss.bind();
    m_gpu_Ss.activate_attributes();
    m_gpu_Is.bind();
    vao.draw_points(m_gpu_Ss.size());
  }
}
//----------------------------------------------------------------------------
auto autonomous_particles_renderer::draw_properties() -> bool {
  ImGui::Text("number of particles: %i", (m_gpu_Ss.size()));
  if (m_currently_reading) {
    const ImU32 col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
    ImGui::Spinner("##spinner", 8, 3, col);
  } else {
    if (ImGui::Button("load")) {
      load_data();
    }
  }
  return false;
}
//----------------------------------------------------------------------------
void autonomous_particles_renderer::load_data() {
  if (m_currently_reading) {
    return;
  }
  m_currently_reading = true;
  auto run = [node = this] {
    netcdf::file f_in{
        "/home/steve/libs/tatooine2/build/autonomous_particles/"
        "dg_grid_advected.nc",
        netCDF::NcFile::read};
    auto var = f_in.variable<float>("transformations");
    std::lock_guard lock{node->m_gpu_data_mutex};
    node->m_gpu_Ss.resize(var.size(0));
    {
      auto vbo_map = node->m_gpu_Ss.wmap();
      var.read(reinterpret_cast<float*>(&vbo_map.front()));
    }
    node->m_gpu_Is.resize(var.size(0));
    {
      auto ibo_map = node->m_gpu_Is.wmap();
      std::iota(begin(ibo_map), end(ibo_map), (unsigned int)(0));
    }
    node->m_currently_reading = false;
  };
  //run();
  this->scene().window().do_async(run);
}
//----------------------------------------------------------------------------
bool autonomous_particles_renderer::is_transparent() const { return false; }
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
