#include <tatooine/flowexplorer/scene.h>

#include <tatooine/flowexplorer/nodes/autonomous_particles_renderer.h>
#include <tatooine/netcdf.h>
#include <tatooine/rendering/yavin_interop.h>
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
    if (ImGui::Button("load advection")) {
      load_advection();
    }
    if (ImGui::Button("load back_calculation")) {
      load_back_calculation();
    }
  }
  return false;
}
//----------------------------------------------------------------------------
auto autonomous_particles_renderer::load_back_calculation() -> void {
  load_data(
      "/home/steve/libs/tatooine2/build/autonomous_particles/"
      "dg_grid_back_calculation.nc");
}
//----------------------------------------------------------------------------
auto autonomous_particles_renderer::load_advection() -> void {
  load_data(
      "/home/steve/libs/tatooine2/build/autonomous_particles/"
      "dg_grid_advected.nc");
}
//----------------------------------------------------------------------------
void autonomous_particles_renderer::load_data(std::string_view const& file) {
  if (m_currently_reading) {
    return;
  }
  m_currently_reading = true;
  auto run = [file, node = this] {
    netcdf::file f_in{
      std::string{file},
        netCDF::NcFile::read};
    auto var = f_in.variable<float>("transformations");
    node->m_gpu_Ss.resize(var.size(0));
    {
      auto vbo_map = node->m_gpu_Ss.wmap();
      auto ptr = reinterpret_cast<float*>(&vbo_map.front());
      size_t const chunk_size = 1000;
      for (size_t i = 0; i < var.size(0); i += chunk_size) {
        size_t cnt = chunk_size;
        if (i + chunk_size >= var.size(0)) {
          cnt = var.size(0) - i - 1;
        }
        var.read_chunk(std::vector<size_t>{i, 0, 0},
                       std::vector<size_t>{cnt, 2, 3}, ptr);
        ptr += cnt * 6;
      }
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
