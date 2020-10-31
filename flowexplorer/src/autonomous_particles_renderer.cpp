#include <tatooine/flowexplorer/nodes/autonomous_particles_renderer.h>
#include <tatooine/netcdf.h>
#include <tatooine/rendering/yavin_interop.h>
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
autonomous_particles_renderer::~autonomous_particles_renderer() {
  if (m_reading_thread) {
    m_reading_thread->join();
  };
}
//----------------------------------------------------------------------------
void autonomous_particles_renderer::render(mat4f const& P, mat4f const& V) {
  // if (!m_currently_reading) {
  if (!m_gpu_Ss.empty()) {
    //m_shader.bind();
    //m_shader.set_view_projection_matrix(P * V);
    //m_gpu_Ss.bind();
    //m_gpu_Ss.activate_attributes();
    //yavin::gl::draw_arrays(GL_POINTS, 0, m_gpu_Ss.size());
  }
  //}
}
//----------------------------------------------------------------------------
auto autonomous_particles_renderer::draw_properties() -> bool {
  ImGui::Text("number of particles: %i", (m_gpu_Ss.size()));
  if (ImGui::Button("load")) {
    load_data();
  }
  return false;
}
//----------------------------------------------------------------------------
void autonomous_particles_renderer::load_data() {
  if (!m_currently_reading) {
    m_reading_thread = std::make_unique<std::thread>([this] {
      m_currently_reading = true;
      netcdf::file f_in{
          "/home/steve/libs/tatooine2/build/autonomous_particles/"
          "dg_grid_advected.nc",
          netCDF::NcFile::read};
      auto var = f_in.variable<float>("transformations");
      std::cerr << var.size(0) << '\n';
      // std::lock_guard lock{m_gpu_data_mutex};
      m_gpu_Ss.resize(var.size(0));
      //{
      //  auto vbo_map = m_gpu_Ss.wmap();
      //  var.read(reinterpret_cast<float*>(&vbo_map.front()));
      //}
      m_currently_reading = true;
    });
  }
}
//----------------------------------------------------------------------------
bool autonomous_particles_renderer::is_transparent() const { return false; }
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
