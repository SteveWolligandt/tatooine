#ifndef TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_RENDERER_H
#define TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_RENDERER_H
//==============================================================================
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/netcdf.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct autonomous_particles_renderer
    : renderable<autonomous_particles_renderer> {
  struct shader : yavin::shader {
    shader() {
      add_stage<yavin::vertexshader>(
          "/home/steve/libs/tatooine2/flowexplorer/shaders/"
          "autonomous_particles_renderer.vert");
      add_stage<yavin::geometryshader>(
          "/home/steve/libs/tatooine2/flowexplorer/shaders/"
          "autonomous_particles_renderer.geom");
      add_stage<yavin::fragmentshader>(
          "/home/steve/libs/tatooine2/flowexplorer/shaders/"
          "autonomous_particles_renderer.frag");
      create();
    }
    //------------------------------------------------------------------------------
    void set_view_projection_matrix(mat4f const& A) {
      set_uniform_mat4("view_projection_matrix", A.data_ptr());
    }
  };
  //============================================================================
  yavin::indexeddata<vec2f, vec2f, vec2f> m_gpu_Ss;
  shader m_shader;
  //----------------------------------------------------------------------------
  autonomous_particles_renderer(flowexplorer::scene& s)
      : renderable<autonomous_particles_renderer>{
            "Autonomous Particles Renderer", s} {}
  //----------------------------------------------------------------------------
  void render(mat<float, 4, 4> const& P,
              mat<float, 4, 4> const& V) override {
    m_shader.bind();
    m_shader.set_view_projection_matrix(P * V);
    m_gpu_Ss.draw_points();
  }
  //----------------------------------------------------------------------------
  void update(const std::chrono::duration<double>& dt) override {}
  //----------------------------------------------------------------------------
  void draw_properties() override {
    if (ImGui::Button("load")) {
      load_data();
    }
  }
  //----------------------------------------------------------------------------
  void load_data() {
    netcdf::file f_in{
        "/home/steve/libs/tatooine2/build/autonomous_particles/"
        "dg_grid_advected.nc",
        netCDF::NcFile::read};
    auto var = f_in.variable<float>("transformations");
    m_gpu_Ss.vertexbuffer().resize(var.size(0));
    m_gpu_Ss.indexbuffer().resize(var.size(0));
    {
      auto                      vbo_map = m_gpu_Ss.vertexbuffer().wmap();
      std::vector<size_t>       is{0, 0, 0};
      std::vector<size_t> const cnt{1, 2, 3};
      for (; is.front() < var.size(0); ++is.front()) {
        auto const data_in = var.read_chunk(is, cnt);
        vbo_map[is.front()] =
            yavin::tuple{vec2f{data_in(0, 0, 0), data_in(0, 1, 0)},
                         vec2f{data_in(0, 0, 1), data_in(0, 1, 1)},
                         vec2f{data_in(0, 0, 2), data_in(0, 1, 2)}};
      }
    }
    auto ibo_map = m_gpu_Ss.indexbuffer().wmap();
    for (size_t i = 0; i < var.size(0); ++i) {
      ibo_map[i] = i;
    }
  }
  //----------------------------------------------------------------------------
  bool is_transparent() const override { return false; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
REGISTER_NODE(tatooine::flowexplorer::nodes::autonomous_particles_renderer)
#endif
