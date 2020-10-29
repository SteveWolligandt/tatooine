#ifndef TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_RENDERER_H
#define TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_RENDERER_H
//==============================================================================
#include <tatooine/flowexplorer/renderable.h>
#include <yavin/glfunctions.h>
#include <yavin/vertexbuffer.h>

#include <memory>
#include <thread>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct autonomous_particles_renderer
    : renderable<autonomous_particles_renderer> {
  struct shader : yavin::shader {
    shader();
    //------------------------------------------------------------------------------
    void set_view_projection_matrix(mat4f const& A);
  };
  //============================================================================
  yavin::vertexbuffer<vec2f, vec2f, vec2f> m_gpu_Ss;
  shader                                   m_shader;
  std::unique_ptr<std::thread>             m_reading_thread;
  bool                                     m_currently_reading = false;
  std::mutex                               m_gpu_data_mutex;
  //----------------------------------------------------------------------------
  autonomous_particles_renderer(flowexplorer::scene& s);
  virtual ~autonomous_particles_renderer();
  //----------------------------------------------------------------------------
  void render(mat4f const& P, mat4f const& V) override;
  //----------------------------------------------------------------------------
  void draw_properties() override;
  //----------------------------------------------------------------------------
  void load_data();
  //----------------------------------------------------------------------------
  bool is_transparent() const override;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
REGISTER_NODE(tatooine::flowexplorer::nodes::autonomous_particles_renderer)
#endif
