#ifndef TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_EVALUATOR_H
#define TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_EVALUATOR_H
//==============================================================================
#include <tatooine/flowexplorer/nodes/autonomous_particles_flowmap.h>
#include <tatooine/flowexplorer/nodes/position.h>
#include <tatooine/flowexplorer/point_shader.h>
#include <tatooine/flowexplorer/renderable.h>
#include <yavin/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct autonomous_particles_flowmap_evaluator
    : renderable<autonomous_particles_flowmap_evaluator> {
  static std::vector<std::string> const items;

  yavin::indexeddata<vec3f>             m_gpu_data;
  point_shader                          m_shader;
  position<2>*                          m_x0      = nullptr;
  autonomous_particles_flowmap*         m_flowmap = nullptr;
  std::array<GLfloat, 4>                m_color;
  int                                   m_point_size     = 1;
  bool                                  m_is_evaluatable = false;
  unsigned int                          m_current_item   = 0;
  //----------------------------------------------------------------------------
  autonomous_particles_flowmap_evaluator(flowexplorer::scene& s);
  //----------------------------------------------------------------------------
  virtual ~autonomous_particles_flowmap_evaluator() = default;
  //============================================================================
  auto render(mat4f const& projection_matrix, mat4f const& view_matrix)
      -> void override;
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool override;
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool override;
  //----------------------------------------------------------------------------
  auto on_pin_connected(ui::input_pin& this_pin, ui::output_pin& other_pin)
      -> void override;
  //----------------------------------------------------------------------------
  auto on_property_changed() -> void override;
  //----------------------------------------------------------------------------
  auto evaluate() -> void;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::autonomous_particles_flowmap_evaluator,
    TATOOINE_REFLECTION_INSERT_METHOD(point_size, m_point_size),
    TATOOINE_REFLECTION_INSERT_METHOD(color, m_color),
    TATOOINE_REFLECTION_INSERT_METHOD(current_item, m_current_item))
#endif
