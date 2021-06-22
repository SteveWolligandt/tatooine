#ifndef TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_EVALUATOR_H
#define TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_EVALUATOR_H
//==============================================================================
#include <tatooine/flowexplorer/nodes/autonomous_particles_flowmap.h>
#include <tatooine/flowexplorer/nodes/position.h>
#include <tatooine/flowexplorer/point_shader.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/gl/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct autonomous_particles_flowmap_evaluator
    : renderable<autonomous_particles_flowmap_evaluator> {
  static std::vector<std::string> const items;
  using gpu_vec  = vec<GLfloat, 3>;

  autonomous_particles_flowmap* m_flowmap        = nullptr;
  position<2>*                  m_x0             = nullptr;
  bool                          m_is_evaluatable = false;
  unsigned int                  m_current_item   = 0;

  vec2                        m_x1;
  gl::indexeddata<gpu_vec>    m_gpu_data;
  int                         m_pointsize = 1;
  std::array<GLfloat, 4>      m_color{0.0f, 0.0f, 0.0f, 1.0f};
  //----------------------------------------------------------------------------
  autonomous_particles_flowmap_evaluator(flowexplorer::scene& s);
  //----------------------------------------------------------------------------
  virtual ~autonomous_particles_flowmap_evaluator() = default;
  //============================================================================
  auto draw_properties() -> bool override;
  //----------------------------------------------------------------------------
  auto render(mat<GLfloat, 4, 4> const& projection_matrix,
              mat<GLfloat, 4, 4> const& view_matrix) -> void override;
  //----------------------------------------------------------------------------
  auto on_pin_connected(ui::input_pin& this_pin, ui::output_pin& other_pin)
      -> void override;
  //----------------------------------------------------------------------------
  auto on_property_changed() -> void override;
  //----------------------------------------------------------------------------
  auto evaluate() -> void;
  //----------------------------------------------------------------------------
  auto on_mouse_drag(int /*offset_x*/, int /*offset_y*/) -> bool override {
    return false;
  }
  //----------------------------------------------------------------------------
  auto set_vbo_data() -> void;
  auto create_indexed_data() -> void;
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool override { return m_color[3] < 1; }
  };
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::autonomous_particles_flowmap_evaluator,
    TATOOINE_REFLECTION_INSERT_METHOD(point_size, m_pointsize),
    TATOOINE_REFLECTION_INSERT_METHOD(color, m_color),
    TATOOINE_REFLECTION_INSERT_METHOD(current_item, m_current_item))
#endif
