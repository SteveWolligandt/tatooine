#ifndef TATOOINE_FLOWEXPLORER_NODES_RANDOM_POINTS_H
#define TATOOINE_FLOWEXPLORER_NODES_RANDOM_POINTS_H
//==============================================================================
#include <tatooine/flowexplorer/renderable.h>
#include <yavin/indexeddata.h>
#include <tatooine/rendering/yavin_interop.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct random_points : renderable<random_points>{
  std::vector<vec2>         m_points2d;
  std::vector<vec3>         m_points3d;
  yavin::indexeddata<vec3f> m_points_gpu;
  ui::input_pin&            m_input;
  //============================================================================
  int m_num_points = 10;
  //============================================================================
  random_points(flowexplorer::scene& s);
  virtual ~random_points() = default;
  auto render(mat4f const&, mat4f const&) -> void override;
  auto on_pin_connected(ui::input_pin& /*this_pin*/,
                        ui::output_pin& /*other_pin*/) -> void override;
  auto on_property_changed() -> void override;
  auto update_points() -> void;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::random_points,
    TATOOINE_REFLECTION_INSERT_METHOD(num_points, m_num_points))
#endif
