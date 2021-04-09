#ifndef TATOOINE_FLOWEXPLORER_NODES_PARALLEL_VECTORS_H
#define TATOOINE_FLOWEXPLORER_NODES_PARALLEL_VECTORS_H
//==============================================================================
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/line.h>
#include <yavin/indexeddata.h>
#include <tatooine/rendering/yavin_interop.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct parallel_vectors : renderable<parallel_vectors> {
  ui::input_pin&               v_pin;
  ui::input_pin&               w_pin;
  ui::input_pin&               grid_pin;
  std::vector<line<real_t, 3>> m_lines;
  yavin::indexeddata<vec3f> m_geometry;
  //============================================================================
  parallel_vectors(flowexplorer::scene& s);
  ~parallel_vectors() = default;
  //============================================================================
  auto render(mat4f const& P, mat4f const& V) -> void override;
  auto on_pin_connected(ui::input_pin&, ui::output_pin&) -> void override;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::parallel_vectors)
#endif
