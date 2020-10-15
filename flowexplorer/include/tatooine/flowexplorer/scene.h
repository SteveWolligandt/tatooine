#ifndef TATOOINE_FLOWEXPLORER_SCENE_H
#define TATOOINE_FLOWEXPLORER_SCENE_H
//==============================================================================
#include <tatooine/flowexplorer/link_info.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/rendering/camera_controller.h>

#include <memory>
#include <vector>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct scene {
  std::vector<std::unique_ptr<base_renderable>> m_renderables;
  std::vector<std::unique_ptr<ui::base_node>>   m_nodes;
  ax::NodeEditor::EditorContext* m_node_editor_context = nullptr;
  ImVector<link_info> m_links;  // List of live links. It is dynamic unless you
                                // want to create read-only view over nodes.
  rendering::camera_controller<float>* m_cam;
  int m_next_link = 100;
  //============================================================================
  scene(rendering::camera_controller<float>& ctrl);
  ~scene();
  //============================================================================
  void render(std::chrono::duration<double> const& dt);
  auto find_node(ax::NodeEditor::NodeId id) -> base_renderable*;
  //----------------------------------------------------------------------------
  auto find_pin(ax::NodeEditor::PinId id) -> ui::pin*;
  //----------------------------------------------------------------------------
  void node_creators();
  //----------------------------------------------------------------------------
  void draw_nodes();
  //----------------------------------------------------------------------------
  void draw_links();
  //----------------------------------------------------------------------------
  void create_link();
  //----------------------------------------------------------------------------
  void remove_link();
  //----------------------------------------------------------------------------
  void draw_node_editor(size_t const pos_x, size_t const pos_y,
                        size_t const width, size_t const height, 
                        bool show);
  //----------------------------------------------------------------------------
  void write(std::string const& filepath) const;
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
