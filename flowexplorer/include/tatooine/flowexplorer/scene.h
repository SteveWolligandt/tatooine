#ifndef TATOOINE_FLOWEXPLORER_SCENE_H
#define TATOOINE_FLOWEXPLORER_SCENE_H
//==============================================================================
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/flowexplorer/ui/link.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/rendering/camera_controller.h>

#include <memory>
#include <vector>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct scene {
  std::vector<std::unique_ptr<renderable>> m_renderables;
  std::vector<std::unique_ptr<ui::node>>   m_nodes;
  std::vector<ui::link>                    m_links;
  ax::NodeEditor::EditorContext*           m_node_editor_context = nullptr;
  rendering::camera_controller<float>*     m_cam;
  //============================================================================
  scene(rendering::camera_controller<float>& ctrl);
  ~scene();
  //============================================================================
  void render(std::chrono::duration<double> const& dt);
  auto find_node(size_t const id) -> ui::node*;
  //----------------------------------------------------------------------------
  auto find_pin(size_t const id) -> ui::pin*;
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
                        size_t const width, size_t const height, bool& show);
  //----------------------------------------------------------------------------
  void write(std::string const& filepath) const;
  void read(std::string const& filepath);
  //----------------------------------------------------------------------------
  template <typename F>
  auto do_in_context(F&& f) const {
    ax::NodeEditor::SetCurrentEditor(m_node_editor_context);
    f();
    ax::NodeEditor::SetCurrentEditor(nullptr);
  }
  void clear() {
    m_nodes.clear();
    m_renderables.clear();
    m_links.clear();
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
