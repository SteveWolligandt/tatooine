#ifndef TATOOINE_FLOWEXPLORER_SCENE_H
#define TATOOINE_FLOWEXPLORER_SCENE_H
//==============================================================================
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/flowexplorer/ui/link.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/rendering/camera_controller.h>
#include <tatooine/filesystem.h>

#include <memory>
#include <vector>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window;
struct scene {
 private:
  static std::set<std::string_view>              items;
  static bool                                    items_created;
  std::list<std::unique_ptr<ui::base::node>>     m_nodes;
  std::list<std::unique_ptr<base::renderable>>   m_renderables;
  std::list<std::unique_ptr<ui::link>>           m_links;
  ax::NodeEditor::EditorContext*       m_node_editor_context = nullptr;
  rendering::camera_controller<float>* m_cam;
  flowexplorer::window*                m_window;
  bool                                 m_new_link              = false;
  ui::input_pin*                       m_new_link_start_input  = nullptr;
  ui::output_pin*                      m_new_link_start_output = nullptr;

 public:
  //============================================================================
  scene(rendering::camera_controller<float>& ctrl, flowexplorer::window* w);
  scene(rendering::camera_controller<float>& ctrl, flowexplorer::window* w,
        filesystem::path const& path);
  ~scene();
  //============================================================================
  auto links()             const -> auto const& { return m_links; }
  auto links()                   -> auto& { return m_links; }
  auto new_link()          const -> bool  { return m_new_link; }
  //------------------------------------------------------------------------------
  auto nodes()             const -> auto const& { return m_nodes; }
  auto nodes()                   -> auto& { return m_nodes; }
  //------------------------------------------------------------------------------
  auto renderables()       const -> auto const& { return m_renderables; }
  auto renderables()             -> auto&       { return m_renderables; }
  //----------------------------------------------------------------------------
  auto camera()            const -> auto const& { return *m_cam; }
  auto camera()                  -> auto&       { return *m_cam; }
  //----------------------------------------------------------------------------
  auto window()            const -> auto const& { return *m_window; }
  auto window()                  -> auto&       { return *m_window; }
  //----------------------------------------------------------------------------
  auto camera_controller() const -> auto const& { return *m_cam; }
  auto camera_controller()       -> auto&       { return *m_cam; }
  //============================================================================
  auto remove_link(ui::link const& link_to_remove) -> void;
  auto render(std::chrono::duration<double> const& dt) -> void;
  //----------------------------------------------------------------------------
  auto find_node(ax::NodeEditor::NodeId const& id) -> ui::base::node*;
  //----------------------------------------------------------------------------
  auto find_input_pin(ax::NodeEditor::PinId const& id) -> ui::input_pin*;
  auto find_output_pin(ax::NodeEditor::PinId const& id) -> ui::output_pin*;
  //----------------------------------------------------------------------------
  auto node_creators(size_t const width) -> void;
  //----------------------------------------------------------------------------
  auto draw_nodes() -> void;
  //----------------------------------------------------------------------------
  auto draw_links() -> void;
  //----------------------------------------------------------------------------
  auto link(ui::input_pin& in, ui::output_pin& out) -> ui::link&;
  //----------------------------------------------------------------------------
  auto can_create_link(ui::input_pin const& , ui::input_pin const& )
      -> bool;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto can_create_link(ui::output_pin const&, ui::output_pin const&) -> bool;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto can_create_link(ui::input_pin const&, ui::output_pin const&) -> bool;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto can_create_link(ui::output_pin const& , ui::input_pin const& )
      -> bool;
  //----------------------------------------------------------------------------
  auto can_create_new_link(ui::input_pin const&) -> bool;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto can_create_new_link(ui::output_pin const&) -> bool;
  //----------------------------------------------------------------------------
  auto query_link_creation() -> void;
  auto query_link_and_node_deletions() -> void;
  //----------------------------------------------------------------------------
  auto draw_node_editor(size_t const pos_x, size_t const pos_y,
                        size_t const width, size_t const height,
                        bool& show) -> void;
  //----------------------------------------------------------------------------
  auto write(filesystem::path const& filepath) const -> void;
  auto read(filesystem::path const& filepath) -> void;
  //----------------------------------------------------------------------------
  auto open_file(filesystem::path const& filepath) -> void;
  //----------------------------------------------------------------------------
  template <typename F>
  auto do_in_context(F&& f) const -> decltype(auto) {
    ax::NodeEditor::SetCurrentEditor(m_node_editor_context);
    if constexpr (!std::is_same_v<std::invoke_result_t<F>, void>) {
      decltype(auto) ret = f();
      ax::NodeEditor::SetCurrentEditor(nullptr);
      return ret;
    } else {
      f();
      ax::NodeEditor::SetCurrentEditor(nullptr);
    }
  }
  //----------------------------------------------------------------------------
  auto clear() -> void;
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#include <tatooine/flowexplorer/window.h>
#endif
