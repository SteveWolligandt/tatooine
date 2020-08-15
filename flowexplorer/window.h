#ifndef TATOOINE_FLOWEXPLORER_WINDOW_H
#define TATOOINE_FLOWEXPLORER_WINDOW_H
//==============================================================================
#include "imgui-node-editor/imgui_node_editor.h"
#include <tatooine/boundingbox.h>
#include <tatooine/gpu/first_person_window.h>
#include <tatooine/interpolation.h>

#include "boundingbox.h"
#include "grid.h"
#include "pathlines_boundingbox.h"
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window : first_person_window {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  bool                                     show_nodes_gui;
  std::vector<std::unique_ptr<renderable>> m_renderables;

  int mouse_x, mouse_y;

  struct LinkInfo {
    ax::NodeEditor::LinkId id;
    ax::NodeEditor::PinId  input_id;
    ax::NodeEditor::PinId  output_id;
  };
  ax::NodeEditor::EditorContext* m_node_editor_context = nullptr;
  bool m_first_frame = true;  // Flag set for first frame only, some action need
                             // to be executed once.
  ImVector<LinkInfo> m_links;  // List of live links. It is dynamic unless you
                               // want to create read-only view over nodes.
  int m_next_link = 100;

  //----------------------------------------------------------------------------
  // ctor
  //----------------------------------------------------------------------------
  template <typename V, typename VReal, size_t N>
  window(const vectorfield<V, VReal, N, N>& v)
      : show_nodes_gui{true} {
    m_node_editor_context = ax::NodeEditor::CreateEditor();
    add_key_pressed_event([&](auto k) {
      if (k == yavin::KEY_SPACE) {
        show_nodes_gui = !show_nodes_gui;
        std::cerr << show_nodes_gui << '\n';
      }
      if (k == yavin::KEY_SPACE) {
        // try {
        //  shader = std::make_unique<gpu::line_shader>(
        //    line_color[0], line_color[1], line_color[2], contour_color[0],
        //    contour_color[1], contour_color[2], line_width, contour_width,
        //    ambient_factor, diffuse_factor, specular_factor, shininess);
        //} catch (std::exception& e) { std::cerr << e.what() << '\n'; }
      }
    });
    add_mouse_motion_event([&](int x, int y) {
      mouse_x = x;
      mouse_y = y;
    });
    add_button_released_event([&](auto b) {
      // if (b == yavin::BUTTON_LEFT) {
      //  auto       r  = cast_ray(mouse_x, mouse_y);
      //  const auto x0 = r(0.5);
      //  if (v.in_domain(x0, 0)) {
      //    lines.push_back(integrator.integrate(v, x0, 0, btau, ftau));
      //    line_renderers.push_back(gpu::upload(lines.back()));
      //  }
      //}
    });

    start(v);
  }
  //----------------------------------------------------------------------------
  ~window() {
    ax::NodeEditor::DestroyEditor(m_node_editor_context);
  }
  //============================================================================
  template <typename V, typename VReal, size_t N>
  void start(const vectorfield<V, VReal, N, N>& v) {
    render_loop([&](const auto& dt) {
      // if (shader->files_changed()) {
      //  try {
      //    shader = std::make_unique<gpu::line_shader>(
      //        line_color[0], line_color[1], line_color[2], contour_color[0],
      //        contour_color[1], contour_color[2], line_width, contour_width,
      //        ambient_factor, diffuse_factor, specular_factor, shininess);
      //  } catch (std::exception& e) { std::cerr << e.what() << '\n'; }
      //}
      yavin::gl::clear_color(255, 255, 255, 255);
      yavin::clear_color_depth_buffer();
      for (auto& r : m_renderables) {
        if (r->is_active()) {
          r->update(dt);
        }
      }
      for (auto& r : m_renderables) {
        if (r->is_active()) {
          r->render(projection_matrix(), view_matrix());
        }
      }
      render_ui(v);
    });
  }
  //----------------------------------------------------------------------------
  void render_node_editor() {
    namespace ed = ax::NodeEditor;
    ImGui::GetStyle().WindowRounding =
        0.0f;  // <- Set this on init or use ImGui::PushStyleVar()
    ImGui::GetStyle().ChildRounding     = 0.0f;
    ImGui::GetStyle().FrameRounding     = 0.0f;
    ImGui::GetStyle().GrabRounding      = 0.0f;
    ImGui::GetStyle().PopupRounding     = 0.0f;
    ImGui::GetStyle().ScrollbarRounding = 0.0f;
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(this->width() / 3, this->height()));
    bool _b=true;
    ImGui::Begin("Node Editor", &_b,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoTitleBar);
    ed::SetCurrentEditor(m_node_editor_context);

    if (ImGui::Button("add bounding box")) {
      m_renderables.emplace_back(
          new boundingbox{*this, vec{-1.0, -1.0, -1.0}, vec{1.0, 1.0, 1.0}});
    }
    // Start interaction with editor.
    ed::Begin("My Editor", ImVec2(0.0, 0.0f));

    int unique_id = 1;

    //
    // 1) Commit known data to editor
    //

    size_t i = 0;
    for (auto& r : m_renderables) {
      ImGui::PushID(i++);
      r->draw_ui();
      ImGui::PopID();
    }

    // Submit Links
    for (auto& link_info : m_links) {
      ed::Link(link_info.id, link_info.input_id, link_info.output_id);
    }

    //
    // 2) Handle interactions
    //

    // Handle creation action, returns true if editor want to create new
    // object (node or link)
    if (ed::BeginCreate()) {
      ed::PinId inputPinId, outputPinId;
      if (ed::QueryNewLink(&inputPinId, &outputPinId)) {
        // QueryNewLink returns true if editor want to create new link between
        // pins.
        //
        // Link can be created only for two valid pins, it is up to you to
        // validate if connection make sense. Editor is happy to make any.
        //
        // Link always goes from input to output. User may choose to drag
        // link from output pin or input pin. This determine which pin ids
        // are valid and which are not:
        //   * input valid, output invalid - user started to drag new ling
        //   from input pin
        //   * input invalid, output valid - user started to drag new ling
        //   from output pin
        //   * input valid, output valid   - user dragged link over other pin,
        //   can be validated

        if (inputPinId && outputPinId) {  // both are valid, let's accept link
          // ed::AcceptNewItem() return true when user release mouse button.
          if (ed::AcceptNewItem()) {
            // Since we accepted new link, lets add one to our list of links.
            m_links.push_back(
                {ed::LinkId(m_next_link++), inputPinId, outputPinId});

            // Draw new link.
            ed::Link(m_links.back().id, m_links.back().input_id,
                     m_links.back().output_id);
          }

          // You may choose to reject connection between these nodes
          // by calling ed::RejectNewItem(). This will allow editor to give
          // visual feedback by changing link thickness and color.
        }
      }
    }
    ed::EndCreate();  // Wraps up object creation action handling.

    // Handle deletion action
    if (ed::BeginDelete()) {
      // There may be many links marked for deletion, let's loop over them.
      ed::LinkId deletedLinkId;
      while (ed::QueryDeletedLink(&deletedLinkId)) {
        // If you agree that link can be deleted, accept deletion.
        if (ed::AcceptDeletedItem()) {
          // Then remove link from your data.
          for (auto& link : m_links) {
            if (link.id == deletedLinkId) {
              m_links.erase(&link);
              break;
            }
          }
        }

        // You may reject link deletion by calling:
        // ed::RejectDeletedItem();
      }
    }
    ed::EndDelete();  // Wrap up deletion action

    // End of interaction with editor.
    ed::End();

    if (m_first_frame) {
      ed::NavigateToContent(0.0f);
    }
    ImGui::End();

    ed::SetCurrentEditor(nullptr);

    m_first_frame = false;
  }
  //----------------------------------------------------------------------------
  //template <typename V, typename VReal, size_t N>
  //void render_gui(const vectorfield<V, VReal, N, N>& v) {
  //  ImGui::Begin("GUI", &show_gui);
  //  if (ImGui::Button("add bounding box")) {
  //    m_renderables.emplace_back(
  //        new boundingbox{*this, vec{-1.0, -1.0, -1.0}, vec{1.0, 1.0, 1.0}});
  //  }
  //  if (ImGui::Button("add grid")) {
  //    m_renderables.emplace_back(new grid{*this, linspace{-1.0, 1.0, 3},
  //                                        linspace{-1.0, 1.0, 3},
  //                                        linspace{-1.0, 1.0, 3}});
  //  }
  //  if (ImGui::Button("add pathline bounding box")) {
  //    m_renderables.emplace_back(new pathlines_boundingbox{
  //        *this, v, vec{-1.0, -1.0, -1.0}, vec{1.0, 1.0, 1.0}});
  //  }
  //  ImGui::End();
  //}
  //----------------------------------------------------------------------------
  template <typename V, typename VReal, size_t N>
  void render_ui(const vectorfield<V, VReal, N, N>& v) {
    if (show_nodes_gui) {
      render_node_editor();
    }
    //if (show_gui) {
    //  render_gui(v);
    //}
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
