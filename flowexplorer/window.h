#ifndef TATOOINE_FLOWEXPLORER_WINDOW_H
#define TATOOINE_FLOWEXPLORER_WINDOW_H
//==============================================================================
#include "imgui-node-editor/imgui_node_editor.h"
#include <tatooine/boundingbox.h>
#include <tatooine/gpu/first_person_window.h>
#include <tatooine/interpolation.h>

#include "boundingbox.h"
#include "grid.h"
#include "random_pathlines.h"
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
auto show_label(const char* label, ImColor color) {
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetTextLineHeight());
  auto size = ImGui::CalcTextSize(label);

  auto padding = ImGui::GetStyle().FramePadding;
  auto spacing = ImGui::GetStyle().ItemSpacing;

  ImGui::SetCursorPos(ImGui::GetCursorPos() + ImVec2(spacing.x, -spacing.y));

  auto rectMin = ImGui::GetCursorScreenPos() - padding;
  auto rectMax = ImGui::GetCursorScreenPos() + size + padding;

  auto drawList = ImGui::GetWindowDrawList();
  drawList->AddRectFilled(rectMin, rectMax, color, size.y * 0.15f);
  ImGui::TextUnformatted(label);
}
//==============================================================================
struct window : first_person_window {
  struct NodeIdLess {
    bool operator()(const ax::NodeEditor::NodeId& lhs, const ax::NodeEditor::NodeId& rhs) const {
      return lhs.AsPointer() < rhs.AsPointer();
    }
  };
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
  auto find_node(ax::NodeEditor::NodeId id) -> renderable* {
    for (auto& r : m_renderables) {
      if (r->id() == id) {
        return r.get();
      }
    }
    return nullptr;
  }
  //----------------------------------------------------------------------------
  auto find_pin(ax::NodeEditor::PinId id) -> ui::pin* {
    for (auto& r : m_renderables) {
      for (auto& p : r->input_pins()) {
        if (p.id() == id) {
          return &p;
        }
      }
      for (auto& p : r->output_pins()) {
        if (p.id() == id) {
          return &p;
        }
      }
    }
    return nullptr;
  }
  //----------------------------------------------------------------------------
  template <typename V, typename VReal, size_t N>
  void render_node_editor(const vectorfield<V, VReal, N, N>& v) {
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
      std::cerr << "inserted boundingbox: " << m_renderables.back().get()
                << '\n';
    }
    if (ImGui::Button("add random pathlines")) {
      m_renderables.emplace_back(new random_pathlines<double, 3>{*this, v});
      std::cerr << "inserted random_pathlines: " << m_renderables.back().get()
                << '\n';
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
      ed::PinId start_pin_id, end_pin_id;
      if (ed::QueryNewLink(&start_pin_id, &end_pin_id)) {
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

        if (start_pin_id && end_pin_id) {  // both are valid, let's accept link
          
          ui::pin* start_pin = find_pin(start_pin_id);
          ui::pin* end_pin = find_pin(end_pin_id);

          if (start_pin->node().id() == end_pin->node().id()) {
            show_label("cannot connect to same node", ImColor(45, 32, 32, 180));
            ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
          } else if (start_pin->kind() == end_pin->kind()) {
            show_label("both are input or output", ImColor(45, 32, 32, 180));
            ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
          } else if (ed::AcceptNewItem()) {
            start_pin->node().on_pin_connected(*start_pin, *end_pin);
            end_pin->node().on_pin_connected(*end_pin, *start_pin);
            // Since we accepted new link, lets add one to our list of links.
            m_links.push_back(
                {ed::LinkId(m_next_link++), start_pin_id, end_pin_id});

            // Draw new link.
            ed::Link(m_links.back().id, m_links.back().input_id,
                     m_links.back().output_id);
          }
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
              ui::pin* input_pin  = find_pin(link.input_id);
              ui::pin* output_pin = find_pin(link.output_id);
              input_pin->node().on_pin_disconnected(*input_pin);
              output_pin->node().on_pin_disconnected(*output_pin);
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
      render_node_editor(v);
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
