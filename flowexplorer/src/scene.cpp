#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/demangling.h>
#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/window.h>
#include <tatooine/interpolation.h>
#include <tatooine/flowexplorer/ui/pinkind.h>
#include <tatooine/flowexplorer/nodes/axis_aligned_bounding_box.h>
#include <toml++/toml.h>

#include <fstream>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
bool scene::items_created = false;
std::set<std::string_view> scene::items;
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
scene::scene(rendering::camera_controller<float>& cam, flowexplorer::window* w)
    : m_cam{&cam}, m_window{w} {
  m_node_editor_context = ax::NodeEditor::CreateEditor();
  if (!items_created) {
    iterate_registered_names(name) { items.insert(name->f()); }
    items_created = true;
  }
  

  ImVec4 *colors                         = ImGui::GetStyle().Colors;
//colors[ImGuiCol_Text]                  = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
//colors[ImGuiCol_TextDisabled]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
//colors[ImGuiCol_WindowBg]              = ImVec4(0.06f, 0.06f, 0.06f, 0.94f);
//colors[ImGuiCol_ChildBg]               = ImVec4(1.00f, 1.00f, 1.00f, 0.00f);
//colors[ImGuiCol_PopupBg]               = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
//colors[ImGuiCol_Border]                = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
//colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
//colors[ImGuiCol_FrameBg]               = ImVec4(0.20f, 0.21f, 0.22f, 0.54f);
//colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.40f, 0.40f, 0.40f, 0.40f);
//colors[ImGuiCol_FrameBgActive]         = ImVec4(0.18f, 0.18f, 0.18f, 0.67f);
//colors[ImGuiCol_TitleBg]               = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
//colors[ImGuiCol_TitleBgActive]         = ImVec4(0.29f, 0.29f, 0.29f, 1.00f);
//colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
//colors[ImGuiCol_MenuBarBg]             = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
//colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
//colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
//colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
//colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
//colors[ImGuiCol_CheckMark]             = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
//colors[ImGuiCol_SliderGrab]            = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
//colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
//colors[ImGuiCol_Button]                = ImVec4(0.44f, 0.44f, 0.44f, 0.40f);
//colors[ImGuiCol_ButtonHovered]         = ImVec4(0.46f, 0.47f, 0.48f, 1.00f);
//colors[ImGuiCol_ButtonActive]          = ImVec4(0.42f, 0.42f, 0.42f, 1.00f);
//colors[ImGuiCol_Header]                = ImVec4(0.70f, 0.70f, 0.70f, 0.31f);
//colors[ImGuiCol_HeaderHovered]         = ImVec4(0.70f, 0.70f, 0.70f, 0.80f);
//colors[ImGuiCol_HeaderActive]          = ImVec4(0.48f, 0.50f, 0.52f, 1.00f);
//colors[ImGuiCol_Separator]             = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
//colors[ImGuiCol_SeparatorHovered]      = ImVec4(0.72f, 0.72f, 0.72f, 0.78f);
//colors[ImGuiCol_SeparatorActive]       = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
//colors[ImGuiCol_ResizeGrip]            = ImVec4(0.91f, 0.91f, 0.91f, 0.25f);
//colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.81f, 0.81f, 0.81f, 0.67f);
//colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.46f, 0.46f, 0.46f, 0.95f);
//colors[ImGuiCol_PlotLines]             = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
//colors[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
//colors[ImGuiCol_PlotHistogram]         = ImVec4(0.73f, 0.60f, 0.15f, 1.00f);
//colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
//colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.87f, 0.87f, 0.87f, 0.35f);
//colors[ImGuiCol_ModalWindowDarkening]  = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
//colors[ImGuiCol_DragDropTarget]        = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
//colors[ImGuiCol_NavHighlight]          = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
//colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);

  ImGui::GetStyle().WindowRounding    = 0.0f;
  ImGui::GetStyle().ChildRounding     = 0.0f;
  ImGui::GetStyle().FrameRounding     = 0.0f;
  ImGui::GetStyle().GrabRounding      = 0.0f;
  ImGui::GetStyle().PopupRounding     = 0.0f;
  ImGui::GetStyle().ScrollbarRounding = 0.0f;


  namespace ed = ax::NodeEditor;
  ed::SetCurrentEditor(m_node_editor_context);
  auto& style = ed::GetStyle();
  style.Colors[ed::StyleColor_Bg]                 = colors[ImGuiCol_ChildBg];
  style.Colors[ed::StyleColor_Grid]               = style.Colors[ed::StyleColor_Bg];
  style.Colors[ed::StyleColor_NodeBg]             = ImColor( 32,  32,  32, 200);
  style.Colors[ed::StyleColor_NodeBorder]         = ImColor(127, 127, 127,  96);
  style.Colors[ed::StyleColor_HovNodeBorder]      = ImColor(200, 200, 200, 255);
  style.Colors[ed::StyleColor_SelNodeBorder]      = ImColor(255, 255, 255, 255);
  style.Colors[ed::StyleColor_NodeSelRect]        = ImColor(  5, 130, 255,  64);
  style.Colors[ed::StyleColor_NodeSelRectBorder]  = ImColor(  5, 130, 255, 128);
  style.Colors[ed::StyleColor_HovLinkBorder]      = ImColor( 50, 176, 255, 255);
  style.Colors[ed::StyleColor_SelLinkBorder]      = ImColor(255, 176,  50, 255);
  style.Colors[ed::StyleColor_LinkSelRect]        = ImColor(  5, 130, 255,  64);
  style.Colors[ed::StyleColor_LinkSelRectBorder]  = ImColor(  5, 130, 255, 128);
  style.Colors[ed::StyleColor_PinRect]            = ImColor( 60, 180, 255, 100);
  style.Colors[ed::StyleColor_PinRectBorder]      = ImColor( 60, 180, 255, 128);
  style.Colors[ed::StyleColor_Flow]               = ImColor(255, 128,  64, 255);
  style.Colors[ed::StyleColor_FlowMarker]         = ImColor(102, 255, 153, 255);
  style.Colors[ed::StyleColor_GroupBg]            = ImColor(  0,   0,   0, 160);
  style.Colors[ed::StyleColor_GroupBorder]        = ImColor(255, 255, 255,  32);
  style.NodeRounding                              = 5.0f;
  ed::SetCurrentEditor(nullptr);
}
//------------------------------------------------------------------------------
scene::scene(rendering::camera_controller<float>& cam, flowexplorer::window* w,
             filesystem::path const& path)
    : scene{cam, w} {
  read(path);
}
//------------------------------------------------------------------------------
scene::~scene() { ax::NodeEditor::DestroyEditor(m_node_editor_context); }
//------------------------------------------------------------------------------
auto scene::remove_link(ui::link const& link_to_remove) -> void {
  auto same_id = [&link_to_remove](auto const& present_link) {
    return link_to_remove.get_id_number() == present_link->get_id_number();
  };
  m_links.erase(std::remove_if(begin(m_links), end(m_links), same_id),
                end(m_links));
}
//------------------------------------------------------------------------------
auto scene::render(std::chrono::duration<double> const& dt) -> void {
  gl::clear_color(255, 255, 255, 255);
  gl::clear_color_depth_buffer();
  for (auto& n : m_nodes) {
    n->update_property_links();
    if (n->is_active()) {
      n->update(dt);
    }
  }
  for (auto& r : m_renderables) {
    r->update_property_links();
    if (r->is_active()) {
      r->update(dt);
    }
  }

  // render non-transparent objects
  gl::enable_depth_write();
  gl::disable_blending();
  for (auto& r : m_renderables) {
    if (r->is_active()) {
      if (!r->is_transparent()) {
        r->render(m_cam->projection_matrix(), m_cam->view_matrix());
      }
    }
  }

  // render transparent objects
  gl::disable_depth_write();
  gl::enable_blending();
  gl::blend_func_alpha();
  for (auto& r : m_renderables) {
    if (r->is_active()) {
      if (r->is_transparent()) {
        r->render(m_cam->projection_matrix(), m_cam->view_matrix());
      }
    }
  }
  gl::enable_depth_test();
  gl::enable_depth_write();
}
//------------------------------------------------------------------------------
auto scene::find_node(ax::NodeEditor::NodeId const& id) -> ui::base::node* {
  for (auto& n : m_nodes) {
    if (*n == id) {
      return n.get();
    }
  }
  for (auto& r : m_renderables) {
    if (*r == id) {
      return r.get();
    }
  }
  return nullptr;
}
//------------------------------------------------------------------------------
auto scene::find_input_pin(ax::NodeEditor::PinId const& id) -> ui::input_pin* {
  auto find = [&id](auto& cont) -> ui::input_pin* {
    for (auto& n : cont) {
      for (auto& p : n->input_pins()) {
        if (*p == id) {
          return p.get();
        }
      }
    }
    return nullptr;
  };

  if (auto ptr = find(m_renderables); ptr != nullptr) {
    return ptr;
  }
  if (auto ptr = find(m_nodes); ptr != nullptr) {
    return ptr;
  }
  return nullptr;
}
//------------------------------------------------------------------------------
auto scene::find_output_pin(ax::NodeEditor::PinId const& id)
    -> ui::output_pin* {
  auto find = [&id](auto& cont) -> ui::output_pin* {
    for (auto& n : cont) {
      if (n->has_self_pin() && n->self_pin() == id) {
        return &n->self_pin();
      }
      for (auto& p : n->output_pins()) {
        if (*p == id) {
          return p.get();
        }
      }
    }
    return nullptr;
  };

  if (auto ptr = find(m_renderables); ptr != nullptr) {
    return ptr;
  }
  if (auto ptr = find(m_nodes); ptr != nullptr) {
    return ptr;
  }
  return nullptr;
}
//------------------------------------------------------------------------------
auto scene::draw_nodes() -> void {
  namespace ed = ax::NodeEditor;
  size_t i     = 0;

  for (auto& n : m_nodes) {
    ImGui::PushID(i++);
    n->draw_node();
    ImGui::PopID();
  }
  for (auto& r : m_renderables) {
    ImGui::PushID(i++);
    r->draw_node();
    ImGui::PopID();
  }
}
//------------------------------------------------------------------------------
auto scene::draw_links() -> void {
  namespace ed = ax::NodeEditor;
  for (auto& link : m_links) {
    ed::Link(link->get_id(), link->input().get_id(), link->output().get_id());
  }
}
//------------------------------------------------------------------------------
auto scene::link(ui::input_pin& in, ui::output_pin& out) -> ui::link& {
  auto& l = m_links.emplace_back(std::make_unique<ui::link>(in, out));
  in.link(*l);
  out.link(*l);

  namespace ed = ax::NodeEditor;
  ed::Link(l->get_id(), in.get_id(), out.get_id());
  return *l;
}
//------------------------------------------------------------------------------
auto scene::can_create_link(ui::input_pin const& /*pin0*/,
                            ui::input_pin const& /*pin1*/) -> bool {
  return false;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto scene::can_create_link(ui::output_pin const& /*pin0*/,
                            ui::output_pin const& /*pin1*/) -> bool {
  return false;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto scene::can_create_link(ui::output_pin const& pin0,
                            ui::input_pin const&  pin1) -> bool {
  return can_create_link(pin1, pin0);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto scene::can_create_link(ui::input_pin const&  pin0,
                            ui::output_pin const& pin1) -> bool {
  return std::any_of(begin(pin0.types()), end(pin0.types()),
                     [&pin1](auto t) { return *t == pin1.type(); });
}
//----------------------------------------------------------------------------
auto scene::can_create_new_link(ui::input_pin const& pin) -> bool{
  if (m_new_link_start_output == nullptr) {
    return false;
  }
  return can_create_link(pin, *m_new_link_start_output);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto scene::can_create_new_link(ui::output_pin const& pin) -> bool {
  if (m_new_link_start_input == nullptr) {
    return false;
  }
  return can_create_link(pin, *m_new_link_start_input);
}
//------------------------------------------------------------------------------
auto scene::query_link_creation() -> void {
  namespace ed = ax::NodeEditor;
  if (ed::BeginCreate()) {
    ed::PinId pin_id0, pin_id1;
    if (ed::QueryNewLink(&pin_id0, &pin_id1)) {
      if (pin_id0 && pin_id1) {  // both are valid, let's accept link

        auto input_pin0  = find_input_pin(pin_id0);
        auto output_pin0 = find_output_pin(pin_id0);
        auto input_pin1  = find_input_pin(pin_id1);
        auto output_pin1 = find_output_pin(pin_id1);

        size_t num_found_pins = 0;
        if (input_pin0 != nullptr) {
          ++num_found_pins;
        }
        if (input_pin1 != nullptr) {
          ++num_found_pins;
        }
        if (output_pin0 != nullptr) {
          ++num_found_pins;
        }
        if (output_pin1 != nullptr) {
          ++num_found_pins;
        }
        assert(num_found_pins == 2);

        auto pins = [&]() -> std::pair<ui::input_pin*, ui::output_pin*> {
          if (input_pin0 != nullptr && output_pin1 != nullptr) {
            return {input_pin0, output_pin1};
          } else if (input_pin1 != nullptr && output_pin0 != nullptr) {
            return {input_pin1, output_pin0};
          }
          return {nullptr, nullptr};
        }();
        auto input_pin  = pins.first;
        auto output_pin = pins.second;

        if (input_pin0 != nullptr && input_pin1 != nullptr) {
          show_label("cannot two input nodes", ImColor(45, 32, 32, 180));
          ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
        } else if (output_pin0 != nullptr && output_pin1 != nullptr) {
          show_label("cannot two output nodes", ImColor(45, 32, 32, 180));
          ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
        } else if (input_pin->node().get_id() == output_pin->node().get_id()) {
          show_label("cannot connect to same node", ImColor(45, 32, 32, 180));
          ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
        } else if (!can_create_link(*input_pin, *output_pin)) {
          std::string msg = "Types do not match:\n";
          // msg += type_name(input_pin->type());
          // msg += "\n";
          msg += type_name(output_pin->type());
          show_label(msg.c_str(), ImColor(45, 32, 32, 180));
          ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
        } else if (ed::AcceptNewItem()) {

          // For all links check if the queried input pin already is linked. If
          // so remove old input.
          for (auto link_it = begin(m_links); link_it != end(m_links);
               ++link_it) {
            auto present_input_pin =
                find_input_pin((*link_it)->input().get_id());
            if (*present_input_pin == *input_pin) {
              present_input_pin->unlink();
              break;
            }
          }

          link(*input_pin, *output_pin);
        }
      }
    }

    ed::PinId pinId = 0;
    if (ed::QueryNewNode(&pinId)) {
      m_new_link_start_input  = find_input_pin(pinId.Get());
      m_new_link_start_output = find_output_pin(pinId.Get());
      m_new_link              = true;

      if (ed::AcceptNewItem()) {
        m_new_link              = false;
        m_new_link_start_input  = nullptr;
        m_new_link_start_output = nullptr;
        ed::Suspend();
        ImGui::OpenPopup("Create New Node");
        ed::Resume();
      }
    }

  } else {
    m_new_link              = false;
    m_new_link_start_input  = nullptr;
    m_new_link_start_output = nullptr;
  }
  ed::EndCreate();
}
//------------------------------------------------------------------------------
auto scene::query_link_and_node_deletions() -> void {
  namespace ed = ax::NodeEditor;
  // Handle deletion action
  if (ed::BeginDelete()) {
    // There may be many links marked for deletion, let's loop over them.
    ed::LinkId deleted_link_id;
    while (ed::QueryDeletedLink(&deleted_link_id)) {
      // If you agree that link can be deleted, accept deletion.
      if (ed::AcceptDeletedItem()) {
        // Then remove link from your data.
        auto link_it = std::find_if(begin(m_links), end(m_links),
            [&](auto const& l){return *l == deleted_link_id;} );
        if (link_it != end(m_links)) {
          auto input_pin  = find_input_pin((*link_it)->input().get_id_number());
          input_pin->unlink();
          break;
        }
      }
    }
  }
  ed::NodeId node_id = 0;
  while (ed::QueryDeletedNode(&node_id)) {
    if (ed::AcceptDeletedItem()) {
      auto node_it = std::find_if(
          begin(m_nodes), end(m_nodes),
          [node_id](auto& node) { return node->get_id() == node_id; });
      if (node_it != end(m_nodes)) {
        for (auto& pin : node_it->get()->input_pins()) {
          if (pin->is_linked()) {
            m_links.erase(std::remove_if(begin(m_links), end(m_links),
                                         [&](auto const& l) {
                                           return *l == pin->link();
                                         }),
                          end(m_links));
            pin->unlink();
          }
        }
        for (auto& pin : node_it->get()->output_pins()) {
          pin->unlink_all();
        }
        m_nodes.erase(node_it);

      } else {
        auto renderable_it =
            std::find_if(begin(m_renderables), end(m_renderables),
                         [node_id](auto& renderable) {
                           return renderable->get_id() == node_id;
                         });
        for (auto& pin : renderable_it->get()->input_pins()) {
          pin->unlink();
        }
        for (auto& pin : renderable_it->get()->output_pins()) {
          pin->unlink_all();
        }
        m_renderables.erase(renderable_it);
      }
    }
  }
  ed::EndDelete();
}
//------------------------------------------------------------------------------
void scene::draw_node_editor(size_t const pos_x, size_t const pos_y,
                             size_t const width, size_t const height,
                             bool& show) {
  namespace ed = ax::NodeEditor;
  window().push_regular_font();
  ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y));
  ImGui::SetNextWindowSize(ImVec2(window().width()*2/3.0, height));

  ed::SetCurrentEditor(m_node_editor_context);
  ImGui::Begin("Scene", &show,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoBringToFrontOnFocus |
                   ImGuiWindowFlags_NoTitleBar);
  node_creators(width - 20);
  ed::Begin("My Editor", ImVec2(0.0, 0.0f));
  draw_nodes();
  draw_links();
  query_link_creation();
  query_link_and_node_deletions();
  ed::End();
  ImGui::End();

  ImGui::SetNextWindowPos(ImVec2(window().width() * 2 / 3.0, pos_y));
  ImGui::SetNextWindowSize(ImVec2(window().width() / 3.0, height));
  ImGui::Begin("NodeDetail", &show,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoBringToFrontOnFocus |
                   ImGuiWindowFlags_NoTitleBar);
  ed::NodeId selected_node_id;
  if (ed::GetSelectedNodes(&selected_node_id, 1) == 1) {
    auto selected_node = find_node(selected_node_id);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
    auto const old_title = selected_node->title();

    window().push_header1_font();
    if (ImGui::InputText("##title", &selected_node->title())) {
      selected_node->on_title_changed(old_title);
    }
    window().pop_font();
    ImGui::PopStyleColor();
    ImGui::Separator();
    auto const changed = selected_node->draw_properties();
    if (changed) {
      selected_node->notify_property_changed();
    }

  }

  ImGui::End();
  ed::SetCurrentEditor(nullptr);
  window().pop_font();
}
//------------------------------------------------------------------------------
void scene::node_creators(size_t const width) {
  ImTextureID aabb2d_id =
      reinterpret_cast<ImTextureID>(window().aabb2d_icon_tex().id());
  if (ImGui::ImageButton(aabb2d_id, ImVec2(50 * window().ui_scale_factor(),
                                           50 * window().ui_scale_factor()))) {
    m_renderables.emplace_back(new nodes::aabb2d{*this});
  }
  ImGui::SameLine();
  ImTextureID aabb3d_id =
      reinterpret_cast<ImTextureID>(window().aabb3d_icon_tex().id());
  if (ImGui::ImageButton(aabb3d_id, ImVec2(50 * window().ui_scale_factor(),
                                           50 * window().ui_scale_factor()))) {
    m_renderables.emplace_back(new nodes::aabb3d{*this});
  }
  ImGui::PushItemWidth(width);
  if (ImGui::BeginCombo("##combo", nullptr)) {
    for (auto const& item : items) {
      if (ImGui::Selectable(std::string{item}.c_str(), false)) {
        insert_registered_element(*this, item);
      }
    }
    ImGui::EndCombo();
  }
  ImGui::PopItemWidth();
}
//------------------------------------------------------------------------------
void scene::write(filesystem::path const& filepath) const {
  toml::table toml_scene;

  auto write_nodes = [&](std::string_view const& kind, auto const& field) {
    for (auto const& node : field) {
      auto        serialized_node = node->serialize();
      auto        pos             = node->node_position();
      toml::array serialized_input_pins, serialized_output_pins;
      for (auto const& pin : node->input_pins()) {
        toml::table serialized_pin;
        serialized_pin.insert("active", pin->is_active());
        serialized_pin.insert("id", long(pin->get_id_number()));
        serialized_input_pins.push_back(serialized_pin);
      }
      for (auto const& pin : node->output_pins()) {
        toml::table serialized_pin;
        serialized_pin.insert("active", pin->is_active());
        serialized_pin.insert("id", long(pin->get_id_number()));
        serialized_output_pins.push_back(serialized_pin);
      }
      serialized_node.insert("kind", kind);
      serialized_node.insert("input_pins", serialized_input_pins);
      serialized_node.insert("output_pins", serialized_output_pins);
      serialized_node.insert("node_position",
                             toml::array{pos[0] / window().ui_scale_factor(),
                                         pos[1] / window().ui_scale_factor()});
      serialized_node.insert("node_title", node->title());
      serialized_node.insert("node_type", node->type_name());
      serialized_node.insert("active", node->is_active());
      if (node->has_self_pin()) {
        serialized_node.insert("self_pin",
                               long(node->self_pin().get_id_number()));
      }
      toml_scene.insert(std::to_string(node->get_id_number()), serialized_node);
    }
  };
  write_nodes("node", m_nodes);
  write_nodes("renderable", m_renderables);

  for (auto const& link : m_links) {
    toml::table serialized_link;
    serialized_link.insert("input", long(link->input().get_id_number()));
    serialized_link.insert("output", long(link->output().get_id_number()));
    serialized_link.insert("kind", "link");
    toml_scene.insert(std::to_string(link->get_id_number()), serialized_link);
  }

  // write camera
  // TODO write perspective camera settings
  toml::table serialized_camera;
  toml::table serialized_orthographic_camera;
  auto const& ortho_cam = m_cam->orthographic_camera();
  serialized_orthographic_camera.insert(
      "eye",
      toml::array{ortho_cam.eye()(0), ortho_cam.eye()(1), ortho_cam.eye()(2)});
  serialized_orthographic_camera.insert(
      "lookat", toml::array{ortho_cam.lookat()(0), ortho_cam.lookat()(1),
                            ortho_cam.lookat()(2)});
  serialized_orthographic_camera.insert(
      "up",
      toml::array{ortho_cam.up()(0), ortho_cam.up()(1), ortho_cam.up()(2)});
  serialized_orthographic_camera.insert("far", ortho_cam.far());
  serialized_orthographic_camera.insert("height", ortho_cam.height());
  serialized_orthographic_camera.insert("near", ortho_cam.near());
  serialized_camera.insert("orthographic", serialized_orthographic_camera);
  serialized_camera.insert("kind", "camera");
  serialized_camera.insert("controller", type_name(m_cam->controller().type()));
  toml_scene.insert("camera", serialized_camera);

  std::ofstream fout{filepath};
  if (fout.is_open()) {
    fout << toml_scene << '\n';
  }
}
//------------------------------------------------------------------------------
void scene::read(filesystem::path const& filepath) {
  clear();
  ax::NodeEditor::SetCurrentEditor(m_node_editor_context);
  auto const toml_scene = toml::parse_file(filepath.string());

  // read nodes and renderables
  for (auto const& [id_string, item] : toml_scene) {
    auto const& serialized_node = *item.as_table();
    auto const  kind            = serialized_node["kind"].as_string()->get();

    if (kind == "node" || kind == "renderable") {
      auto const node_type_name =
          serialized_node["node_type"].as_string()->get();

      ui::base::node* n = insert_registered_element(*this, node_type_name);
      assert(n != nullptr);

      // id string to size_t
      std::stringstream id_stream{id_string};
      size_t            id;
      id_stream >> id;
      n->set_id(id);
      auto const& serialized_input_pins =
          *serialized_node["input_pins"].as_array();
      auto const& serialized_output_pins =
          *serialized_node["output_pins"].as_array();
      size_t i = 0;
      for (auto& pin : n->input_pins()) {
        auto& serialized_pin = *serialized_input_pins[i++].as_table();
        pin->set_id(size_t(serialized_pin["id"].as_integer()->get()));
        pin->set_active(serialized_pin["active"].as_boolean()->get());
      }
      i = 0;
      for (auto& pin : n->output_pins()) {
        auto& serialized_pin = *serialized_output_pins[i++].as_table();
        pin->set_id(size_t(serialized_pin["id"].as_integer()->get()));
        pin->set_active(serialized_pin["active"].as_boolean()->get());
      }

      // set node position
      auto const x = (*serialized_node["node_position"].as_array())[0]
                         .as_floating_point()
                         ->get();
      auto const y = (*serialized_node["node_position"].as_array())[1]
                         .as_floating_point()
                         ->get();
      ImVec2 pos{static_cast<float>(x) * window().ui_scale_factor(),
                 static_cast<float>(y) * window().ui_scale_factor()};
      ax::NodeEditor::SetNodePosition(id, pos);

      // set title
      auto const title = serialized_node["node_title"].as_string()->get();
      n->set_title(title);

      // enable or disable
      auto const active = serialized_node["active"].as_boolean()->get();
      n->set_active(active);

      // enable or disable
      if (n->has_self_pin()) {
        n->self_pin().set_id(serialized_node["self_pin"].as_integer()->get());
      }

      n->deserialize(serialized_node);
    }
  }

  // read links
  for (auto const& [id_string, item] : toml_scene) {
    auto const& serialized_node = *item.as_table();
    auto const  kind            = serialized_node["kind"].as_string()->get();
    if (kind == "link") {
      // id string to size_t
      std::stringstream id_stream{id_string};
      size_t            id;
      id_stream >> id;
      size_t const input_id   = serialized_node["input"].as_integer()->get();
      size_t const output_id  = serialized_node["output"].as_integer()->get();
      auto         input_pin  = find_input_pin(input_id);
      auto         output_pin = find_output_pin(output_id);
      assert(input_pin != nullptr);
      assert(output_pin != nullptr);
      auto& l = m_links.emplace_back(
          std::make_unique<ui::link>(id, *input_pin, *output_pin));
      input_pin->link(*l);
      output_pin->link(*l);
      // ax::NodeEditor::Link(l.get_id(), l.input().get_id(),
      // l.output().get_id());
    }
  }
  for (auto const& [id_string, item] : toml_scene) {
    auto const& serialized_node = *item.as_table();
    auto const  kind            = serialized_node["kind"].as_string()->get();
    if (kind == "camera") {
      auto const& controller_type_name =
          serialized_node["controller"].as_string()->get();
      if (controller_type_name ==
          type_name(typeid(rendering::orthographic_camera_controller<float>))) {
        m_cam->use_orthographic_camera();
        m_cam->use_orthographic_controller();
      }
      // TODO read perspective camera settings
      {
        auto const& serialized_orthographic_camera =
            *serialized_node["orthographic"].as_table();

        auto const& eye = *serialized_orthographic_camera["eye"].as_array();
        auto const& lookat =
            *serialized_orthographic_camera["lookat"].as_array();
        auto const& up = *serialized_orthographic_camera["up"].as_array();

        auto const near =
            serialized_orthographic_camera["near"].as_floating_point()->get();
        auto const far =
            serialized_orthographic_camera["far"].as_floating_point()->get();
        auto const height =
            serialized_orthographic_camera["height"].as_floating_point()->get();

        m_cam->orthographic_camera().setup(
            vec3f{eye[0].as_floating_point()->get(),
                  eye[1].as_floating_point()->get(),
                  eye[2].as_floating_point()->get()},
            vec3f{lookat[0].as_floating_point()->get(),
                  lookat[1].as_floating_point()->get(),
                  lookat[2].as_floating_point()->get()},
            vec3f{up[0].as_floating_point()->get(),
                  up[1].as_floating_point()->get(),
                  up[2].as_floating_point()->get()},
            height, near, far, window().width(), window().height());
      }
    }
  }
  ax::NodeEditor::SetCurrentEditor(nullptr);
}
//------------------------------------------------------------------------------
void scene::open_file(filesystem::path const& filepath) {
  auto const ext = filepath.extension().string();
  if (ext == ".toml" || ext == "toml" ||
      ext == ".scene" || ext == "scene") {
    read(filepath);
  } else if (ext == ".vtk" || ext == "vtk") {
    // TODO add vtk reader
  }
}
//----------------------------------------------------------------------------
auto scene::clear() -> void {
  m_nodes.clear();
  m_renderables.clear();
  m_links.clear();
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
