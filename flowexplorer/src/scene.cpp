#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/boundingbox.h>
#include <tatooine/demangling.h>
#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/window.h>
#include <tatooine/interpolation.h>
#include <tatooine/rendering/yavin_interop.h>
#include <toml++/toml.h>

#include <fstream>
#include <yavin>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
bool scene::items_created = false;
std::set<std::string_view> scene::items;

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
}
//----------------------------------------------------------------------------
scene::scene(rendering::camera_controller<float>& cam, flowexplorer::window* w,
             std::filesystem::path const& path)
    : m_cam{&cam}, m_window{w} {
  m_node_editor_context = ax::NodeEditor::CreateEditor();
  read(path);
}
//----------------------------------------------------------------------------
scene::~scene() { ax::NodeEditor::DestroyEditor(m_node_editor_context); }
//------------------------------------------------------------------------------
void scene::render(std::chrono::duration<double> const& dt) {
  yavin::gl::clear_color(255, 255, 255, 255);
  yavin::clear_color_depth_buffer();
  for (auto& r : m_renderables) {
    if (r->is_enabled()) {
      r->update(dt);
    }
  }

  // render non-transparent objects
  yavin::enable_depth_write();
  yavin::disable_blending();
  for (auto& r : m_renderables) {
    if (r->is_enabled()) {
      if (!r->is_transparent()) {
        r->render(m_cam->projection_matrix(), m_cam->view_matrix());
      }
    }
  }

  // render transparent objects
  yavin::disable_depth_write();
  yavin::enable_blending();
  yavin::blend_func_alpha();
  for (auto& r : m_renderables) {
    if (r->is_enabled()) {
      if (r->is_transparent()) {
        r->render(m_cam->projection_matrix(), m_cam->view_matrix());
      }
    }
  }
  yavin::enable_depth_test();
  yavin::enable_depth_write();
}
//----------------------------------------------------------------------------
auto scene::find_node(size_t const id) -> ui::base::node* {
  for (auto& n : m_nodes) {
    if (n->get_id_number() == id) {
      return n.get();
    }
  }
  for (auto& r : m_renderables) {
    if (r->get_id_number() == id) {
      return r.get();
    }
  }
  return nullptr;
}
//----------------------------------------------------------------------------
auto scene::find_pin(size_t const id) -> ui::pin* {
  for (auto& r : m_renderables) {
    for (auto& p : r->input_pins()) {
      if (p.get_id_number() == id) {
        return &p;
      }
    }
    for (auto& p : r->output_pins()) {
      if (p.get_id_number() == id) {
        return &p;
      }
    }
  }
  for (auto& n : m_nodes) {
    for (auto& p : n->input_pins()) {
      if (p.get_id_number() == id) {
        return &p;
      }
    }
    for (auto& p : n->output_pins()) {
      if (p.get_id_number() == id) {
        return &p;
      }
    }
  }
  return nullptr;
}
//----------------------------------------------------------------------------
void scene::draw_nodes() {
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
//----------------------------------------------------------------------------
void scene::draw_links() {
  namespace ed = ax::NodeEditor;
  for (auto& link_info : m_links) {
    ed::Link(link_info.get_id(), link_info.input().get_id(),
             link_info.output().get_id());
  }
}
//----------------------------------------------------------------------------
void scene::create_link() {
  namespace ed = ax::NodeEditor;
  if (ed::BeginCreate()) {
    ed::PinId input_pin_id, output_pin_id;
    if (ed::QueryNewLink(&input_pin_id, &output_pin_id)) {
      if (input_pin_id && output_pin_id) {  // both are valid, let's accept link

        ui::pin* input_pin  = find_pin(input_pin_id.Get());
        ui::pin* output_pin = find_pin(output_pin_id.Get());

        if (input_pin->kind() == ui::pinkind::output) {
          std::swap(input_pin, output_pin);
          std::swap(input_pin_id, output_pin_id);
        }

        if (input_pin->node().get_id() == output_pin->node().get_id()) {
          show_label("cannot connect to same node", ImColor(45, 32, 32, 180));
          ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
        } else if (input_pin->kind() == output_pin->kind()) {
          show_label("both are input or output", ImColor(45, 32, 32, 180));
          ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
        } else if (input_pin->type() != output_pin->type()) {
          std::string msg = "Types do not match:\n";
          msg += type_name(input_pin->type());
          msg += "\n";
          msg += type_name(output_pin->type());
          show_label(msg.c_str(), ImColor(45, 32, 32, 180));
          ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
        } else if (ed::AcceptNewItem()) {
          // remove old input link if present
          for (auto link_it = begin(m_links); link_it != end(m_links);
               ++link_it) {
            ui::pin* present_input_pin =
                find_pin(link_it->input().get_id_number());
            if (present_input_pin->get_id() == input_pin->get_id()) {
              ui::pin* present_output_pin =
                  find_pin(link_it->output().get_id_number());
              present_input_pin->node().on_pin_disconnected(*present_input_pin);
              present_output_pin->node().on_pin_disconnected(
                  *present_output_pin);
              m_links.erase(link_it);
              break;
            }
          }

          input_pin->node().on_pin_connected(*input_pin, *output_pin);
          output_pin->node().on_pin_connected(*output_pin, *input_pin);
          // Since we accepted new link, lets add one to our list of links.
          m_links.emplace_back(*input_pin, *output_pin);

          // Draw new link.
          ed::Link(m_links.back().get_id(), m_links.back().input().get_id(),
                   m_links.back().output().get_id());
        }
      }
    }
  }
}
//----------------------------------------------------------------------------
void scene::remove_link() {
  namespace ed = ax::NodeEditor;
  ed::EndCreate();
  // Handle deletion action
  if (ed::BeginDelete()) {
    // There may be many links marked for deletion, let's loop over them.
    ed::LinkId deletedLinkId;
    while (ed::QueryDeletedLink(&deletedLinkId)) {
      // If you agree that link can be deleted, accept deletion.
      if (ed::AcceptDeletedItem()) {
        // Then remove link from your data.
        for (auto link_it = begin(m_links); link_it != end(m_links);
             ++link_it) {
          if (link_it->get_id() == deletedLinkId) {
            ui::pin* input_pin  = find_pin(link_it->input().get_id_number());
            ui::pin* output_pin = find_pin(link_it->output().get_id_number());
            input_pin->node().on_pin_disconnected(*input_pin);
            output_pin->node().on_pin_disconnected(*output_pin);
            m_links.erase(link_it);
            break;
          }
        }
      }
    }
  }
  ed::EndDelete();
}
//----------------------------------------------------------------------------
void scene::draw_node_editor(size_t const pos_x, size_t const pos_y,
                             size_t const width, size_t const height,
                             bool& show) {
  namespace ed                        = ax::NodeEditor;
  ImGui::GetStyle().WindowRounding    = 0.0f;
  ImGui::GetStyle().ChildRounding     = 0.0f;
  ImGui::GetStyle().FrameRounding     = 0.0f;
  ImGui::GetStyle().GrabRounding      = 0.0f;
  ImGui::GetStyle().PopupRounding     = 0.0f;
  ImGui::GetStyle().ScrollbarRounding = 0.0f;
  ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y));
  ImGui::SetNextWindowSize(ImVec2(width, height));
  ImGui::Begin("Node Editor", &show,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoBringToFrontOnFocus |
                   ImGuiWindowFlags_NoTitleBar);
  ed::SetCurrentEditor(m_node_editor_context);

  node_creators();
  ed::Begin("My Editor", ImVec2(0.0, 0.0f));
  draw_nodes();
  draw_links();
  create_link();
  remove_link();
  ed::End();
  ImGui::End();
  ed::SetCurrentEditor(nullptr);
}
//----------------------------------------------------------------------------
void scene::node_creators() {
  if (ImGui::BeginCombo("##combo", nullptr)) {
    for (auto const& item : items) {
      if (ImGui::Selectable(std::string{item}.c_str(), false)) {
        ui::base::node* n = insert_registered_element(*this, item);
      }
    }
    ImGui::EndCombo();
  }
}
//------------------------------------------------------------------------------
void scene::write(std::filesystem::path const& filepath) const {
  toml::table toml_scene;

  auto write_nodes = [&](std::string_view const& kind, auto const& field) {
    for (auto const& node : field) {
      auto        serialized_node = node->serialize();
      auto        pos             = node->node_position();
      toml::array input_pin_ids, output_pin_ids;
      for (auto const& input_pin : node->input_pins()) {
        input_pin_ids.push_back(long(input_pin.get_id_number()));
      }
      for (auto const& output_pin : node->output_pins()) {
        output_pin_ids.push_back(long(output_pin.get_id_number()));
      }
      serialized_node.insert("kind", kind);
      serialized_node.insert("input_pin_ids", input_pin_ids);
      serialized_node.insert("output_pin_ids", output_pin_ids);
      serialized_node.insert("node_position", toml::array{pos[0], pos[1]});
      serialized_node.insert("node_title", node->title());
      serialized_node.insert("node_type", node->type_name());
      serialized_node.insert("enabled", node->is_enabled());
      toml_scene.insert(std::to_string(node->get_id_number()), serialized_node);
    }
  };
  write_nodes("node", m_nodes);
  write_nodes("renderable", m_renderables);

  for (auto const& link : m_links) {
    toml::table serialized_link;
    serialized_link.insert("input", long(link.input().get_id_number()));
    serialized_link.insert("output", long(link.output().get_id_number()));
    serialized_link.insert("kind", "link");
    toml_scene.insert(std::to_string(link.get_id_number()), serialized_link);
  }

  std::ofstream fout{filepath};
  if (fout.is_open()) {
    fout << toml_scene << '\n';
  }
}  //------------------------------------------------------------------------------
void scene::read(std::filesystem::path const& filepath) {
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

      // id string to size_t
      std::stringstream id_stream{id_string};
      size_t            id;
      id_stream >> id;
      n->set_id(id);
      auto const& input_pin_ids = *serialized_node["input_pin_ids"].as_array();
      auto const& output_pin_ids =
          *serialized_node["output_pin_ids"].as_array();
      size_t i = 0;
      for (auto& input_pin : n->input_pins()) {
        input_pin.set_id(size_t(input_pin_ids[i++].as_integer()->get()));
      }
      i = 0;
      for (auto& output_pin : n->output_pins()) {
        output_pin.set_id(size_t(output_pin_ids[i++].as_integer()->get()));
      }

      // set node position
      auto const x = (*serialized_node["node_position"].as_array())[0]
                         .as_floating_point()
                         ->get();
      auto const y = (*serialized_node["node_position"].as_array())[1]
                         .as_floating_point()
                         ->get();
      ImVec2 pos{static_cast<float>(x), static_cast<float>(y)};
      ax::NodeEditor::SetNodePosition(id, pos);

      // set title
      auto const title = serialized_node["node_title"].as_string()->get();
      n->set_title(title);

      // enable or disable
      auto const enabled = serialized_node["enabled"].as_boolean()->get();
      n->enable(enabled);

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
      auto&        input_pin  = *find_pin(input_id);
      auto&        output_pin = *find_pin(output_id);
      m_links.push_back(ui::link{id, input_pin, output_pin});
      input_pin.node().on_pin_connected(input_pin, output_pin);
      output_pin.node().on_pin_connected(output_pin, input_pin);
    }
  }
  ax::NodeEditor::SetCurrentEditor(nullptr);
}
//------------------------------------------------------------------------------
void scene::open_file(std::filesystem::path const& filepath) {
  auto const ext = filepath.extension().string();
  if (ext == ".toml" || ext == "toml" ||
      ext == ".scene" || ext == "scene") {
    read(filepath);
  } else if (ext == ".vtk" || ext == "vtk") {
    // TODO add vtk reader
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
