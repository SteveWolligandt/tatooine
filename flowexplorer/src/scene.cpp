#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/boundingbox.h>
#include <tatooine/demangling.h>
#include <tatooine/flowexplorer/nodes/abcflow.h>
#include <tatooine/flowexplorer/scene.h>
#include <toml++/toml.h>
#include <fstream>

#include <yavin>
//#include <tatooine/flowexplorer/nodes/autonomous_particle.h>
//#include <tatooine/flowexplorer/nodes/boundingbox.h>
//#include <tatooine/flowexplorer/nodes/position.h>
//#include <tatooine/flowexplorer/nodes/doublegyre.h>
//#include <tatooine/flowexplorer/nodes/saddle.h>
//#include <tatooine/flowexplorer/nodes/duffing_oscillator.h>
//#include <tatooine/flowexplorer/nodes/lic.h>
//#include <tatooine/flowexplorer/nodes/random_pathlines.h>
//#include <tatooine/flowexplorer/nodes/rayleigh_benard_convection.h>
//#include <tatooine/flowexplorer/nodes/spacetime_vectorfield.h>
#include <tatooine/interpolation.h>
#include <tatooine/rendering/yavin_interop.h>
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
scene::scene(rendering::camera_controller<float>& cam) : m_cam{&cam} {
  m_node_editor_context = ax::NodeEditor::CreateEditor();
}
//----------------------------------------------------------------------------
scene::~scene() {
  ax::NodeEditor::DestroyEditor(m_node_editor_context);
}
//------------------------------------------------------------------------------
void scene::render(std::chrono::duration<double> const& dt) {
  yavin::gl::clear_color(255, 255, 255, 255);
  yavin::clear_color_depth_buffer();
  for (auto& r : m_renderables) {
    r->update(dt);
  }

  // render non-transparent objects
  yavin::enable_depth_write();
  yavin::disable_blending();
  for (auto& r : m_renderables) {
    if (!r->is_transparent()) {
      r->render(m_cam->projection_matrix(), m_cam->view_matrix());
    }
  }

  // render transparent objects
  yavin::disable_depth_write();
  yavin::enable_blending();
  yavin::blend_func_alpha();
  for (auto& r : m_renderables) {
    if (r->is_transparent()) {
      r->render(m_cam->projection_matrix(), m_cam->view_matrix());
    }
  }
  yavin::enable_depth_test();
  yavin::enable_depth_write();
}
//----------------------------------------------------------------------------
auto scene::find_node(ax::NodeEditor::NodeId id) -> base_renderable* {
  for (auto& r : m_renderables) {
    if (r->id() == id) {
      return r.get();
    }
  }
  return nullptr;
}
//----------------------------------------------------------------------------
auto scene::find_pin(ax::NodeEditor::PinId id) -> ui::pin* {
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
  for (auto& n : m_nodes) {
    for (auto& p : n->input_pins()) {
      if (p.id() == id) {
        return &p;
      }
    }
    for (auto& p : n->output_pins()) {
      if (p.id() == id) {
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

  auto draw_ui = [&i](auto&& f, auto const& node) {
    ImGui::PushID(i++);
    namespace ed = ax::NodeEditor;
    ed::BeginNode(node->id());
    ImGui::TextUnformatted(node->name().c_str());
    f();
    for (auto& input_pin : node->input_pins()) {
      ed::BeginPin(input_pin.id(), ed::PinKind::Input);
      std::string in = "-> " + input_pin.name();
      ImGui::TextUnformatted(in.c_str());
      ed::EndPin();
    }
    for (auto& output_pin : node->output_pins()) {
      ed::BeginPin(output_pin.id(), ed::PinKind::Output);
      std::string out = output_pin.name() + " ->";
      ImGui::TextUnformatted(out.c_str());
      ed::EndPin();
    }
    ed::EndNode();
    ImGui::PopID();
  };

  for (auto& n : m_nodes) {       draw_ui([&n] { n->draw_ui(); }, n); }
  for (auto& r : m_renderables) { draw_ui([&r] { r->draw_ui(); }, r); }
}
//----------------------------------------------------------------------------
void scene::draw_links() {
  namespace ed = ax::NodeEditor;
  for (auto& link_info : m_links) {
    ed::Link(link_info.id, link_info.input_id, link_info.output_id);
  }
}
//----------------------------------------------------------------------------
void scene::create_link() {
  namespace ed = ax::NodeEditor;
  if (ed::BeginCreate()) {
    ed::PinId input_pin_id, output_pin_id;
    if (ed::QueryNewLink(&input_pin_id, &output_pin_id)) {
      if (input_pin_id && output_pin_id) {  // both are valid, let's accept link

        ui::pin* input_pin  = find_pin(input_pin_id);
        ui::pin* output_pin = find_pin(output_pin_id);

        if (input_pin->kind() == ui::pinkind::output) {
          std::swap(input_pin, output_pin);
          std::swap(input_pin_id, output_pin_id);
        }

        if (input_pin->node().id() == output_pin->node().id()) {
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
          for (auto& present_link : m_links) {
            ui::pin* present_input_pin = find_pin(present_link.input_id);
            if (present_input_pin->id() == input_pin->id()) {
              ui::pin* present_output_pin = find_pin(present_link.output_id);
              present_input_pin->node().on_pin_disconnected(*present_input_pin);
              present_output_pin->node().on_pin_disconnected(
                  *present_output_pin);
              m_links.erase(&present_link);
              break;
            }
          }

          input_pin->node().on_pin_connected(*input_pin, *output_pin);
          output_pin->node().on_pin_connected(*output_pin, *input_pin);
          // Since we accepted new link, lets add one to our list of links.
          m_links.push_back(
              {ed::LinkId(m_next_link++), input_pin_id, output_pin_id});

          // Draw new link.
          ed::Link(m_links.back().id, m_links.back().input_id,
                   m_links.back().output_id);
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
    }
  }
  ed::EndDelete();
}
//----------------------------------------------------------------------------
void scene::draw_node_editor(size_t const pos_x, size_t const pos_y,
                             size_t const width, size_t const height, 
                             bool show) {
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
  //if (ImGui::Button("2D Position")) {
  //  m_renderables.emplace_back(new nodes::position<2>{*this});
  //}
  //ImGui::SameLine();
  //if (ImGui::Button("3D Position")) {
  //  m_renderables.emplace_back(new nodes::position<3>{*this});
  //}
  // vectorfields
  if (ImGui::Button("ABC Flow")) {
    m_nodes.emplace_back(new nodes::abcflow<double>{});
  }
  //ImGui::SameLine();
  //if (ImGui::Button("Rayleigh Benard Convection")) {
  //  m_nodes.emplace_back(new nodes::rayleigh_benard_convection<double>{});
  //}
  //if (ImGui::Button("Doublegyre Flow")) {
  //  m_nodes.emplace_back(new nodes::doublegyre<double>{});
  //}
  //if (ImGui::Button("Saddle Flow")) {
  //  m_nodes.emplace_back(new nodes::saddle<double>{});
  //}
  //ImGui::SameLine();
  //if (ImGui::Button("Duffing Oscillator Flow")) {
  //  m_nodes.emplace_back(new nodes::duffing_oscillator<double>{});
  //}
  //
  //// vectorfield operations
  //if (ImGui::Button("Spacetime Vector Field")) {
  //  m_nodes.emplace_back(new nodes::spacetime_vectorfield<double>{});
  //}
  //
  //// bounding boxes
  //if (ImGui::Button("2D BoundingBox")) {
  //  m_renderables.emplace_back(
  //      new nodes::boundingbox{*this, vec{-1.0, -1.0}, vec{1.0, 1.0}});
  //}
  //ImGui::SameLine();
  //if (ImGui::Button("3D BoundingBox")) {
  //  m_renderables.emplace_back(new nodes::boundingbox{
  //      *this, vec{-1.0, -1.0, -1.0}, vec{1.0, 1.0, 1.0}});
  //}
  //
  //// Algorithms
  //if (ImGui::Button("Random Path Lines")) {
  //  m_renderables.emplace_back(new nodes::random_pathlines<double, 3>{*this});
  //}
  //ImGui::SameLine();
  //if (ImGui::Button("LIC")) {
  //  m_renderables.emplace_back(new nodes::lic<double>{*this});
  //}
  //ImGui::SameLine();
  //if (ImGui::Button("Autonomous Particle")) {
  //  m_renderables.emplace_back(new nodes::autonomous_particle{*this});
  //}
}
//------------------------------------------------------------------------------
void scene::write(std::string const& filepath) const {
  toml::table data;

  for (auto const& node : m_nodes) {
    toml::table node_table;
    auto pos = node->node_position();
    node_table.insert("position", toml::array{pos[0], pos[1]});
    node_table.insert("type", node->node_type_name());
    data.insert(std::to_string(node->id().Get()), node_table);
  }

  std::ofstream fout {filepath};
  if (fout.is_open()) {
    fout << data << '\n';
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
