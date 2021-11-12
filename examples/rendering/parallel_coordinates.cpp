#include <tatooine/rendering/first_person_window.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
namespace ImGui {
//==============================================================================
auto parallel_coordinates(const char*                             label,
                          std::vector<std::string> const&         names,
                          std::vector<std::vector<double>> const& data) -> int {
  // visuals
  size_t const smoothness = 64;  // curve smoothness: the higher number of
                                 // segments, the smoother curve
  size_t const curve_width = 4;  // main curved line width
  size_t const line_width  = 1;  // handlers: small lines width
  size_t const grab_radius = 6;  // handlers: circle radius
  size_t const grab_border = 2;  // handlers: circle border width

  const ImGuiStyle& Style    = GetStyle();
  ImDrawList*       DrawList = GetWindowDrawList();
  ImGuiWindow*      Window   = GetCurrentWindow();
  if (Window->SkipItems) return false;

  int changed = 0;
  //Dummy(ImVec2(0, 3));

  // prepare canvas
  float const width = 100;
  auto const  canvas =
      ImVec2{names.size() * width, ImGui::GetContentRegionAvail().y};
  auto const bb = ImRect{Window->DC.CursorPos, Window->DC.CursorPos + canvas};
  RenderFrame(bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBg, 1), true,
              Style.FrameRounding);

  for (auto const& point:data) {
    for (size_t i = 0; i < size(point)-1; ++i) {
      DrawList->AddLine(
          ImVec2(bb.Min.x + i * width, bb.Min.y + canvas.y * point[i]),
          ImVec2(bb.Min.x + (i + 1) * width,
                 bb.Min.y + canvas.y * point[i + 1]),
          GetColorU32(ImGuiCol_TextDisabled));
    }
  }
  //for (int i = 0; i <= canvas.y; i += (canvas.y / 4)) {
  //  DrawList->AddLine(ImVec2(bb.Min.x, bb.Min.y + i),
  //                    ImVec2(bb.Max.x, bb.Min.y + i),
  //                    GetColorU32(ImGuiCol_TextDisabled));
  //}


  {
    char buf[128];
    sprintf(buf, "0##%s", label);

    // handle grabbers
    auto handle_grabber_fixed = [&](float* v) {
      ImVec2 pos = ImVec2(v[0], 1 - v[1]) * (bb.Max - bb.Min) + bb.Min;
      SetCursorScreenPos(pos - ImVec2(grab_radius, grab_radius));
      InvisibleButton((buf[0]++, buf),
                      ImVec2(2 * grab_radius, 2 * grab_radius));
      if (IsItemActive() || IsItemHovered()) {
        SetTooltip("(%4.3f, %4.3f)", v[0], v[1]);
      }
      if (IsItemActive() && IsMouseDragging(0)) {
        v[1] -= GetIO().MouseDelta.y / canvas.y;
        v[1] = std::max<float>(0.0f, std::min<float>(1.0f, v[1]));
        changed = true;
      }
    };
    auto handle_grabber = [&](float* v) {
      ImVec2 pos = ImVec2(v[0], 1 - v[1]) * (bb.Max - bb.Min) + bb.Min;
      SetCursorScreenPos(pos - ImVec2(grab_radius, grab_radius));
      InvisibleButton((buf[0]++, buf),
                      ImVec2(2 * grab_radius, 2 * grab_radius));
      if (IsItemActive() || IsItemHovered()) {
        SetTooltip("(%4.3f, %4.3f)", v[0], v[1]);
      }
      if (IsItemActive() && IsMouseDragging(0)) {
        v[0] += GetIO().MouseDelta.x / canvas.x;
        v[1] -= GetIO().MouseDelta.y / canvas.y;
        v[0]    = std::max<float>(0.0f, std::min<float>(1.0f, v[0]));
        v[1]    = std::max<float>(0.0f, std::min<float>(1.0f, v[1]));
        changed = true;
      }
    };


    // draw lines and grabbers
    float  luma = IsItemActive() || IsItemHovered() ? 0.5f : 1.0f;
    ImVec4 cyan(0.00f, 0.75f, 1.00f, luma);
    ImVec4 white(GetStyle().Colors[ImGuiCol_Text]);
    //for (size_t i = 0; i < num_handles; ++i) {
    //  ImVec2 p00 =
    //      ImVec2(handles[i * 4], 1 - handles[i * 4 + 1]) * (bb.Max - bb.Min) +
    //      bb.Min;
    //  ImVec2 p01 = ImVec2(handles[i * 4 + 2], 1 - handles[i * 4 + 3]) *
    //                   (bb.Max - bb.Min) +
    //               bb.Min;
    //  DrawList->AddLine(p00, p01, ImColor(white), line_width);
    //  DrawList->AddCircleFilled(p00, grab_radius, ImColor(white));
    //  DrawList->AddCircleFilled(p00, grab_radius - grab_border, ImColor(cyan));
    //  DrawList->AddCircleFilled(p01, grab_radius, ImColor(white));
    //  DrawList->AddCircleFilled(p01, grab_radius - grab_border, ImColor(cyan));
    //}

    // restore cursor pos
    SetCursorScreenPos(ImVec2(bb.Min.x, bb.Max.y + grab_radius));  // :P
  }

  return changed;
}
//==============================================================================
}  // namespace ImGui
//==============================================================================
auto win = std::unique_ptr<rendering::first_person_window>{};
//==============================================================================
auto render_loop(auto const& dt) {
  gl::clear_color_depth_buffer();
  ImGui::parallel_coordinates(
      "##pc", std::vector<std::string>{"abc", "def"},
      std::vector{std::vector{0.0, 1.0}, std::vector{0.5, 0.0},
                  std::vector{1.0, 0.0}});
}
//==============================================================================
auto main() -> int {
  auto win = std::make_unique<rendering::first_person_window>(800, 600);
  gl::clear_color(255, 255, 255, 255);
  win->render_loop([&](auto const& dt) { render_loop(dt); });
}
