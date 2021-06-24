#include <tatooine/gl/bezier_widget.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui_internal.h>
//==============================================================================
namespace ImGui {
//==============================================================================
auto BezierValue(float dt01, float v0[4], float v1[4]) -> float {
  auto const steps = 256;
  ImVec2 Q[4] = {
      {v0[0], v0[1]}, {v0[2], v0[3]}, {v1[0], v1[1]}, {v1[2], v1[3]}};
  ImVec2 results[steps + 1];
  BezierTable<steps>(Q, results);
  return results[(int)((dt01 < 0 ? 0 : dt01 > 1 ? 1 : dt01) * 256)].y;
}
//------------------------------------------------------------------------------
auto Bezier(const char* label, float v0[4], float v1[4]) -> int {
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

  // int changed = SliderFloat4(label, P, 0, 1, "%.3f", 1.0f);
  int changed = 0;
  //int hovered = IsItemActive() || IsItemHovered();  // IsItemDragged() ?
  Dummy(ImVec2(0, 3));

  // prepare canvas
  const float avail = GetContentRegionAvailWidth();
  const float dim   = ImMin(avail, 128.f);
  ImVec2      Canvas(dim, dim);

  ImRect bb(Window->DC.CursorPos, Window->DC.CursorPos + Canvas);

  RenderFrame(bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBg, 1), true,
              Style.FrameRounding);

  // background grid
  for (int i = 0; i <= Canvas.x; i += (Canvas.x / 4)) {
    DrawList->AddLine(ImVec2(bb.Min.x + i, bb.Min.y),
                      ImVec2(bb.Min.x + i, bb.Max.y),
                      GetColorU32(ImGuiCol_TextDisabled));
  }
  for (int i = 0; i <= Canvas.y; i += (Canvas.y / 4)) {
    DrawList->AddLine(ImVec2(bb.Min.x, bb.Min.y + i),
                      ImVec2(bb.Max.x, bb.Min.y + i),
                      GetColorU32(ImGuiCol_TextDisabled));
  }

  // eval curve
  ImVec2 Q[4] = {
      {v0[0], v0[1]}, {v0[2], v0[3]}, {v1[0], v1[1]}, {v1[2], v1[3]}};
  ImVec2 results[smoothness + 1];
  BezierTable<smoothness>(Q, results);

  // control points: 2 lines and 2 circles
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
        v[1] -= GetIO().MouseDelta.y / Canvas.y;
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
        v[0] += GetIO().MouseDelta.x / Canvas.x;
        v[1] -= GetIO().MouseDelta.y / Canvas.y;
        v[0]    = std::max<float>(0.0f, std::min<float>(1.0f, v[0]));
        v[1]    = std::max<float>(0.0f, std::min<float>(1.0f, v[1]));
        changed = true;
      }
    };
    handle_grabber(v0 + 2);
    handle_grabber(v1 + 2);
    handle_grabber_fixed(v0);
    handle_grabber_fixed(v1);

    // draw curve
    {
      ImColor color(GetStyle().Colors[ImGuiCol_PlotLines]);
      for (size_t i = 0; i < smoothness; ++i) {
        ImVec2 p = {results[i + 0].x, 1 - results[i + 0].y};
        ImVec2 q = {results[i + 1].x, 1 - results[i + 1].y};
        ImVec2 r(p.x * (bb.Max.x - bb.Min.x) + bb.Min.x,
                 p.y * (bb.Max.y - bb.Min.y) + bb.Min.y);
        ImVec2 s(q.x * (bb.Max.x - bb.Min.x) + bb.Min.x,
                 q.y * (bb.Max.y - bb.Min.y) + bb.Min.y);
        DrawList->AddLine(r, s, color, curve_width);
      }
    }

    // draw lines and grabbers
    float  luma = IsItemActive() || IsItemHovered() ? 0.5f : 1.0f;
    ImVec4 pink(1.00f, 0.00f, 0.75f, luma), cyan(0.00f, 0.75f, 1.00f, luma);
    ImVec4 white(GetStyle().Colors[ImGuiCol_Text]);
    ImVec2 p00 = ImVec2(v0[0], 1 - v0[1]) * (bb.Max - bb.Min) + bb.Min;
    ImVec2 p01 = ImVec2(v0[2], 1 - v0[3]) * (bb.Max - bb.Min) + bb.Min;
    ImVec2 p10 = ImVec2(v1[0], 1 - v1[1]) * (bb.Max - bb.Min) + bb.Min;
    ImVec2 p11 = ImVec2(v1[2], 1 - v1[3]) * (bb.Max - bb.Min) + bb.Min;
    DrawList->AddLine(p00, p01, ImColor(white), line_width);
    DrawList->AddLine(p10, p11, ImColor(white), line_width);
    DrawList->AddCircleFilled(p00, grab_radius, ImColor(white));
    DrawList->AddCircleFilled(p00, grab_radius - grab_border, ImColor(pink));
    DrawList->AddCircleFilled(p01, grab_radius, ImColor(white));
    DrawList->AddCircleFilled(p01, grab_radius - grab_border, ImColor(pink));
    DrawList->AddCircleFilled(p10, grab_radius, ImColor(white));
    DrawList->AddCircleFilled(p10, grab_radius - grab_border, ImColor(cyan));
    DrawList->AddCircleFilled(p11, grab_radius, ImColor(white));
    DrawList->AddCircleFilled(p11, grab_radius - grab_border, ImColor(cyan));

    // restore cursor pos
    SetCursorScreenPos(ImVec2(bb.Min.x, bb.Max.y + grab_radius));  // :P
  }

  return changed;
}
//==============================================================================
}  // namespace ImGui
//==============================================================================
