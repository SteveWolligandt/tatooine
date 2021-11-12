#include <tatooine/rendering/first_person_window.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
namespace ImGui {
//==============================================================================
auto parallel_coordinates(const char*                             imgui_label,
                          std::vector<std::string> const&         labels,
                          std::vector<std::vector<double>> const& data,
                          std::vector<std::pair<double, double>>& ranges)
    -> int {

  auto              active_bar = std::numeric_limits<std::size_t>::max();
  const ImGuiStyle& Style      = GetStyle();
  ImDrawList*       DrawList   = GetWindowDrawList();
  ImGuiWindow*      Window     = GetCurrentWindow();
  if (Window->SkipItems) {
    return false;
  }

  int changed = 0;

  // prepare canvas
  float const space_between_bars = 200;
  float const bar_width          = 10;
  auto const  canvas = ImVec2{(labels.size() - 1) * space_between_bars,
                             ImGui::GetContentRegionAvail().y - 100};
  auto const  bb = ImRect{Window->DC.CursorPos, Window->DC.CursorPos + canvas};
  auto        extent = bb.Max - bb.Min;
  // interaction
  {
    char buf[128];
    sprintf(buf, "0##%s", imgui_label);
    size_t const grab_radius = 12;  // handlers: circle radius
    for (std::size_t i = 0; i < size(labels); ++i) {
      // handle bottom grabber of bar
      {
        ImVec2 pos = ImVec2(bb.Min.x + space_between_bars * i,
                            bb.Min.y + ranges[i].first * extent.y);
        SetCursorScreenPos(pos - ImVec2(grab_radius, grab_radius));
        InvisibleButton((buf[0]++, buf),
                        ImVec2(2 * grab_radius, 2 * grab_radius));
        if (IsItemActive() || IsItemHovered()) {
          active_bar = i;
        }
        if (IsItemActive() && IsMouseDragging(0)) {
          ranges[i].first += GetIO().MouseDelta.y / canvas.y;
          ranges[i].first =
              std::max<float>(0.0f, std::min<float>(1.0f, ranges[i].first));
          changed = true;
        }
      }
      // handle top grabber of bar
      {
        ImVec2 pos = ImVec2(bb.Min.x + space_between_bars * i,
                            bb.Min.y + ranges[i].second * extent.y);
        SetCursorScreenPos(pos - ImVec2(grab_radius, grab_radius));
        InvisibleButton((buf[0]++, buf),
                        ImVec2(2 * grab_radius, 2 * grab_radius));
        if (IsItemActive() || IsItemHovered()) {
          active_bar = i;
        }
        if (IsItemActive() && IsMouseDragging(0)) {
          ranges[i].second += GetIO().MouseDelta.y / canvas.y;
          ranges[i].second =
              std::max<float>(0.0f, std::min<float>(1.0f, ranges[i].second));
          changed = true;
        }
      }
    }
  }

  // drawing
  RenderFrame(bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBg, 1), true,
              Style.FrameRounding);

  auto const data_line_col =
      ColorConvertFloat4ToU32(ImVec4{1.0f, 1.0f, 1.0f, 0.3f});
  auto const data_line_outside_col =
      ColorConvertFloat4ToU32(ImVec4{1.0f, 1.0f, 1.0f, 0.01f});

  // draw data lines
  for (auto const& point : data) {
    bool inside = true;
    for (size_t i = 0; i < size(point); ++i) {
      if (point[i] < ranges[i].first || point[i] > ranges[i].second) {
        inside = false;
        break;
      }
    }
    for (size_t i = 0; i < size(point) - 1; ++i) {
      DrawList->AddLine(ImVec2(bb.Min.x + i * space_between_bars,
                               bb.Min.y + extent.y * point[i]),
                        ImVec2(bb.Min.x + (i + 1) * space_between_bars,
                               bb.Min.y + extent.y * point[i + 1]),
                        inside ? data_line_col : data_line_outside_col);
    }
  }

  SetCursorScreenPos(ImVec2(bb.Min.x, bb.Max.y));
  for (std::size_t i = 0; i < size(labels); ++i) {
    float base   = bb.Min.y + ranges[i].first * extent.y;
    float height = (ranges[i].second - ranges[i].first) * extent.y;
    DrawList->AddRectFilled(
        ImVec2(bb.Min.x + i * space_between_bars - bar_width, base),
        ImVec2(bb.Min.x + i * space_between_bars + bar_width, base + height),
        ColorConvertFloat4ToU32(
            ImVec4{1.0f, 1.0f, 1.0f, active_bar == i ? 0.8f : 0.5f}));
  }
  SetCursorScreenPos(ImVec2(bb.Min.x, bb.Max.y + 10));
  Text(labels.front().c_str());
  for (std::size_t i = 1; i < size(labels); ++i) {
    SameLine(i * space_between_bars);
    Text(labels[i].c_str());
  }

  return changed;
}
//==============================================================================
}  // namespace ImGui
//==============================================================================
auto win = std::unique_ptr<rendering::first_person_window>{};
//==============================================================================
std::vector<std::vector<double>>       data;
std::vector<std::string>               labels;
std::vector<std::pair<double, double>> ranges;
auto                                   render_loop(auto const& dt) {
  gl::clear_color_depth_buffer();
  ImGui::parallel_coordinates("pc", labels, data, ranges);
}
//==============================================================================
auto main() -> int {
  auto win = std::make_unique<rendering::first_person_window>(800, 600);
  gl::clear_color(255, 255, 255, 255);
  auto rand = random::uniform{0.0, 1.0};
  labels.emplace_back("x");
  labels.emplace_back("y");
  labels.emplace_back("z");
  labels.emplace_back("1");
  labels.emplace_back("2");
  labels.emplace_back("3");
  labels.emplace_back("4");
  for (std::size_t i = 0; i < labels.size(); ++i) {
    ranges.emplace_back(0, 1);
  }
  for (std::size_t j = 0; j < 1000; ++j) {
    auto& col = data.emplace_back();
    for (std::size_t i = 0; i < labels.size(); ++i) {
      col.push_back(rand());
    }
  }
  win->render_loop([&](auto const& dt) { render_loop(dt); });
}
