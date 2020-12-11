#ifndef TATOOINE_FLOWEXPLORER_UI_DRAW_ICON_H
#define TATOOINE_FLOWEXPLORER_UI_DRAW_ICON_H
//==============================================================================
#include <yavin/imgui.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
enum class icon_type : ImU32 {
  flow,
  circle,
  square,
  grid,
  round_square,
  diamond
};
//==============================================================================
auto icon(const ImVec2& size, icon_type type, bool filled,
          const ImVec4& color      = ImVec4(1, 1, 1, 1),
          const ImVec4& innerColor = ImVec4(0, 0, 0, 0)) -> void;
//------------------------------------------------------------------------------
auto draw_icon(ImDrawList* drawList, const ImVec2& a, const ImVec2& b,
               icon_type type, bool filled, ImU32 color, ImU32 innerColor)
    -> void;
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#endif
