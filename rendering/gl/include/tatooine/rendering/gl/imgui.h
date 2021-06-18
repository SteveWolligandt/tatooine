#ifndef YAVIN_IMGUI_H
#define YAVIN_IMGUI_H
//==============================================================================
#include <yavin/dllexport.h>
#include <yavin/imgui_includes.h>
//==============================================================================
namespace ImGui {
//==============================================================================
DLL_API bool InputDouble2(const char* label, double v[2],
                          const char*         format = "%.3lf",
                          ImGuiInputTextFlags flags  = 0);
DLL_API bool InputDouble3(const char* label, double v[3],
                          const char*         format = "%.3lf",
                          ImGuiInputTextFlags flags  = 0);
DLL_API bool InputDouble4(const char* label, double v[4],
                          const char*         format = "%.3lf",
                          ImGuiInputTextFlags flags  = 0);

DLL_API bool DragDouble(const char* label, double* v, double v_speed = 1.0,
                        double v_min = 0.0, double v_max = 0.0,
                        const char* format = "%.3lf", float power = 1.0);
DLL_API bool DragDouble2(const char* label, double v[2], double v_speed = 1.0,
                         double v_min = 0.0, double v_max = 0.0,
                         const char* format = "%.3lf", float power = 1.0);
DLL_API bool DragDouble3(const char* label, double v[3], double v_speed = 1.0,
                         double v_min = 0.0, double v_max = 0.0,
                         const char* format = "%.3lf", float power = 1.0);
DLL_API bool DragDouble4(const char* label, double v[4], double v_speed = 1.0,
                         double v_min = 0.0, double v_max = 0.0,
                         const char* format = "%.3lf", float power = 1.0);
DLL_API bool BufferingBar(const char* label, float value,
                          const ImVec2& size_arg, const ImU32& bg_col,
                          const ImU32& fg_col);
DLL_API bool Spinner(const char* label, float radius, int thickness,
                     const ImU32& color);
////------------------------------------------------------------------------------
//// layouting
////------------------------------------------------------------------------------
//enum ImGuiLayoutType_ {
//  ImGuiLayoutType_Horizontal = 0,
//  ImGuiLayoutType_Vertical   = 1
//};
//
//enum ImGuiLayoutItemType_ {
//  ImGuiLayoutItemType_Item,
//  ImGuiLayoutItemType_Spring
//};
//typedef int ImGuiLayoutType;      // -> enum ImGuiLayoutType_         // Enum:
//                                  // Horizontal or vertical
//typedef int ImGuiLayoutItemType;  // -> enum ImGuiLayoutItemType_    // Enum:
//                                  // Item or Spring
//struct ImGuiLayoutItem {
//  ImGuiLayoutItemType Type;  // Type of an item
//  ImRect              MeasuredBounds;
//
//  float SpringWeight;   // Weight of a spring
//  float SpringSpacing;  // Spring spacing
//  float SpringSize;     // Calculated spring size
//
//  float CurrentAlign;
//  float CurrentAlignOffset;
//
//  unsigned int VertexIndexBegin;
//  unsigned int VertexIndexEnd;
//
//  ImGuiLayoutItem(ImGuiLayoutItemType type) {
//    Type           = type;
//    MeasuredBounds = ImRect(
//        0, 0, 0, 0);  // FIXME: @thedmd are you sure the default ImRect value
//                      // FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX aren't enough here?
//    SpringWeight       = 1.0f;
//    SpringSpacing      = -1.0f;
//    SpringSize         = 0.0f;
//    CurrentAlign       = 0.0f;
//    CurrentAlignOffset = 0.0f;
//    VertexIndexBegin = VertexIndexEnd = (ImDrawIdx)0;
//  }
//};
//
//struct ImGuiLayout {
//  ImGuiID         Id;
//  ImGuiLayoutType Type;
//  bool            Live;
//  ImVec2          Size;  // Size passed to BeginLayout
//  ImVec2 CurrentSize;    // Bounds of layout known at the beginning the frame.
//  ImVec2 MinimumSize;    // Minimum possible size when springs are collapsed.
//  ImVec2 MeasuredSize;   // Measured size with springs expanded.
//
//  ImVector<ImGuiLayoutItem> Items;
//  int                       CurrentItemIndex;
//  int                       ParentItemIndex;
//  ImGuiLayout*              Parent;
//  ImGuiLayout*              FirstChild;
//  ImGuiLayout*              NextSibling;
//  float                     Align;  // Current item alignment.
//  float  Indent;    // Indent used to align items in vertical layout.
//  ImVec2 StartPos;  // Initial cursor position when BeginLayout is called.
//  ImVec2
//      StartCursorMaxPos;  // Maximum cursor position when BeginLayout is called.
//
//  ImGuiLayout(ImGuiID id, ImGuiLayoutType type) {
//    Id   = id;
//    Type = type;
//    Live = false;
//    Size = CurrentSize = MinimumSize = MeasuredSize = ImVec2(0, 0);
//    CurrentItemIndex                                = 0;
//    ParentItemIndex                                 = 0;
//    Parent = FirstChild = NextSibling = NULL;
//    Align                             = -1.0f;
//    Indent                            = 0.0f;
//    StartPos                          = ImVec2(0, 0);
//    StartCursorMaxPos                 = ImVec2(0, 0);
//  }
//};
//DLL_API ImGuiLayout* FindLayout(ImGuiID id, ImGuiLayoutType type);
//DLL_API ImGuiLayout* CreateNewLayout(ImGuiID id, ImGuiLayoutType type,
//                                     ImVec2 size);
//DLL_API void         BeginLayout(ImGuiID id, ImGuiLayoutType type, ImVec2 size,
//                                 float align);
//DLL_API void         EndLayout(ImGuiLayoutType type);
//DLL_API void         PushLayout(ImGuiLayout* layout);
//DLL_API void         PopLayout(ImGuiLayout* layout);
//DLL_API void         BalanceLayoutSprings(ImGuiLayout& layout);
//DLL_API ImVec2       BalanceLayoutItemAlignment(ImGuiLayout&     layout,
//                                                ImGuiLayoutItem& item);
//DLL_API void         BalanceLayoutItemsAlignment(ImGuiLayout& layout);
//DLL_API void         BalanceChildLayouts(ImGuiLayout& layout);
//DLL_API ImVec2 CalculateLayoutSize(ImGuiLayout& layout, bool collapse_springs);
//DLL_API ImGuiLayoutItem* GenerateLayoutItem(ImGuiLayout&        layout,
//                                            ImGuiLayoutItemType type);
//DLL_API float            CalculateLayoutItemAlignmentOffset(ImGuiLayout&     layout,
//                                                            ImGuiLayoutItem& item);
//DLL_API void TranslateLayoutItem(ImGuiLayoutItem& item, const ImVec2& offset);
//DLL_API void BeginLayoutItem(ImGuiLayout& layout);
//DLL_API void EndLayoutItem(ImGuiLayout& layout);
//DLL_API void AddLayoutSpring(ImGuiLayout& layout, float weight, float spacing);
//DLL_API void SignedIndent(float indent);
//
//DLL_API void BeginHorizontal(const char*   str_id,
//                             const ImVec2& size  = ImVec2(0, 0),
//                             float         align = -1.0f);
//DLL_API void BeginHorizontal(const void*   ptr_id,
//                             const ImVec2& size  = ImVec2(0, 0),
//                             float         align = -1.0f);
//DLL_API void BeginHorizontal(int id, const ImVec2& size = ImVec2(0, 0),
//                             float align = -1);
//DLL_API void EndHorizontal();
//DLL_API void BeginVertical(const char*   str_id,
//                           const ImVec2& size  = ImVec2(0, 0),
//                           float         align = -1.0f);
//DLL_API void BeginVertical(const void*   ptr_id,
//                           const ImVec2& size  = ImVec2(0, 0),
//                           float         align = -1.0f);
//DLL_API void BeginVertical(int id, const ImVec2& size = ImVec2(0, 0),
//                           float align = -1);
//DLL_API void EndVertical();
//DLL_API void Spring(float weight = 1.0f, float spacing = -1.0f);
////------------------------------------------------------------------------------
//inline auto layout_align() -> float& {
//  static float l = 0.5f;
//  return l;
//}
//inline auto layout_type() -> auto& {
//  static ImGuiLayoutType LayoutType;
//  return LayoutType;
//}
//inline auto parent_layout_type() -> auto& {
//  static ImGuiLayoutType ParentLayoutType;
//  return ParentLayoutType;
//}
//inline auto current_layout() -> auto& {
//  static ImGuiLayout* CurrentLayout = nullptr;
//  return CurrentLayout;
//}
//inline auto current_layout_item() -> auto& {
//  static ImGuiLayoutItem* CurrentLayoutItem = nullptr;
//  return CurrentLayoutItem;
//}
//inline auto layout_stack() -> auto& {
//  static ImVector<ImGuiLayout*> LayoutStack;
//  return LayoutStack;
//}
//inline auto layouts() -> auto& {
//  static ImGuiStorage Layouts;
//  return Layouts;
//}
//==============================================================================
}  // namespace ImGui
//==============================================================================
#include <yavin/imgui_api_backend.h>
#include <yavin/imgui_render_backend.h>
//==============================================================================
#endif
