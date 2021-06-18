#include <yavin/imgui.h>
//==============================================================================
namespace ImGui {
//==============================================================================
bool InputDouble2(const char* label, double v[2], const char* format,
                  ImGuiInputTextFlags flags) {
  return InputScalarN(label, ImGuiDataType_Double, v, 2, nullptr, nullptr,
                      format, flags);
}
//------------------------------------------------------------------------------
bool InputDouble3(const char* label, double v[3], const char* format,
                  ImGuiInputTextFlags flags) {
  return InputScalarN(label, ImGuiDataType_Double, v, 3, nullptr, nullptr,
                      format, flags);
}
//------------------------------------------------------------------------------
bool InputDouble4(const char* label, double v[4], const char* format,
                  ImGuiInputTextFlags flags) {
  return InputScalarN(label, ImGuiDataType_Double, v, 4, nullptr, nullptr,
                      format, flags);
}
//------------------------------------------------------------------------------
bool DragDouble(const char* label, double* v, double v_speed, double v_min,
                double v_max, const char* format, float power) {
  return DragScalar(label, ImGuiDataType_Double, v, v_speed, &v_min, &v_max,
                    format, power);
}
//------------------------------------------------------------------------------
bool DragDouble2(const char* label, double v[2], double v_speed, double v_min,
                 double v_max, const char* format, float power) {
  return DragScalarN(label, ImGuiDataType_Double, v, 2, v_speed, &v_min, &v_max,
                     format, power);
}
//------------------------------------------------------------------------------
bool DragDouble3(const char* label, double v[3], double v_speed, double v_min,
                 double v_max, const char* format, float power) {
  return DragScalarN(label, ImGuiDataType_Double, v, 3, v_speed, &v_min, &v_max,
                     format, power);
}
//------------------------------------------------------------------------------
bool DragDouble4(const char* label, double v[4], double v_speed, double v_min,
                 double v_max, const char* format, float power) {
  return DragScalarN(label, ImGuiDataType_Double, v, 4, v_speed, &v_min, &v_max,
                     format, power);
}
//------------------------------------------------------------------------------
bool BufferingBar(const char* label, float value, const ImVec2& size_arg,
                  const ImU32& bg_col, const ImU32& fg_col) {
  ImGuiWindow* window = GetCurrentWindow();
  if (window->SkipItems) return false;

  ImGuiContext&     g     = *GImGui;
  const ImGuiStyle& style = g.Style;
  const ImGuiID     id    = window->GetID(label);

  ImVec2 pos  = window->DC.CursorPos;
  ImVec2 size = size_arg;
  size.x -= style.FramePadding.x * 2;

  const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
  ItemSize(bb, style.FramePadding.y);
  if (!ItemAdd(bb, id)) return false;

  // Render
  const float circleStart = size.x * 0.7f;
  const float circleEnd   = size.x;
  const float circleWidth = circleEnd - circleStart;

  window->DrawList->AddRectFilled(bb.Min, ImVec2(pos.x + circleStart, bb.Max.y),
                                  bg_col);
  window->DrawList->AddRectFilled(
      bb.Min, ImVec2(pos.x + circleStart * value, bb.Max.y), fg_col);

  const float t     = g.Time;
  const float r     = size.y / 2;
  const float speed = 1.5f;

  const float a = speed * 0;
  const float b = speed * 0.333f;
  const float c = speed * 0.666f;

  const float o1 =
      (circleWidth + r) * (t + a - speed * (int)((t + a) / speed)) / speed;
  const float o2 =
      (circleWidth + r) * (t + b - speed * (int)((t + b) / speed)) / speed;
  const float o3 =
      (circleWidth + r) * (t + c - speed * (int)((t + c) / speed)) / speed;

  window->DrawList->AddCircleFilled(
      ImVec2(pos.x + circleEnd - o1, bb.Min.y + r), r, bg_col);
  window->DrawList->AddCircleFilled(
      ImVec2(pos.x + circleEnd - o2, bb.Min.y + r), r, bg_col);
  window->DrawList->AddCircleFilled(
      ImVec2(pos.x + circleEnd - o3, bb.Min.y + r), r, bg_col);
  return true;
}
//------------------------------------------------------------------------------
bool Spinner(const char* label, float radius, int thickness,
             const ImU32& color) {
  ImGuiWindow* window = GetCurrentWindow();
  if (window->SkipItems) return false;

  ImGuiContext&     g     = *GImGui;
  const ImGuiStyle& style = g.Style;
  const ImGuiID     id    = window->GetID(label);

  ImVec2 pos = window->DC.CursorPos;
  ImVec2 size((radius)*2, (radius + style.FramePadding.y) * 2);

  const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
  ItemSize(bb, style.FramePadding.y);
  if (!ItemAdd(bb, id)) return false;

  // Render
  window->DrawList->PathClear();

  int num_segments = 30;
  int start        = abs(ImSin(g.Time * 1.8f) * (num_segments - 5));

  const float a_min = IM_PI * 2.0f * ((float)start) / (float)num_segments;
  const float a_max =
      IM_PI * 2.0f * ((float)num_segments - 3) / (float)num_segments;

  const ImVec2 centre =
      ImVec2(pos.x + radius, pos.y + radius + style.FramePadding.y);

  for (int i = 0; i < num_segments; i++) {
    const float a = a_min + ((float)i / (float)num_segments) * (a_max - a_min);
    window->DrawList->PathLineTo(
        ImVec2(centre.x + ImCos(a + g.Time * 8) * radius,
               centre.y + ImSin(a + g.Time * 8) * radius));
  }

  window->DrawList->PathStroke(color, false, thickness);
  return true;
}
//------------------------------------------------------------------------------
void StdStringNonStdResize(std::string& s, int size) {
  IM_ASSERT(size >= 0);
  const int oldLength = s.length();
  if (size < oldLength)
    s = s.substr(0, size);
  else if (size > oldLength)
    for (int i = 0, icnt = size - oldLength; i < icnt; i++)
      s += '\0';
}
////------------------------------------------------------------------------------
//// layouting
////------------------------------------------------------------------------------
//ImGuiLayout* FindLayout(ImGuiID id, ImGuiLayoutType type) {
//  IM_ASSERT(type == ImGuiLayoutType_Horizontal ||
//            type == ImGuiLayoutType_Vertical);
//
//  ImGuiLayout* layout = (ImGuiLayout*)layouts().GetVoidPtr(id);
//  if (!layout) return nullptr;
//
//  if (layout->Type != type) {
//    layout->Type        = type;
//    layout->MinimumSize = ImVec2(0.0f, 0.0f);
//    layout->Items.clear();
//  }
//
//  return layout;
//}
////------------------------------------------------------------------------------
//ImGuiLayout* CreateNewLayout(ImGuiID id, ImGuiLayoutType type, ImVec2 size) {
//  IM_ASSERT(type == ImGuiLayoutType_Horizontal ||
//            type == ImGuiLayoutType_Vertical);
//
//  ImGuiLayout* layout = IM_NEW(ImGuiLayout)(id, type);
//  layout->Size        = size;
//
//  layouts().SetVoidPtr(id, layout);
//
//  return layout;
//}
//void BeginLayout(ImGuiID id, ImGuiLayoutType type, ImVec2 size, float align) {
//  ImGuiWindow* window = GetCurrentWindow();
//
//  PushID(id);
//
//  // Find or create
//  ImGuiLayout* layout = FindLayout(id, type);
//  if (!layout) layout = CreateNewLayout(id, type, size);
//
//  layout->Live = true;
//
//  PushLayout(layout);
//
//  if (layout->Size.x != size.x || layout->Size.y != size.y) layout->Size = size;
//
//  if (align < 0.0f)
//    layout->Align = -1.0f;
//  else
//    layout->Align = ImClamp(align, 0.0f, 1.0f);
//
//  // Start capture
//  layout->CurrentItemIndex = 0;
//
//  layout->CurrentSize.x =
//      layout->Size.x > 0.0f ? layout->Size.x : layout->MinimumSize.x;
//  layout->CurrentSize.y =
//      layout->Size.y > 0.0f ? layout->Size.y : layout->MinimumSize.y;
//
//  layout->StartPos          = window->DC.CursorPos;
//  layout->StartCursorMaxPos = window->DC.CursorMaxPos;
//
//  if (type == ImGuiLayoutType_Vertical) {
//    // Push empty item to recalculate cursor position.
//    PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
//    Dummy(ImVec2(0.0f, 0.0f));
//    PopStyleVar();
//
//    // Indent horizontal position to match edge of the layout.
//    layout->Indent = layout->StartPos.x - window->DC.CursorPos.x;
//    SignedIndent(layout->Indent);
//  }
//
//  BeginLayoutItem(*layout);
//}
////------------------------------------------------------------------------------
//void EndLayout(ImGuiLayoutType type) {
//  ImGuiWindow* window = GetCurrentWindow();
//  IM_ASSERT(current_layout());
//  IM_ASSERT(current_layout()->Type == type);
//
//  ImGuiLayout* layout = current_layout();
//
//  EndLayoutItem(*layout);
//
//  if (layout->CurrentItemIndex < layout->Items.Size)
//    layout->Items.resize(layout->CurrentItemIndex);
//
//  if (layout->Type == ImGuiLayoutType_Vertical) SignedIndent(-layout->Indent);
//
//  PopLayout(layout);
//
//  const bool auto_width  = layout->Size.x <= 0.0f;
//  const bool auto_height = layout->Size.y <= 0.0f;
//
//  ImVec2 new_size = layout->Size;
//  if (auto_width) new_size.x = layout->CurrentSize.x;
//  if (auto_height) new_size.y = layout->CurrentSize.y;
//
//  ImVec2 new_minimum_size = CalculateLayoutSize(*layout, true);
//
//  if (new_minimum_size.x != layout->MinimumSize.x ||
//      new_minimum_size.y != layout->MinimumSize.y) {
//    layout->MinimumSize = new_minimum_size;
//
//    // Shrink
//    if (auto_width) new_size.x = new_minimum_size.x;
//    if (auto_height) new_size.y = new_minimum_size.y;
//  }
//
//  if (!auto_width) new_size.x = layout->Size.x;
//  if (!auto_height) new_size.y = layout->Size.y;
//
//  layout->CurrentSize = new_size;
//
//  ImVec2 measured_size = new_size;
//  if ((auto_width || auto_height) && layout->Parent) {
//    if (layout->Type == ImGuiLayoutType_Horizontal && auto_width &&
//        layout->Parent->CurrentSize.x > 0)
//      layout->CurrentSize.x = layout->Parent->CurrentSize.x;
//    else if (layout->Type == ImGuiLayoutType_Vertical && auto_height &&
//             layout->Parent->CurrentSize.y > 0)
//      layout->CurrentSize.y = layout->Parent->CurrentSize.y;
//
//    BalanceLayoutSprings(*layout);
//
//    measured_size = layout->CurrentSize;
//  }
//
//  layout->CurrentSize = new_size;
//
//  PopID();
//
//  ImVec2 current_layout_item_max = ImVec2(0.0f, 0.0f);
//  if (current_layout_item())
//    current_layout_item_max = ImMax(current_layout_item()->MeasuredBounds.Max,
//                                    layout->StartPos + new_size);
//
//  window->DC.CursorPos    = layout->StartPos;
//  window->DC.CursorMaxPos = layout->StartCursorMaxPos;
//  ItemSize(new_size);
//  ItemAdd(ImRect(layout->StartPos, layout->StartPos + measured_size), 0);
//
//  if (current_layout_item())
//    current_layout_item()->MeasuredBounds.Max = current_layout_item_max;
//
//  if (layout->Parent == nullptr) BalanceChildLayouts(*layout);
//
//  // window->DrawList->AddRect(layout->StartPos, layout->StartPos +
//  // measured_size, IM_COL32(0,255,0,255));           // [DEBUG]
//  // window->DrawList->AddRect(window->DC.LastItemRect.Min,
//  // window->DC.LastItemRect.Max, IM_COL32(255,255,0,255));   // [DEBUG]
//}
////------------------------------------------------------------------------------
//ImVec2 CalculateLayoutSize(ImGuiLayout& layout, bool collapse_springs) {
//  ImVec2 bounds = ImVec2(0.0f, 0.0f);
//
//  if (layout.Type == ImGuiLayoutType_Vertical) {
//    for (int i = 0; i < layout.Items.Size; i++) {
//      ImGuiLayoutItem& item      = layout.Items[i];
//      ImVec2           item_size = item.MeasuredBounds.GetSize();
//
//      if (item.Type == ImGuiLayoutItemType_Item) {
//        bounds.x = ImMax(bounds.x, item_size.x);
//        bounds.y += item_size.y;
//      } else {
//        bounds.y += ImFloor(item.SpringSpacing);
//
//        if (!collapse_springs) bounds.y += item.SpringSize;
//      }
//    }
//  } else {
//    for (int i = 0; i < layout.Items.Size; i++) {
//      ImGuiLayoutItem& item      = layout.Items[i];
//      ImVec2           item_size = item.MeasuredBounds.GetSize();
//
//      if (item.Type == ImGuiLayoutItemType_Item) {
//        bounds.x += item_size.x;
//        bounds.y = ImMax(bounds.y, item_size.y);
//      } else {
//        bounds.x += ImFloor(item.SpringSpacing);
//
//        if (!collapse_springs) bounds.x += item.SpringSize;
//      }
//    }
//  }
//
//  return bounds;
//}
////------------------------------------------------------------------------------
//void PushLayout(ImGuiLayout* layout) {
//  if (layout) {
//    layout->Parent = current_layout();
//    if (layout->Parent != nullptr)
//      layout->ParentItemIndex = layout->Parent->CurrentItemIndex;
//    if (current_layout()) {
//      layout->NextSibling          = current_layout()->FirstChild;
//      layout->FirstChild           = nullptr;
//      current_layout()->FirstChild = layout;
//    } else {
//      layout->NextSibling = nullptr;
//      layout->FirstChild  = nullptr;
//    }
//  }
//
//  layout_stack().push_back(layout);
//  current_layout()      = layout;
//  current_layout_item() = nullptr;
//}
////------------------------------------------------------------------------------
//void PopLayout(ImGuiLayout* layout) {
//  IM_ASSERT(!layout_stack().empty());
//  IM_ASSERT(layout_stack().back() == layout);
//
//  layout_stack().pop_back();
//
//  if (!layout_stack().empty()) {
//    current_layout() = layout_stack().back();
//    current_layout_item() =
//        &current_layout()->Items[current_layout()->CurrentItemIndex];
//  } else {
//    current_layout()      = nullptr;
//    current_layout_item() = nullptr;
//  }
//}
//void BalanceLayoutSprings(ImGuiLayout& layout) {
//  // Accumulate amount of occupied space and springs weights
//  float total_spring_weight = 0.0f;
//
//  int last_spring_item_index = -1;
//  for (int i = 0; i < layout.Items.Size; i++) {
//    ImGuiLayoutItem& item = layout.Items[i];
//    if (item.Type == ImGuiLayoutItemType_Spring) {
//      total_spring_weight += item.SpringWeight;
//      last_spring_item_index = i;
//    }
//  }
//
//  // Determine occupied space and available space depending on layout type
//  const bool is_horizontal = (layout.Type == ImGuiLayoutType_Horizontal);
//  const bool is_auto_sized =
//      ((is_horizontal ? layout.Size.x : layout.Size.y) <= 0.0f) &&
//      (layout.Parent == nullptr);
//  const float occupied_space =
//      is_horizontal ? layout.MinimumSize.x : layout.MinimumSize.y;
//  const float available_space =
//      is_auto_sized
//          ? occupied_space
//          : (is_horizontal ? layout.CurrentSize.x : layout.CurrentSize.y);
//  const float free_space = ImMax(available_space - occupied_space, 0.0f);
//
//  float span_start     = 0.0f;
//  float current_weight = 0.0f;
//  for (int i = 0; i < layout.Items.Size; i++) {
//    ImGuiLayoutItem& item = layout.Items[i];
//    if (item.Type != ImGuiLayoutItemType_Spring) continue;
//
//    float last_spring_size = item.SpringSize;
//
//    if (free_space > 0.0f && total_spring_weight > 0.0f) {
//      float next_weight = current_weight + item.SpringWeight;
//      float span_end =
//          ImFloor((i == last_spring_item_index)
//                      ? free_space
//                      : (free_space * next_weight / total_spring_weight));
//      float spring_size = span_end - span_start;
//      item.SpringSize   = spring_size;
//      span_start        = span_end;
//      current_weight    = next_weight;
//    } else {
//      item.SpringSize = 0.0f;
//    }
//
//    // If spring changed its size, fix positioning of following items to avoid
//    // one frame visual bugs.
//    if (last_spring_size != item.SpringSize) {
//      float difference = item.SpringSize - last_spring_size;
//
//      ImVec2 offset =
//          is_horizontal ? ImVec2(difference, 0.0f) : ImVec2(0.0f, difference);
//
//      item.MeasuredBounds.Max += offset;
//
//      for (int j = i + 1; j < layout.Items.Size; j++) {
//        ImGuiLayoutItem& translated_item = layout.Items[j];
//
//        TranslateLayoutItem(translated_item, offset);
//
//        translated_item.MeasuredBounds.Min += offset;
//        translated_item.MeasuredBounds.Max += offset;
//      }
//    }
//  }
//}
//
//ImVec2 BalanceLayoutItemAlignment(ImGuiLayout& layout, ImGuiLayoutItem& item) {
//  // Fixup item alignment if necessary.
//  ImVec2 position_correction = ImVec2(0.0f, 0.0f);
//  if (item.CurrentAlign > 0.0f) {
//    float item_align_offset = CalculateLayoutItemAlignmentOffset(layout, item);
//    if (item.CurrentAlignOffset != item_align_offset) {
//      float offset = item_align_offset - item.CurrentAlignOffset;
//
//      if (layout.Type == ImGuiLayoutType_Horizontal)
//        position_correction.y = offset;
//      else
//        position_correction.x = offset;
//
//      TranslateLayoutItem(item, position_correction);
//
//      item.CurrentAlignOffset = item_align_offset;
//    }
//  }
//
//  return position_correction;
//}
//
//void BalanceLayoutItemsAlignment(ImGuiLayout& layout) {
//  for (int i = 0; i < layout.Items.Size; ++i) {
//    ImGuiLayoutItem& item = layout.Items[i];
//    BalanceLayoutItemAlignment(layout, item);
//  }
//}
//
//bool HasAnyNonZeroSpring(ImGuiLayout& layout) {
//  for (int i = 0; i < layout.Items.Size; ++i) {
//    ImGuiLayoutItem& item = layout.Items[i];
//    if (item.Type != ImGuiLayoutItemType_Spring) continue;
//    if (item.SpringWeight > 0) return true;
//  }
//  return false;
//}
//
//void BalanceChildLayouts(ImGuiLayout& layout) {
//  for (ImGuiLayout* child = layout.FirstChild; child != nullptr;
//       child              = child->NextSibling) {
//    // ImVec2 child_layout_size = child->CurrentSize;
//
//    // Propagate layout size down to child layouts.
//    //
//    // TODO: Distribution assume inner layout is only
//    //       element inside parent item and assigns
//    //       all available space to it.
//    //
//    //       Investigate how to split space between
//    //       adjacent layouts.
//    //
//    //       Investigate how to measure non-layout items
//    //       to treat them as fixed size blocks.
//    //
//    if (child->Type == ImGuiLayoutType_Horizontal && child->Size.x <= 0.0f)
//      child->CurrentSize.x = layout.CurrentSize.x;
//    else if (child->Type == ImGuiLayoutType_Vertical && child->Size.y <= 0.0f)
//      child->CurrentSize.y = layout.CurrentSize.y;
//
//    BalanceChildLayouts(*child);
//
//    // child->CurrentSize = child_layout_size;
//
//    if (HasAnyNonZeroSpring(*child)) {
//      // Expand item measured bounds to make alignment correct.
//      ImGuiLayoutItem& item = layout.Items[child->ParentItemIndex];
//
//      if (child->Type == ImGuiLayoutType_Horizontal && child->Size.x <= 0.0f)
//        item.MeasuredBounds.Max.x =
//            ImMax(item.MeasuredBounds.Max.x,
//                  item.MeasuredBounds.Min.x + layout.CurrentSize.x);
//      else if (child->Type == ImGuiLayoutType_Vertical && child->Size.y <= 0.0f)
//        item.MeasuredBounds.Max.y =
//            ImMax(item.MeasuredBounds.Max.y,
//                  item.MeasuredBounds.Min.y + layout.CurrentSize.y);
//    }
//  }
//
//  BalanceLayoutSprings(layout);
//  BalanceLayoutItemsAlignment(layout);
//}
//
//ImGuiLayoutItem* GenerateLayoutItem(ImGuiLayout&        layout,
//                                    ImGuiLayoutItemType type) {
//  IM_ASSERT(layout.CurrentItemIndex <= layout.Items.Size);
//
//  if (layout.CurrentItemIndex < layout.Items.Size) {
//    ImGuiLayoutItem& item = layout.Items[layout.CurrentItemIndex];
//    if (item.Type != type) item = ImGuiLayoutItem(type);
//  } else {
//    layout.Items.push_back(ImGuiLayoutItem(type));
//  }
//
//  current_layout_item() = &layout.Items[layout.CurrentItemIndex];
//
//  return &layout.Items[layout.CurrentItemIndex];
//}
//
//// Calculate how many pixels from top/left layout edge item need to be moved to
//// match layout alignment.
//float CalculateLayoutItemAlignmentOffset(ImGuiLayout&     layout,
//                                         ImGuiLayoutItem& item) {
//  if (item.CurrentAlign <= 0.0f) return 0.0f;
//
//  ImVec2 item_size = item.MeasuredBounds.GetSize();
//
//  float layout_extent = (layout.Type == ImGuiLayoutType_Horizontal)
//                            ? layout.CurrentSize.y
//                            : layout.CurrentSize.x;
//  float item_extent =
//      (layout.Type == ImGuiLayoutType_Horizontal) ? item_size.y : item_size.x;
//
//  if (item_extent <= 0 [> || layout_extent <= item_extent<]) return 0.0f;
//
//  float align_offset =
//      ImFloor(item.CurrentAlign * (layout_extent - item_extent));
//
//  return align_offset;
//}
//
//void TranslateLayoutItem(ImGuiLayoutItem& item, const ImVec2& offset) {
//  if ((offset.x == 0.0f && offset.y == 0.0f) ||
//      (item.VertexIndexBegin == item.VertexIndexEnd))
//    return;
//
//  // IMGUI_DEBUG_LOG("TranslateLayoutItem by %f,%f\n", offset.x, offset.y);
//  ImDrawList* draw_list = GetWindowDrawList();
//
//  ImDrawVert* begin = draw_list->VtxBuffer.Data + item.VertexIndexBegin;
//  ImDrawVert* end   = draw_list->VtxBuffer.Data + item.VertexIndexEnd;
//
//  for (ImDrawVert* vtx = begin; vtx < end; ++vtx) {
//    vtx->pos.x += offset.x;
//    vtx->pos.y += offset.y;
//  }
//}
//
//void SignedIndent(float indent) {
//  if (indent > 0.0f)
//    Indent(indent);
//  else if (indent < 0.0f)
//    Unindent(-indent);
//}
//
//void BeginLayoutItem(ImGuiLayout& layout) {
//  ImGuiContext&    g      = *GImGui;
//  ImGuiWindow*     window = g.CurrentWindow;
//  ImGuiLayoutItem& item = *GenerateLayoutItem(layout, ImGuiLayoutItemType_Item);
//
//  item.CurrentAlign = layout.Align;
//  if (item.CurrentAlign < 0.0f)
//    item.CurrentAlign = ImClamp(layout_align(), 0.0f, 1.0f);
//
//  // Align item according to data from previous frame.
//  // If layout changes in current frame alignment will
//  // be corrected in EndLayout() to it visualy coherent.
//  item.CurrentAlignOffset = CalculateLayoutItemAlignmentOffset(layout, item);
//  if (item.CurrentAlign > 0.0f) {
//    if (layout.Type == ImGuiLayoutType_Horizontal) {
//      window->DC.CursorPos.y += item.CurrentAlignOffset;
//    } else {
//      float new_position = window->DC.CursorPos.x + item.CurrentAlignOffset;
//
//      // Make placement behave like in horizontal case when next
//      // widget is placed at very same Y position. This indent
//      // make sure for vertical layout placed widgets has same X position.
//      SignedIndent(item.CurrentAlignOffset);
//
//      window->DC.CursorPos.x = new_position;
//    }
//  }
//
//  item.MeasuredBounds.Min = item.MeasuredBounds.Max = window->DC.CursorPos;
//  item.VertexIndexBegin                             = item.VertexIndexEnd =
//      window->DrawList->_VtxCurrentIdx;
//}
//
//void EndLayoutItem(ImGuiLayout& layout) {
//  ImGuiContext& g      = *GImGui;
//  ImGuiWindow*  window = g.CurrentWindow;
//  IM_ASSERT(layout.CurrentItemIndex < layout.Items.Size);
//
//  ImGuiLayoutItem& item = layout.Items[layout.CurrentItemIndex];
//
//  ImDrawList* draw_list = window->DrawList;
//  item.VertexIndexEnd   = draw_list->_VtxCurrentIdx;
//
//  if (item.CurrentAlign > 0.0f && layout.Type == ImGuiLayoutType_Vertical)
//    SignedIndent(-item.CurrentAlignOffset);
//
//  // Fixup item alignment in case item size changed in current frame.
//  ImVec2 position_correction = BalanceLayoutItemAlignment(layout, item);
//
//  item.MeasuredBounds.Min += position_correction;
//  item.MeasuredBounds.Max += position_correction;
//
//  if (layout.Type == ImGuiLayoutType_Horizontal)
//    window->DC.CursorPos.y = layout.StartPos.y;
//  else
//    window->DC.CursorPos.x = layout.StartPos.x;
//
//  layout.CurrentItemIndex++;
//}
//
//void AddLayoutSpring(ImGuiLayout& layout, float weight, float spacing) {
//  ImGuiContext&    g             = *GImGui;
//  ImGuiWindow*     window        = g.CurrentWindow;
//  ImGuiLayoutItem* previous_item = &layout.Items[layout.CurrentItemIndex];
//
//  // Undo item padding, spring should consume all space between items.
//  if (layout.Type == ImGuiLayoutType_Horizontal)
//    window->DC.CursorPos.x = previous_item->MeasuredBounds.Max.x;
//  else
//    window->DC.CursorPos.y = previous_item->MeasuredBounds.Max.y;
//
//  previous_item = nullptr;  // may be invalid after call to GenerateLayoutItem()
//
//  EndLayoutItem(layout);
//
//  ImGuiLayoutItem* spring_item =
//      GenerateLayoutItem(layout, ImGuiLayoutItemType_Spring);
//
//  spring_item->MeasuredBounds.Min = spring_item->MeasuredBounds.Max =
//      window->DC.CursorPos;
//
//  if (weight < 0.0f) weight = 0.0f;
//
//  if (spring_item->SpringWeight != weight) spring_item->SpringWeight = weight;
//
//  if (spacing < 0.0f) {
//    ImVec2 style_spacing = g.Style.ItemSpacing;
//    if (layout.Type == ImGuiLayoutType_Horizontal)
//      spacing = style_spacing.x;
//    else
//      spacing = style_spacing.y;
//  }
//
//  if (spring_item->SpringSpacing != spacing)
//    spring_item->SpringSpacing = spacing;
//
//  if (spring_item->SpringSize > 0.0f || spacing > 0.0f) {
//    ImVec2 spring_size, spring_spacing;
//    if (layout.Type == ImGuiLayoutType_Horizontal) {
//      spring_spacing = ImVec2(0.0f, g.Style.ItemSpacing.y);
//      spring_size =
//          ImVec2(spacing + spring_item->SpringSize, layout.CurrentSize.y);
//    } else {
//      spring_spacing = ImVec2(g.Style.ItemSpacing.x, 0.0f);
//      spring_size =
//          ImVec2(layout.CurrentSize.x, spacing + spring_item->SpringSize);
//    }
//
//    PushStyleVar(ImGuiStyleVar_ItemSpacing, ImFloor(spring_spacing));
//    Dummy(ImFloor(spring_size));
//    PopStyleVar();
//  }
//
//  layout.CurrentItemIndex++;
//
//  BeginLayoutItem(layout);
//}
////------------------------------------------------------------------------------
//void BeginHorizontal(const char* str_id, const ImVec2& size [> = ImVec2(0, 0)<],
//                     float align [> = -1<]) {
//  ImGuiWindow* window = GetCurrentWindow();
//  BeginLayout(window->GetID(str_id), ImGuiLayoutType_Horizontal, size, align);
//}
//
////------------------------------------------------------------------------------
//void BeginHorizontal(const void* ptr_id, const ImVec2& size [> = ImVec2(0, 0)<],
//                     float align [> = -1<]) {
//  ImGuiWindow* window = GetCurrentWindow();
//  BeginLayout(window->GetID(ptr_id), ImGuiLayoutType_Horizontal, size, align);
//}
//
////------------------------------------------------------------------------------
//void BeginHorizontal(int id, const ImVec2& size [> = ImVec2(0, 0)<],
//                     float align [> = -1<]) {
//  ImGuiWindow* window = GetCurrentWindow();
//  BeginLayout(window->GetID((void*)(intptr_t)id), ImGuiLayoutType_Horizontal,
//              size, align);
//}
////------------------------------------------------------------------------------
//void EndHorizontal() { EndLayout(ImGuiLayoutType_Horizontal); }
////------------------------------------------------------------------------------
//void BeginVertical(const char* str_id, const ImVec2& size [> = ImVec2(0, 0)<],
//                   float align [> = -1<]) {
//  ImGuiWindow* window = GetCurrentWindow();
//  BeginLayout(window->GetID(str_id), ImGuiLayoutType_Vertical, size, align);
//}
////------------------------------------------------------------------------------
//void BeginVertical(const void* ptr_id, const ImVec2& size [> = ImVec2(0, 0)<],
//                   float align [> = -1<]) {
//  ImGuiWindow* window = GetCurrentWindow();
//  BeginLayout(window->GetID(ptr_id), ImGuiLayoutType_Vertical, size, align);
//}
////------------------------------------------------------------------------------
//void BeginVertical(int id, const ImVec2& size [> = ImVec2(0, 0)<],
//                   float align [> = -1<]) {
//  ImGuiWindow* window = GetCurrentWindow();
//  BeginLayout(window->GetID((void*)(intptr_t)id), ImGuiLayoutType_Vertical,
//              size, align);
//}
////------------------------------------------------------------------------------
//void EndVertical() { EndLayout(ImGuiLayoutType_Vertical); }
////------------------------------------------------------------------------------
//// Inserts spring separator in layout
////      weight <= 0     : spring will always have zero size
////      weight > 0      : power of current spring
////      spacing < 0     : use default spacing if pos_x == 0, no spacing if pos_x
////      != 0 spacing >= 0    : enforce spacing amount
//void Spring(float weight [> = 1.0f*/, float spacing /* = -1.0f<]) {
//  IM_ASSERT(current_layout());
//
//  AddLayoutSpring(*current_layout(), weight, spacing);
//}
//==============================================================================
}  // namespace ImGui
//==============================================================================
