#ifndef TATOOINE_GL_IMGUI_TYPE_TRAITS_H
#define TATOOINE_GL_IMGUI_TYPE_TRAITS_H
//==============================================================================
#include <tatooine/gl/imgui_includes.h>
//==============================================================================
namespace ImGui {
//==============================================================================
// Infer ImGuiDataType enum based on actual type
template <typename T>
struct ImGuiDataTypeTraits {
  static const ImGuiDataType value;  // link error
  static const char*         format;
};

template <>
struct ImGuiDataTypeTraits<std::int32_t> {
  static constexpr ImGuiDataType value  = ImGuiDataType_S32;
  static constexpr const char*   format = "%d";
};

template <>
struct ImGuiDataTypeTraits<std::uint32_t> {
  static constexpr ImGuiDataType value  = ImGuiDataType_U32;
  static constexpr const char*   format = "%u";
};

template <>
struct ImGuiDataTypeTraits<std::int64_t> {
  static constexpr ImGuiDataType value  = ImGuiDataType_S64;
  static constexpr const char*   format = "%lld";
};

template <>
struct ImGuiDataTypeTraits<std::uint64_t> {
  static constexpr ImGuiDataType value  = ImGuiDataType_U64;
  static constexpr const char*   format = "%llu";
};

template <>
struct ImGuiDataTypeTraits<float> {
  static constexpr ImGuiDataType value  = ImGuiDataType_Float;
  static constexpr const char*   format = "%.3f";
};

template <>
struct ImGuiDataTypeTraits<double> {
  static constexpr ImGuiDataType value  = ImGuiDataType_Double;
  static constexpr const char*   format = "%.6f";
};
//==============================================================================
}  // namespace ImGui
//==============================================================================
#endif
