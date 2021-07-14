#ifndef TATOOINE_BEZIER_WIDGET
#define TATOOINE_BEZIER_WIDGET
#include <vector>
/// \example
/// {  static auto v = std::vector<float>{ 0.390f, 0.575f, 0.565f, 1.000f };
///    ImGui::Bezier( "easeOutSine", v );       // draw
///    float y = ImGui::BezierValue( 0.5f, v ); // x delta in [0..1] range
/// }
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>

//==============================================================================
namespace ImGui {
//==============================================================================
auto BezierValue(float dt01, std::vector<float> const& handles) -> float;
//------------------------------------------------------------------------------
auto Bezier(const char* label, std::vector<float>& handles) -> int;
//------------------------------------------------------------------------------
template <int steps>
auto BezierTable(ImVec2 P[4], ImVec2 results[steps + 1]) -> void {
  static float C[(steps + 1) * 4], *K = 0;
  if (!K) {
    K = C;
    for (unsigned step = 0; step <= steps; ++step) {
      float t         = (float)step / (float)steps;
      C[step * 4 + 0] = (1 - t) * (1 - t) * (1 - t);  // * P0
      C[step * 4 + 1] = 3 * (1 - t) * (1 - t) * t;    // * P1
      C[step * 4 + 2] = 3 * (1 - t) * t * t;          // * P2
      C[step * 4 + 3] = t * t * t;                    // * P3
    }
  }
  for (unsigned step = 0; step <= steps; ++step) {
    ImVec2 point  = {K[step * 4 + 0] * P[0].x + K[step * 4 + 1] * P[1].x +
                        K[step * 4 + 2] * P[3].x + K[step * 4 + 3] * P[2].x,
                    K[step * 4 + 0] * P[0].y + K[step * 4 + 1] * P[1].y +
                        K[step * 4 + 2] * P[3].y + K[step * 4 + 3] * P[2].y};
    results[step] = point;
  }
}
//==============================================================================
}  // namespace ImGui
//==============================================================================
#endif
