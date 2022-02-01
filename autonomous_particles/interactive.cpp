#include <tatooine/rendering/interactive.h>
using namespace tatooine;
auto main() -> int {
  auto ell = geometry::ellipse{3.0f};
  rendering::interactive(ell);
}
