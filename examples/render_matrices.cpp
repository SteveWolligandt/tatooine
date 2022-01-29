#include <tatooine/rendering/matrices.h>
using namespace tatooine;
auto main() -> int {
  auto V = rendering::look_at_matrix(normalize(vec3{1, 0, 1}), vec3{0, 0, 0});
  std::cout << V << '\n';
  std::cout << V * vec4{0, 0, 0, 1} << '\n';
  std::cout << V * vec4{1, 0, 1, 1} << '\n';


}
