#include <tatooine/gl/context.h>
#include <tatooine/gl/indexbuffer.h>
using namespace tatooine;
auto main() -> int {
  auto ctx     = gl::context{};
  auto indices = gl::indexbuffer{3};
  {
    auto   map = indices.wmap();
    size_t i   = 0;
    for (auto& elem : map) {
      elem = i++;
    }
  }
}
