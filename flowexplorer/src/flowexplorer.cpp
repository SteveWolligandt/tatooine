#include <tatooine/flowexplorer/window.h>
auto main(int argc, char const** argv) -> int {
  if (argc == 1) {
    tatooine::flowexplorer::window w{};
  } else if (argc == 2) {
    tatooine::flowexplorer::window w{argv[1]};
  }
}
