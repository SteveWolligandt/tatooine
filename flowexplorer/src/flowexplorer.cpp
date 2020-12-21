#include <signal.h>
#include <tatooine/flowexplorer/window.h>
auto sig_handler(int signo) -> void {
  if (signo == SIGINT) {
    std::cerr << "received SIGINT\n";
  }
}
auto main(int argc, char const** argv) -> int {
  //if (signal(SIGINT, sig_handler) == SIG_ERR) {
  //  std::cerr << "can't catch SIGINT\n";
  //}
  if (argc == 1) {
    tatooine::flowexplorer::window w{};
  } else if (argc == 2) {
    tatooine::flowexplorer::window w{argv[1]};
  }
}
