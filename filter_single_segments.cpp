#include <tatooine/line.h>
#include <tatooine/progress_bars.h>
#include <ranges>
using namespace tatooine;
auto main(int argc, char **argv) -> int {
  auto lines = indeterminate_progress_bar(
    [&](auto indicator) {
      indicator = "reading";
      return line3::read_vtp(argv[1]);
    });
  std::cout << "Read " << size(lines) << " lines.\n";
  auto filtered_lines = std::vector<line3>{};
  auto no_segment = [](auto const& line){ return line.num_vertices() > 1; };
  std::ranges::copy(lines | std::views::filter(no_segment))

  std::cout << "merged into " << size(merged_lines) << " lines.\n";
  indeterminate_progress_bar(
    [&](auto indicator) {
      indicator = "writing";
      write(merged_lines, argv[2]);
    });
}
