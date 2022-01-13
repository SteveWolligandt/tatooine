#include <tatooine/edgeset.h>
#include <tatooine/vec.h>

#include <vector>
//==============================================================================
using namespace tatooine;
//==============================================================================
struct hierarchy_parent_child {
  std::size_t child;
  std::size_t parent;
};
//==============================================================================
struct hierarchy {
  std::size_t            id        = std::numeric_limits<std::size_t>::max();
  std::size_t            center    = std::numeric_limits<std::size_t>::max();
  vec2                   split_dir = vec2{0, 0};
  std::vector<hierarchy> leafs     = {};
};
//==============================================================================
auto main() -> int {
  // [    0    ]
  //  [1][2][3]
  // [[x][x][x]]
  auto hierarchy_list = std::vector<hierarchy_parent_child>{};
  auto centers        = std::vector<std::size_t>{};
  auto particles      = std::vector<vec2>{};
  auto split_dirs     = std::vector<vec2>{};
  auto edges          = edgeset2{};

  centers.push_back(2);
  centers.push_back(1);
  centers.push_back(2);
  centers.push_back(3);

  split_dirs.emplace_back(1, 0);
  split_dirs.emplace_back(0, 0);
  split_dirs.emplace_back(0, 0);
  split_dirs.emplace_back(0, 0);

  hierarchy_list.emplace_back(0, 0);
  hierarchy_list.emplace_back(1, 0);
  hierarchy_list.emplace_back(2, 0);
  hierarchy_list.emplace_back(3, 0);

  particles.emplace_back(-100, 0);
  particles.emplace_back(0, 0);
  particles.emplace_back(100, 0);

  auto base_particles = std::vector<std::size_t>{};
  using namespace std::ranges;
  for (std::size_t i = 0; i < size(hierarchy_list); ++i) {
    auto const& hl = hierarchy_list[i];
    if (hl.child == hl.parent) {
      base_particles.push_back(i);
    }
  }
  auto h = hierarchy{};
  for (auto const& b : base_particles) {
    h.leafs.push_back();
    h.leafs.back().id = b;
  }
}
