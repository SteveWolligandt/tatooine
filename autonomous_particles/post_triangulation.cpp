#include <tatooine/edgeset.h>
#include <tatooine/vec.h>

#include <vector>
//==============================================================================
using vertex = tatooine::edgeset2::vertex_handle;
//==============================================================================
struct hierarchy_pair {
  std::size_t id;
  std::size_t parent;
};
struct center_of {
  std::size_t id;
  std::size_t center;
};
//==============================================================================
struct hierarchy {
  std::size_t            id     = std::numeric_limits<std::size_t>::max();
  vertex                 center = vertex{};
  std::vector<hierarchy> leafs  = {};

  hierarchy() = default;
  explicit hierarchy(std::size_t const id_) : id{id_} {}
  hierarchy(std::size_t const id_, vertex const center_)
      : id{id_}, center{center_} {}
};
//==============================================================================
auto fill(hierarchy& h, std::vector<hierarchy_pair> const& hps,
          std::vector<vertex> const& centers) -> void {
  for (auto const& hp : hps) {
    if (h.id == hp.parent && hp.parent != hp.id) {
      fill(h.leafs.emplace_back(hp.id, centers[hp.id]), hps, centers);
    }
  }
}
//==============================================================================
auto trace_center_vertex(std::size_t id, tatooine::edgeset2& edgeset,
                         std::vector<std::size_t> const& centers) {
  while (centers[id] != id) {
    id = centers[id];
  }
  return id;
}
//==============================================================================
auto process(tatooine::edgeset2& edgeset, hierarchy const& h) -> void {
  
}
//==============================================================================
auto total_num_particles(std::vector<hierarchy_pair> const& hps) {
  std::size_t num = 0; 
  for (auto const& hp : hps) {
    num = std::max(num, hp.id);
  }
  return num + 1;
}
//==============================================================================
auto main() -> int {
  using namespace std::ranges;
  // [    0    ]
  //  [1][2][3]
  // [[x][x][x]]

  auto  hierarchy_pair_list  = std::vector<hierarchy_pair>{};
  auto  center_list          = std::vector<center_of>{};

  center_list.emplace_back(0, 2);
  center_list.emplace_back(1, 1);
  center_list.emplace_back(2, 2);
  center_list.emplace_back(3, 3);

  hierarchy_pair_list.emplace_back(0, 0);
  hierarchy_pair_list.emplace_back(1, 0);
  hierarchy_pair_list.emplace_back(2, 0);
  hierarchy_pair_list.emplace_back(3, 0);
  //============================================================================
  auto  edges                = tatooine::edgeset2{};
  auto  particle_vertex_link = std::unordered_map<std::size_t, vertex>{};
  // ids and centers are taken from autonomous_particles
  auto v0                 = edges.insert_vertex(-100, 0);
  particle_vertex_link[1] = v0;
  auto v1                 = edges.insert_vertex(0, 0);
  particle_vertex_link[2] = v1;
  auto v2                 = edges.insert_vertex(100, 0);
  particle_vertex_link[3] = v2;

  auto const num_parts       = total_num_particles(hierarchy_pair_list);
  auto       center_vertices = std::vector<vertex>(num_parts);
  {
    auto centers = std::vector<std::size_t>(num_parts);
    for (auto const& c : center_list) {
      centers[c.id] = c.center;
    }
    for (std::size_t i = 0; i < num_parts; ++i) {
      center_vertices[i] =
          particle_vertex_link[trace_center_vertex(centers[i], edges, centers)];
    }
  }

  auto  h                    = hierarchy{};
  for (auto const& hp : hierarchy_pair_list) {
    // search for top nodes
    if (hp.id == hp.parent) {
      ::fill(h.leafs.emplace_back(hp.parent,
                                  particle_vertex_link.at(centers[hp.parent])),
             hierarchy_pair_list, centers);
    }
  }
  process(edges, h);
  edges.write("post.vtp");
}
