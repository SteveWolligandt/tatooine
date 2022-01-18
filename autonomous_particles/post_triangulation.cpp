#include <tatooine/edgeset.h>
#include <tatooine/vec.h>

#include <vector>
//==============================================================================
namespace tatooine {
auto reorder(std::ranges::range auto& data, std::ranges::range auto& order)
    -> void requires integral<std::ranges::range_value_t<decltype(order)>> {
  assert(std::ranges::size(data) == std::ranges::size(order));

  for (std::size_t vv = 0; vv < size(data) - 1; ++vv) {
    if (order[vv] == vv) {
      continue;
    }
    auto oo = std::size_t{};
    for (oo = vv + 1; oo < order.size(); ++oo) {
      if (order[oo] == vv) {
        break;
      }
    }
    std::swap(data[vv], data[order[vv]]);
    std::swap(order[vv], order[oo]);
  }
}
//==============================================================================
using vertex = edgeset2::vertex_handle;
//==============================================================================
struct hierarchy_pair {
  std::size_t id;
  std::size_t parent;
};
//==============================================================================
struct center_of {
  std::size_t id;
  std::size_t center;
};
//------------------------------------------------------------------------------
auto trace_center_vertex(std::size_t                     id,
                         std::vector<std::size_t> const& centers) {
  while (centers[id] != id) {
    id = centers[id];
  }
  return id;
}
//==============================================================================
struct hierarchy {
  std::size_t            id     = std::numeric_limits<std::size_t>::max();
  vertex                 center = vertex{};
  std::vector<hierarchy> leafs  = {};

 public:
  //============================================================================
  hierarchy(std::size_t const id_, vertex const center_)
      : id{id_}, center{center_} {}
  //----------------------------------------------------------------------------
  /// as child node
  hierarchy(std::size_t const id_, vertex const center_,
            std::vector<hierarchy_pair> const& hps,
            std::vector<vertex> const& centers, edgeset2 const& edges)
      : hierarchy{id_, center_} {
    build(hps, centers, edges);
    if (!leafs.empty()) {
      sort_leafs(edges);
    }
  }
  //----------------------------------------------------------------------------
  /// as top node
  hierarchy(std::vector<hierarchy_pair> const& hps,
            std::vector<vertex> const& centers, edgeset2 const& edges) {
    for (auto const& hp : hps) {
      // search for top nodes
      if (hp.id == hp.parent) {
        leafs.emplace_back(hp.parent, centers[hp.parent], hps, centers, edges);
      }
    }
  }
  //----------------------------------------------------------------------------
 private:
  auto build(std::vector<hierarchy_pair> const& hps,
             std::vector<vertex> const& centers, edgeset2 const& edges)
      -> void {
    for (auto const& hp : hps) {
      if (id == hp.parent && hp.parent != hp.id) {
        leafs.emplace_back(hp.id, centers[hp.id], hps, centers, edges);
      }
    }
  }
  //----------------------------------------------------------------------------
  auto sort_leafs(edgeset2 const& edges) -> void {
    auto split_dir = calc_split_dir(edges);
    auto dists     = std::vector<std::pair<std::size_t, real_t>>{};
    dists.reserve(leafs.size());
    dists.emplace_back(0, 0.0);
    for (std::size_t i = 1; i < leafs.size(); ++i) {
      auto offset = edges[leafs[i].center] - edges[leafs[0].center];
      dists.emplace_back(i, std::copysign(squared_euclidean_length(offset),
                                          dot(offset, split_dir)));
    }
    std::ranges::sort(dists, [](auto const& lhs, auto const& rhs) {
      auto const& [i, idist] = lhs;
      auto const& [j, jdist] = rhs;
      return idist < jdist;
    });
    auto reordered_indices = std::vector<std::size_t>(leafs.size());
    using namespace std::ranges;
    copy(dists | views::transform([](auto const& x) { return x.first; }),
         begin(reordered_indices));
    reorder(leafs, reordered_indices);
  }
  //----------------------------------------------------------------------------
  auto calc_split_dir(edgeset2 const& edges) -> vec2 {
    if (leafs.empty()) {
      return vec2::zeros();
    }
    auto dir = normalize(edges[leafs[0].center] - edges[leafs[1].center]);
    if (dir.x() < 0) {
      dir = -dir;
    }
    return dir;
  }
};
//==============================================================================
auto get_front(hierarchy const& h, edgeset2 const& edges,
               vec2 const& other_center, vec2 const& offset_dir) -> std::vector<vertex> {
  using namespace std::ranges;
  if (h.leafs.empty()) {
    return {h.center};
  }
  auto const split_dir  = edges[h.leafs[1].center] - edges[h.leafs[0].center];
  auto       front      = std::vector<vertex>{};
  if (std::abs(cos_angle(offset_dir, split_dir)) < 0.3) {
    // copy all if center-center direction is perpendicular split direction
    for (auto const& l : h.leafs) {
      copy(get_front(l, edges, other_center, offset_dir), std::back_inserter(front));
    }
  } else {
    // copy only nearest particle to front
    if (squared_euclidean_distance(other_center,
                                   edges[h.leafs.front().center]) <
        squared_euclidean_distance(other_center,
                                   edges[h.leafs.back().center])) {
      copy(get_front(h.leafs.front(), edges, other_center, offset_dir),
           std::back_inserter(front));
    } else {
      copy(get_front(h.leafs.back(), edges, other_center, offset_dir),
           std::back_inserter(front));
    }
  }
  return front;
}
//------------------------------------------------------------------------------
auto connect_fronts(std::vector<vertex> front0, std::vector<vertex> front1,
                    edgeset2& edges) -> void {
  if (squared_euclidean_distance(edges[front0.front()], edges[front1.front()]) >
      squared_euclidean_distance(edges[front0.front()], edges[front1.back()])) {
    std::ranges::reverse(front1);
  }

  auto it0  = begin(front0);
  auto it1  = begin(front1);
  auto end0 = end(front0);
  auto end1 = end(front1);

  if (size(front0) > 1 || size(front1) > 1) {
    edges.insert_edge(*it0, *it1);
  }
  while (next(it0) != end0 || next(it1) != end1) {
    if (next(it0) == end0) {
      ++it1;
      edges.insert_edge(*it0, *it1);
    } else if (next(it1) == end1) {
      ++it0;
      edges.insert_edge(*it0, *it1);
    } else {
      auto itn0 = next(it0);
      auto itn1 = next(it1);
      if (squared_euclidean_distance(edges[*it0], edges[*itn1]) <
          squared_euclidean_distance(edges[*it1], edges[*itn0])) {
        edges.insert_edge(*it0, *itn1);
        ++it1;
      } else {
        edges.insert_edge(*it1, *itn0);
        ++it0;
      }
    }
  }
  edges.insert_edge(*it0, *it1);
}
//------------------------------------------------------------------------------
auto process_child(edgeset2& edges, hierarchy const& h) -> void {
  using namespace std::ranges;
  if (!h.leafs.empty()) {
    for (auto const& l : h.leafs) {
      process_child(edges, l);
    }
    connect_fronts(
        get_front(h.leafs[0], edges, edges[h.leafs[1].center],
                  edges[h.leafs[0].center] - edges[h.leafs[1].center]),
        get_front(h.leafs[1], edges, edges[h.leafs[0].center],
                  edges[h.leafs[1].center] - edges[h.leafs[0].center]),
        edges);
    connect_fronts(
        get_front(h.leafs[2], edges, edges[h.leafs[1].center],
                  edges[h.leafs[2].center] - edges[h.leafs[1].center]),
        get_front(h.leafs[1], edges, edges[h.leafs[2].center],
                  edges[h.leafs[1].center] - edges[h.leafs[2].center]),
        edges);
  }
}
//------------------------------------------------------------------------------
auto process_top(edgeset2& edgeset, hierarchy const& h) -> void {
  for (auto const& top_leaf : h.leafs) {
    process_child(edgeset, top_leaf);
  }
}
//==============================================================================
auto print(hierarchy const& h) -> void {
  if (h.leafs.empty()) {
    std::cout << h.center;
  } else {
    if (h.center != h.center.invalid_idx) {
      std::cout << h.center << ":";
    }
    std::cout << "{";
    print(h.leafs.front());
    for (std::size_t i = 1; i < size(h.leafs); ++i) {
      std::cout << ' ';
      print(h.leafs[i]);
    }
    std::cout << "}";
  }
}
//==============================================================================
auto total_num_particles(std::vector<hierarchy_pair> const& hps) {
  std::size_t num = 0;
  for (auto const& hp : hps) {
    num = std::max(num, hp.id);
  }
  return num + 1;
}
//------------------------------------------------------------------------------
auto total_num_particles(std::vector<center_of> const& centers) {
  std::size_t num = 0;
  for (auto const& c : centers) {
    num = std::max(num, c.id);
  }
  return num + 1;
}
//------------------------------------------------------------------------------
auto transform_center_list(std::vector<center_of>& center_list,
                           std::unordered_map<std::size_t, vertex> const& map) {
  auto const num_parts       = total_num_particles(center_list);
  auto       center_vertices = std::vector<vertex>(num_parts);
  auto       centers         = std::vector<std::size_t>(num_parts);
  for (auto const& c : center_list) {
    centers[c.id] = c.center;
  }
  center_list.clear();
  center_list.shrink_to_fit();

  for (std::size_t i = 0; i < num_parts; ++i) {
    center_vertices[i] = map.at(trace_center_vertex(i, centers));
  }
  return center_vertices;
}
//==============================================================================
auto build_vertices_of_edgeset() {
  auto edges = edgeset2{};
  //  centers are taken from autonomous_particles
  edges.insert_vertex(0, 0);
  edges.insert_vertex(-1, 0);
  edges.insert_vertex(1, 0);
  edges.insert_vertex(0, 1);
  edges.insert_vertex(0, -1);
  edges.insert_vertex(0, 2);
  edges.insert_vertex(-2, 0);
  edges.insert_vertex(2, 0);
  edges.insert_vertex(3, 0);
  edges.insert_vertex(4, 0);
  edges.insert_vertex(-2, -2);
  edges.insert_vertex(0, -2);
  edges.insert_vertex(3, -2);
  auto map = std::unordered_map<std::size_t, vertex>{
      std::pair{std::size_t(1), vertex{1}},
      std::pair{std::size_t(3), vertex{2}},
      std::pair{std::size_t(4), vertex{4}},
      std::pair{std::size_t(5), vertex{0}},
      std::pair{std::size_t(6), vertex{3}},
      std::pair{std::size_t(12), vertex{5}},
      std::pair{std::size_t(7), vertex{6}},
      std::pair{std::size_t(9), vertex{7}},
      std::pair{std::size_t(10), vertex{8}},
      std::pair{std::size_t(11), vertex{9}},
      std::pair{std::size_t(14), vertex{10}},
      std::pair{std::size_t(15), vertex{11}},
      std::pair{std::size_t(16), vertex{12}}};
  return std::pair{std::move(edges), std::move(map)};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
auto main() -> int {
  using namespace tatooine;
  auto center_list = std::vector<center_of>{};
  center_list.emplace_back(0, 2);
  center_list.emplace_back(1, 1);
  center_list.emplace_back(2, 5);
  center_list.emplace_back(3, 3);
  center_list.emplace_back(4, 4);
  center_list.emplace_back(5, 5);
  center_list.emplace_back(6, 6);
  center_list.emplace_back(7, 7);
  center_list.emplace_back(8, 10);
  center_list.emplace_back(9, 9);
  center_list.emplace_back(10, 10);
  center_list.emplace_back(11, 11);
  center_list.emplace_back(12, 12);
  center_list.emplace_back(13, 15);
  center_list.emplace_back(14, 14);
  center_list.emplace_back(15, 15);
  center_list.emplace_back(16, 16);
  center_list.emplace_back(17, 18);
  center_list.emplace_back(18, 0);
  //============================================================================
  auto hierarchy_pair_list = std::vector<hierarchy_pair>{};
  hierarchy_pair_list.emplace_back(0, 18);
  hierarchy_pair_list.emplace_back(1, 0);
  hierarchy_pair_list.emplace_back(2, 0);
  hierarchy_pair_list.emplace_back(3, 0);
  hierarchy_pair_list.emplace_back(4, 2);
  hierarchy_pair_list.emplace_back(5, 2);
  hierarchy_pair_list.emplace_back(6, 2);
  hierarchy_pair_list.emplace_back(7, 18);
  hierarchy_pair_list.emplace_back(8, 18);
  hierarchy_pair_list.emplace_back(9, 8);
  hierarchy_pair_list.emplace_back(10, 8);
  hierarchy_pair_list.emplace_back(11, 8);
  hierarchy_pair_list.emplace_back(12, 17);
  hierarchy_pair_list.emplace_back(13, 17);
  hierarchy_pair_list.emplace_back(14, 13);
  hierarchy_pair_list.emplace_back(15, 13);
  hierarchy_pair_list.emplace_back(16, 13);
  hierarchy_pair_list.emplace_back(17, 17);
  hierarchy_pair_list.emplace_back(18, 17);
  //============================================================================
  auto [edges, map] = build_vertices_of_edgeset();
  //============================================================================
  auto center_vertices = transform_center_list(center_list, map);
  auto h               = hierarchy{hierarchy_pair_list, center_vertices, edges};

  print(h);
  process_top(edges, h);
  edges.write("post.vtp");
}
