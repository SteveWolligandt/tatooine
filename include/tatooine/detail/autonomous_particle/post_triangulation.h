#ifndef TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_POST_TRIANGULATION_H
#define TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_POST_TRIANGULATION_H
//==============================================================================
#include <tatooine/edgeset.h>
#include <tatooine/utility/reorder.h>

#include <cstdint>
#include <unordered_map>
//==============================================================================
namespace tatooine::detail::autonomous_particle {
//==============================================================================
template <typename Real, std::size_t NumDimensions>
using vertex = typename tatooine::edgeset<Real, NumDimensions>::vertex_handle;
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
template <typename Real, std::size_t NumDimensions>
struct hierarchy {
  using vertex_type  = vertex<Real, NumDimensions>;
  using edgeset_type = edgeset<Real, NumDimensions>;
  using vec_type     = vec<Real, NumDimensions>;

  std::size_t            id     = std::numeric_limits<std::size_t>::max();
  vertex_type                 center = vertex_type{};
  std::vector<hierarchy> leafs  = {};

 public:
  //============================================================================
  hierarchy(std::size_t const id_, vertex_type const center_)
      : id{id_}, center{center_} {}
  //----------------------------------------------------------------------------
  /// as child node
  hierarchy(std::size_t const id_, vertex_type const center_,
            std::vector<hierarchy_pair> const& hps,
            std::vector<vertex_type> const& centers, edgeset_type const& edges)
      : hierarchy{id_, center_} {
    build(hps, centers, edges);
    if (!leafs.empty()) {
      sort_leafs(edges);
    }
  }
  //----------------------------------------------------------------------------
  /// as top node
  hierarchy(std::vector<hierarchy_pair> const& hps,
            std::vector<vertex_type> const& centers, edgeset_type const& edges) {
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
             std::vector<vertex_type> const& centers, edgeset_type const& edges) -> void {
    for (auto const& hp : hps) {
      if (id == hp.parent && hp.parent != hp.id) {
        leafs.emplace_back(hp.id, centers[hp.id], hps, centers, edges);
      }
    }
  }
  //----------------------------------------------------------------------------
  auto sort_leafs(edgeset_type const& edges) -> void {
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
  auto calc_split_dir(edgeset_type const& edges) -> vec_type {
    if (leafs.empty()) {
      return vec_type::zeros();
    }
    auto dir = normalize(edges[leafs[0].center] - edges[leafs[1].center]);
    if (dir.x() < 0) {
      dir = -dir;
    }
    return dir;
  }
};
//==============================================================================
template <typename Real, std::size_t NumDimensions>
auto get_front(hierarchy<Real, NumDimensions> const& h,
               edgeset<Real, NumDimensions> const&   edges,
               vec<Real, NumDimensions> const&       other_center,
               vec<Real, NumDimensions> const&       offset_dir)
    -> std::vector<vertex<Real, NumDimensions>> {
  using namespace std::ranges;
  if (h.leafs.empty()) {
    return {h.center};
  }
  auto const split_dir = edges[h.leafs[1].center] - edges[h.leafs[0].center];
  auto       front     = std::vector<vertex<Real, NumDimensions>>{};
  if (std::abs(cos_angle(offset_dir, split_dir)) < 0.3) {
    // copy all if center-center direction is perpendicular split direction
    for (auto const& l : h.leafs) {
      copy(get_front(l, edges, other_center, offset_dir),
           std::back_inserter(front));
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
template <typename Real, std::size_t NumDimensions>
auto connect_fronts(std::vector<vertex<Real, NumDimensions>> front0,
                    std::vector<vertex<Real, NumDimensions>> front1,
                    edgeset<Real, NumDimensions>&            edges) -> void {
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
template <typename Real, std::size_t NumDimensions>
auto process_child(edgeset<Real, NumDimensions>&         edges,
                   hierarchy<Real, NumDimensions> const& h) -> void {
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
template <typename Real, std::size_t NumDimensions>
auto process_top(edgeset<Real, NumDimensions>&         edgeset,
                 hierarchy<Real, NumDimensions> const& h) -> void {
  for (auto const& top_leaf : h.leafs) {
    process_child(edgeset, top_leaf);
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
template <typename Real, std::size_t NumDimensions>
auto transform_center_list(
    std::vector<center_of>& center_list,
    std::unordered_map<std::size_t, vertex<Real, NumDimensions>> const& map) {
  auto const num_parts       = total_num_particles(center_list);
  auto       center_vertices = std::vector<vertex<Real, NumDimensions>>(num_parts);
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
}  // namespace tatooine::detail::autonomous_particle
//==============================================================================
#endif
