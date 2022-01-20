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
  std::uint64_t id;
  std::uint64_t parent;
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
  hierarchy(std::size_t const id_)
      : id{id_} {}
  //----------------------------------------------------------------------------
  /// as child node
  hierarchy(std::size_t const id_, std::vector<hierarchy_pair> const& hps,
            std::unordered_map<std::size_t, vertex_type> const& centers,
            edgeset_type const&                                 edges)
      : hierarchy{id_} {
    build(hps, centers, edges);
    if (!leafs.empty()) {
      sort_leafs(edges);
      center = leafs[leafs.size() / 2].center;
    }
  }
  //----------------------------------------------------------------------------
  /// as top node
  hierarchy(std::vector<hierarchy_pair> const&                  hps,
            std::unordered_map<std::size_t, vertex_type> const& centers,
            edgeset_type const&                                 edges) {
    for (auto const& hp : hps) {
      // search for top nodes
      if (hp.id == hp.parent) {
        leafs.emplace_back(hp.parent, hps, centers, edges);
      }
    }
  }
  //----------------------------------------------------------------------------
  auto find_by_id(std::uint64_t const& id) const -> auto const& {
    for (auto const& l : leafs) {
      if (id == l.id) {
        return l;
      }
    }
    return *this;
  }
 private:
  auto build(std::vector<hierarchy_pair> const&                  hps,
             std::unordered_map<std::size_t, vertex_type> const& centers,
             edgeset_type const& edges) -> void {
    for (auto const& hp : hps) {
      if (id == hp.parent && hp.parent != hp.id) {
        leafs.emplace_back(hp.id, hps, centers, edges);
      }
    }
    if (leafs.empty()) {
      center = centers.at(id);
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
  if (std::abs(cos_angle(offset_dir, split_dir)) <
      std::cos(50.0 * M_PI / 180.0)) {
    // copy all if center-center direction is perpendicular split direction
    for (auto const& l : h.leafs) {
      copy(get_front(l, edges, other_center, offset_dir),
           std::back_inserter(front));
    }
  } else {
    // copy only nearest particle to front if center-center direction is parallel
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
  if (dot(edges[front0.back()] - edges[front0.front()],
          edges[front1.back()] - edges[front1.front()]) < 0) {
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
/// Triangulates two particles
template <typename Real, std::size_t NumDimensions>
auto triangulate(edgeset<Real, NumDimensions>&         edges,
                 hierarchy<Real, NumDimensions> const& h0,
                 hierarchy<Real, NumDimensions> const& h1) -> void {
  connect_fronts(get_front(h0, edges, edges[h1.center],
                           edges[h0.center] - edges[h1.center]),
                 get_front(h1, edges, edges[h0.center],
                           edges[h1.center] - edges[h0.center]),
                 edges);
}
//------------------------------------------------------------------------------
/// Triangulates one single initial particle.
template <typename Real, std::size_t NumDimensions>
auto triangulate_initial(edgeset<Real, NumDimensions>&         edges,
                         hierarchy<Real, NumDimensions> const& h) -> void {
  using namespace std::ranges;
  if (!h.leafs.empty()) {
    for (auto const& l : h.leafs) {
      triangulate_initial(edges, l);
    }
    triangulate(edges, h.leafs[0], h.leafs[1]);
    triangulate(edges, h.leafs[2], h.leafs[1]);
  }
}
//------------------------------------------------------------------------------
/// Triangulates set of initial particles. Not connecting initial particles with
/// each other.
template <typename Real, std::size_t NumDimensions>
auto triangulate(edgeset<Real, NumDimensions>&         edgeset,
                 hierarchy<Real, NumDimensions> const& h) -> void {
  for (auto const& top_leaf : h.leafs) {
    triangulate_initial(edgeset, top_leaf);
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
//==============================================================================
}  // namespace tatooine::detail::autonomous_particle
//==============================================================================
#endif
