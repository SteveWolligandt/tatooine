#ifndef TATOOINE_QUADTREE_H
#define TATOOINE_QUADTREE_H

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <limits>
#include <memory>
#include <unordered_set>
#include "boundingbox.h"
#include "intersection.h"
#include "mesh.h"
#include "type_traits.h"

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, typename Mesh>
class quadtree {
 public:
  static constexpr size_t num_dimensions() { return 2; }
  using real_t       = Real;
  using mesh_t       = Mesh;
  using this_t       = quadtree<Real, Mesh>;
  using pos_t        = vec<Real, 2>;
  using node_array_t = std::array<std::unique_ptr<quadtree<Real, Mesh>>, 4>;
  using vert         = typename Mesh::vertex;
  using edge         = typename Mesh::edge;
  using face         = typename Mesh::face;
  using bb_t         = boundingbox<Real, 2>;

  using iterator       = typename node_array_t::iterator;
  using const_iterator = typename node_array_t::const_iterator;

  constexpr static auto real_max = std::numeric_limits<Real>::max();

  //===========================================================================
 private:
  const Mesh*    m_mesh;
  bb_t           m_bb;
  std::set<vert> m_vertices;
  std::set<edge> m_edges;
  std::set<face> m_faces;
  bool           m_splitted;
  size_t         m_depth;
  size_t         m_max_depth;
  node_array_t   m_nodes;


  //===========================================================================
 public:
  /// for specifying search space
  template <size_t i>
  class spatial_constraint {
   public:
    spatial_constraint(const this_t& o, Real min, Real max)
        : m_min{min}, m_max{max}, m_quadtree{&o} {}

    //--------------------------------------------------------------------------
    /// check if a point x is in bounding of the constraint constrained
    bool operator()(const pos_t& x) { return m_min <= x(i) && x(i) < m_max; }

    //--------------------------------------------------------------------------
    /// check if a bounding intersects the bounding of the spatial constraint
    bool operator()(const bb_t& bb) {
      return (m_min <= bb.min(i) && bb.min(i) < m_max) ||
             (m_min <= bb.max(i) && bb.max(i) < m_max) ||
             (bb.min(i) <= m_min && m_max <= bb.max(i));
    }

   private:
    Real          m_min, m_max;
    const this_t* m_quadtree;
  };

  using x_constraint = spatial_constraint<0>;
  using y_constraint = spatial_constraint<1>;

  auto make_x_constraint(Real min, Real max) const {
    return x_constraint(*this, min, max);
  }
  auto make_y_constraint(Real min, Real max) const {
    return y_constraint(*this, min, max);
  }

  //===========================================================================
  enum X : unsigned int { LEFT = 0, RIGHT = 1 };
  enum Y : unsigned int { BOTTOM = 0, TOP = 2 };
  enum node_id : unsigned int {
    LEFT_BOTTOM  = LEFT + BOTTOM,
    RIGHT_BOTTOM = RIGHT + BOTTOM,
    LEFT_TOP     = LEFT + TOP,
    RIGHT_TOP    = RIGHT + TOP,
  };

  constexpr auto to_node_id(X x, Y y) {
    return (node_id)((unsigned int)x + (unsigned int)y);
  }

  //===========================================================================
  /// creates an quadtree of points of a triangular mesh
  quadtree(const Mesh& m, size_t max_depth = std::numeric_limits<size_t>::max(),
           size_t depth = 0)
      : m_mesh{&m},
        m_bb{pos_t{-real_max, -real_max}, pos_t{real_max, real_max}},
        m_splitted{false},
        m_depth{depth},
        m_max_depth{max_depth} {}

  //----------------------------------------------------------------------------
  /// creates an quadtree of a vector of points.
  /// \param[in]  min_boundary  minimum bb point
  /// \param[in]  max_boundary  maximum bb point
  /// \param[in]  depth  quadtree depth
  quadtree(const Mesh& m, const bb_t& bb,
           size_t max_depth = std::numeric_limits<size_t>::max(),
           size_t depth     = 0)
      : m_mesh{&m},
        m_bb{bb},
        m_splitted{false},
        m_depth{depth},
        m_max_depth{max_depth} {}

  //----------------------------------------------------------------------------
  quadtree(const quadtree& other)
      : m_mesh{other.m_mesh},
        m_bb{other.m_bb},
        m_vertices{other.m_vertices},
        m_edges{other.m_edges},
        m_faces{other.m_faces},
        m_splitted{other.m_splitted},
        m_depth{other.m_depth},
        m_max_depth{other.m_max_depth} {
    for (size_t i = 0; i < 4; ++i)
      if (other.m_nodes[i])
        m_nodes[i] = std::make_unique<this_t>(*other.m_nodes[i]);
  }

  //----------------------------------------------------------------------------
  quadtree(quadtree&& other)
      : m_mesh{other.m_mesh},
        m_bb{std::move(other.m_bb)},
        m_vertices{std::move(other.m_vertices)},
        m_edges{std::move(other.m_edges)},
        m_faces{std::move(other.m_faces)},
        m_splitted{other.m_splitted},
        m_depth{other.m_depth},
        m_max_depth{other.m_max_depth},
        m_nodes{std::move(other.m_nodes)} {}

  //----------------------------------------------------------------------------
  auto& operator=(const quadtree& other) {
    m_mesh      = other.m_mesh;
    m_bb        = other.m_bb;
    m_vertices  = other.m_vertices;
    m_edges     = other.m_edges;
    m_faces     = other.m_faces;
    m_splitted  = other.m_splitted;
    m_depth     = other.m_depth;
    m_max_depth = other.m_max_depth;
    for (size_t i = 0; i < 4; ++i)
      if (other.m_nodes[i])
        m_nodes[i] = std::make_unique<this_t>(*other.m_nodes[i]);
    return *this;
  }

  //----------------------------------------------------------------------------
  auto& operator=(quadtree&& other) {
    m_mesh      = other.m_mesh;
    m_bb        = std::move(other.m_bb);
    m_vertices  = std::move(other.m_vertices);
    m_edges     = std::move(other.m_edges);
    m_faces     = std::move(other.m_faces);
    m_splitted  = other.m_splitted;
    m_depth     = other.m_depth;
    m_max_depth = other.m_max_depth;
    m_nodes     = std::move(other.m_nodes);
    return *this;
  }

  //----------------------------------------------------------------------------
  auto&       operator[](node_id i) { return at(i); }
  const auto& operator[](node_id i) const { return at(i); }

  //----------------------------------------------------------------------------
  auto&       operator[](size_t i) { return at(node_id(i)); }
  const auto& operator[](size_t i) const { return at(node_id(i)); }

  //----------------------------------------------------------------------------
  auto&       at(size_t i) { return at(node_id(i)); }
  const auto& at(size_t i) const { return at(node_id(i)); }

  //----------------------------------------------------------------------------
  /// lazy get of node
  auto& at(node_id i) {
    if (!m_nodes[i])
      m_nodes[i] = std::make_unique<this_t>(*m_mesh, bounding(i), m_max_depth,
                                            m_depth + 1);
    return *m_nodes[i];
  }

  //----------------------------------------------------------------------------
  const auto& at(node_id i) const {
    if (!m_nodes[i])
      throw std::runtime_error("node " + std::to_string(i) + " not created");
    return *m_nodes[i];
  }

  //----------------------------------------------------------------------------
  /// pos is positioned left, right or on center depending of center c
  auto x_pos(const pos_t& pos, const pos_t& c) const {
    if (pos(0) <= c(0)) return LEFT;
    return RIGHT;
  }

  //----------------------------------------------------------------------------
  /// pos is positioned bottom, top or on center depending of center c
  auto y_pos(const pos_t& pos, const pos_t& c) const {
    if (pos(1) <= c(1)) return BOTTOM;
    return TOP;
  }

  //----------------------------------------------------------------------------
  iterator begin() { return m_nodes.begin(); }
  iterator end() { return m_nodes.end(); }

  //----------------------------------------------------------------------------
  const_iterator begin() const { return m_nodes.begin(); }
  const_iterator end() const { return m_nodes.end(); }

  //----------------------------------------------------------------------------
  void clear() {
    for (auto& node : m_nodes)
      if (node) node.reset();
  }

  //----------------------------------------------------------------------------
  /// calculates min and max bounding points from edgeset
  void calc_boundaries() {
    m_bb.reset();
    for (const auto& v : m_mesh->vertices()) m_bb += m_mesh->at(v);
  }

  //----------------------------------------------------------------------------
  auto depth() const { return m_depth; }

  //----------------------------------------------------------------------------
  auto max_depth() const { return m_max_depth; }

  //----------------------------------------------------------------------------
  const auto& mesh() const { return *m_mesh; }

  //----------------------------------------------------------------------------
  constexpr auto center() const { return m_bb.center(); }

  //----------------------------------------------------------------------------
  constexpr auto center_x() const {
    return (m_bb.min(0) + m_bb.max(0)) * Real(0.5);
  }

  //----------------------------------------------------------------------------
  constexpr auto center_y() const {
    return (m_bb.min(1) + m_bb.max(1)) * Real(0.5);
  }

  //----------------------------------------------------------------------------
  bool has_child(node_id i) const { return *m_nodes[i]; }
  bool has_left_bottom() const { return has_child(LEFT_BOTTOM); }
  bool has_right_bottom() const { return has_child(RIGHT_BOTTOM); }
  bool has_left_top() const { return has_child(LEFT_TOP); }
  bool has_right_top() const { return has_child(RIGHT_TOP); }

  //----------------------------------------------------------------------------
  constexpr auto& bounding() const { return m_bb; }

  //----------------------------------------------------------------------------
  constexpr auto bounding(node_id i) const { return bounding(i, center()); }

  //----------------------------------------------------------------------------
  /// \param[in] c center of current bounding
  constexpr auto bounding(node_id i, const pos_t& c) const {
    if (m_nodes[i])
      return m_nodes[i]->bounding();
    else
      switch (i) {
        case LEFT_BOTTOM:
          return bb_t{pos_t{m_bb.min(0), m_bb.min(1)}, pos_t{c(0), c(1)}};
        case RIGHT_BOTTOM:
          return bb_t{pos_t{c(0), m_bb.min(1)}, pos_t{m_bb.max(0), c(1)}};
        case LEFT_TOP:
          return bb_t{pos_t{m_bb.min(0), c(1)}, pos_t{c(0), m_bb.max(1)}};
        default:
        case RIGHT_TOP:
          return bb_t{pos_t{c(0), c(1)}, pos_t{m_bb.max(0), m_bb.max(1)}};
      }
  }

  //----------------------------------------------------------------------------
  /// checks if a vertex is in node
  bool is_inside(vert v) const {
    return intersection::point_in_boundingbox(m_mesh->at(v), m_bb);
  }

  //----------------------------------------------------------------------------
  /// checks if a vertex is in node
  bool is_inside(const pos_t& pos) const {
    return intersection::point_in_boundingbox(pos, m_bb);
  }

  //----------------------------------------------------------------------------
  /// checks if a face is partially or completely in node.
  bool is_inside(edge e) const {
    return intersection::edge_in_boundingbox(*m_mesh, e, m_bb);
  }

  //----------------------------------------------------------------------------
  /// checks if a face is partially or completely in node.
  bool is_inside(face f) const {
    assert(m_mesh->num_vertices(f) == 3);
    return intersection::triangle_in_boundingbox(*m_mesh, f, m_bb);
  }

  //----------------------------------------------------------------------------
  void insert(vert v, bool edges = true, bool faces = true) {
    // no index in cell
    if (!m_splitted && m_vertices.empty()) { m_vertices.insert(v); }
    // node has vertex and is not splitted
    // -> split, add new and old index in children and distribute edges
    else if (!m_splitted && !m_vertices.empty()) {
      if (m_depth < m_max_depth) {
        split(v);
      } else {
        m_vertices.insert(v);
      }

      // already splitted and index already distributed
    } else /*if (splitted)*/ {
      insert_in_child(v);
    }

    if (edges) { insert_edges(v); }
    if (faces) { insert_faces(v); }
  }

  //----------------------------------------------------------------------------
  void insert_all_vertices(bool edges = true, bool faces = true) {
    for (const auto& v : m_mesh->vertices()) { insert(v, edges, faces); }
  }

  //----------------------------------------------------------------------------
  /// inserts a new edge
  void insert(edge e, bool no_check = false) {
    if (no_check || is_inside(e)) {
      if (!m_splitted) {
        m_edges.insert(e);
      } else {
        distribute_to_children(e);
      }
    }
  }

  //----------------------------------------------------------------------------
  /// inserts edges of vertex v
  void insert_edges(vert v, bool no_check = false) {
    for (auto e : m_mesh->edges(v)) { insert(e, no_check); }
  }

  //----------------------------------------------------------------------------
  void insert(face f, bool no_check = false) {
    if (no_check || is_inside(f)) {
      if (!m_splitted) {
        m_faces.insert(f);
      } else {
        distribute_to_children(f);
      }
    }
  }

  //----------------------------------------------------------------------------
  /// inserts faces of vertex v
  void insert_faces(vert v, bool no_check = false) {
    for (auto f : m_mesh->faces(v)) { insert(f, no_check); }
  }

  //===========================================================================
  const auto& local_vertices() const { return m_vertices; }
  const auto& local_edges() const { return m_edges; }
  const auto& local_faces() const { return m_faces; }
  auto        is_splitted() const { return m_splitted; }

  //----------------------------------------------------------------------------
  /// returns indices to all points in quadtree
  void vertices(std::set<vert>& pnts) const {
    using namespace boost;
    copy(m_vertices, std::inserter(pnts, pnts.end()));
    for (const auto& node : m_nodes) {
      if (node) { node->vertices(pnts); }
    }
  }


  //----------------------------------------------------------------------------
  /// returns indices to all points in quadtree
  auto vertices() const {
    std::set<vert> pnts;
    vertices(pnts);
    return pnts;
  }

  //----------------------------------------------------------------------------
  template <typename... Preds>
  void vertices(std::set<vert>& verts, Preds&&... preds) const {
    for (const auto& v : m_vertices)
      if ((preds(m_mesh->at(v)) && ...)) verts.insert(v);

    if ((preds(m_bb) && ...))
      for (const auto& node : *this)
        if (node) node->vertices(verts, std::forward<Preds>(preds)...);
  }

  //----------------------------------------------------------------------------
  template <typename... Preds>
  auto vertices(Preds&&... preds) const {
    std::set<vert> verts;
    vertices(verts, std::forward<Preds>(preds)...);
    return verts;
  }

  //----------------------------------------------------------------------------
  auto edges() const {
    auto all_edges = m_edges;
    edges(all_edges);
    return all_edges;
  }

  //----------------------------------------------------------------------------
  void edges(std::set<edge>& all_edges) const {
    using namespace boost;
    if (!m_edges.empty())
      copy(m_edges, std::inserter(all_edges, all_edges.end()));

    for (const auto& node : *this)
      if (node) node->edges(all_edges);
  }

  //----------------------------------------------------------------------------
  /// returns indices to all points in quadtree
  template <typename... Preds>
  auto edges(Preds&&... preds) const {
    std::set<edge> es;
    edges(es, std::forward<Preds>(preds)...);
    return es;
  }

  //----------------------------------------------------------------------------
  template <typename... Preds>
  void edges(std::set<edge>& es, Preds&&... preds) const {
    using namespace boost;
    if ((preds(m_bb) && ...)) {
      copy(m_edges, std::inserter(es, es.end()));
      for (auto& node : *this)
        if (node) node->edges(es, std::forward<Preds>(preds)...);
    }
  }

  //----------------------------------------------------------------------------
  auto faces() const {
    auto all_faces = m_faces;
    faces(all_faces);
    return all_faces;
  }

  //----------------------------------------------------------------------------
  std::vector<std::pair<face, vec<Real, 3>>>& faces(
      std::vector<std::pair<face, vec<Real, 3>>>& fs,
      const vec<Real, 2>&                         x) const {
    using namespace boost;
    using namespace boost::adaptors;
    if (is_inside(x)) {
      if (!m_splitted) {
        for (const auto f : m_faces)
          if (auto is = intersection::point_in_triangle(x, *m_mesh, f); is)
            fs.emplace_back(f, *is);

      } else
        for (size_t i = 0; i < 4; ++i)
          if (m_nodes[i]) m_nodes[i]->faces(fs, x);
    }
    return fs;
  }

  //----------------------------------------------------------------------------
  auto faces(const vec<Real, 2>& x) const {
    std::vector<std::pair<face, vec<Real, 3>>> fs;
    return faces(fs, x);
  }

  //----------------------------------------------------------------------------
  void faces(std::set<face>& all_faces) const {
    using namespace boost;
    if (!m_faces.empty())
      copy(m_faces, std::inserter(all_faces, all_faces.end()));

    for (const auto& node : *this)
      if (node) node->faces(all_faces);
  }

  //----------------------------------------------------------------------------
  /// returns indices to all points in quadtree
  template <typename... Preds>
  auto faces(Preds&&... preds) const {
    std::set<face> es;
    faces(es, std::forward<Preds>(preds)...);
    return es;
  }

  //----------------------------------------------------------------------------
  template <typename... Preds>
  void faces(std::set<face>& fs, Preds&&... preds) const {
    using namespace boost;
    if ((preds(m_bb) && ...)) {
      copy(m_faces, std::inserter(fs, fs.end()));
      for (auto& node : *this)
        if (node) node->faces(fs, std::forward<Preds>(preds)...);
    }
  }
  //===========================================================================
  void split(vert v) {
    m_splitted = true;

    // insert in children
    for (const auto& v : m_vertices) insert_in_child(v);
    insert_in_child(v);
    m_vertices.clear();

    distribute_edges_to_children();
    distribute_faces_to_children();
  }

  //----------------------------------------------------------------------------
  /// calculates node index, creates node if necessary and inserts v into
  /// child
  void insert_in_child(vert v) {
    auto         c   = center();
    const pos_t& pos = m_mesh->at(v);
    auto         i   = to_node_id(x_pos(pos, c), y_pos(pos, c));
    at(i).insert(v, false, false);
  }

  //----------------------------------------------------------------------------
  /// distributes all edges of node to children and clears this node from
  /// edges
  void distribute_edges_to_children() {
    for (auto e : m_edges) distribute_to_children(e);
    m_edges.clear();
  }

  //----------------------------------------------------------------------------
  /// gets called after a split
  void distribute_to_children(edge e) {
    for (unsigned int i = 0; i < 4; ++i) {
      if (intersection::edge_in_boundingbox(*m_mesh, e, bounding(node_id(i)))) {
        at(node_id(i)).insert(e, true);
      }
    }
  }

  //----------------------------------------------------------------------------
  /// distributes all faces of node to children and clears this node from
  /// edges
  void distribute_faces_to_children() {
    for (auto f : m_faces) { distribute_to_children(f); }
    m_faces.clear();
  }

  //----------------------------------------------------------------------------
  /// gets called after a split
  void distribute_to_children(face f) {
    for (unsigned int i = 0; i < 4; ++i) {
      if (intersection::triangle_in_boundingbox(*m_mesh, f, bounding(node_id(i)))) {
        at(node_id(i)).insert(f, true);
      }
    }
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Mesh>
quadtree(const Mesh&, size_t max_depth, size_t depth)
    ->quadtree<typename Mesh::real_t, Mesh>;
template <typename Mesh>
quadtree(const Mesh&, size_t max_depth)
    ->quadtree<typename Mesh::real_t, Mesh>;
template <typename Mesh>
quadtree(const Mesh&)
    ->quadtree<typename Mesh::real_t, Mesh>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Mesh, typename BBReal>
quadtree(const Mesh& m, const boundingbox<BBReal, 2>& bb, size_t max_depth,
         size_t depth)
    ->quadtree<typename Mesh::real_t, Mesh>;
template <typename Mesh, typename BBReal>
quadtree(const Mesh& m, const boundingbox<BBReal, 2>& bb, size_t max_depth)
    ->quadtree<typename Mesh::real_t, Mesh>;
template <typename Mesh, typename BBReal>
quadtree(const Mesh& m, const boundingbox<BBReal, 2>& bb)
    ->quadtree<typename Mesh::real_t, Mesh>;

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
