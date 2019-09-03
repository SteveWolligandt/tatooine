#ifndef TATOOINE_EDGESET_H
#define TATOOINE_EDGESET_H

#include "pointset.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
class edgeset : public pointset<Real, N> {
 public:
  using this_t   = edgeset<Real, N>;
  using parent_t = pointset<Real, N>;

  using typename parent_t::handle;
  using typename parent_t::pos_t;
  using typename parent_t::vertex;

  using parent_t::at;
  using parent_t::operator[];
  using parent_t::is_valid;
  using parent_t::remove;

  template <typename T>
  using vertex_prop = typename parent_t::template vertex_prop<T>;
  using parent_t::vertices;


  //============================================================================
  struct edge : handle {
    edge() = default;
    edge(size_t i) : handle{i} {}
    edge(const edge&)            = default;
    edge(edge&&)                 = default;
    edge& operator=(const edge&) = default;
    edge& operator=(edge&&)      = default;

    bool operator==(const edge& other) const { return this->i == other.i; }
    bool operator!=(const edge& other) const { return this->i != other.i; }
    bool operator<(const edge& other) const { return this->i < other.i; }
    static constexpr auto invalid() { return edge{handle::invalid_idx}; }
  };

  //============================================================================
  struct edge_iterator
      : public boost::iterator_facade<
            edge_iterator, edge, boost::bidirectional_traversal_tag, edge> {
    edge_iterator(edge _e, const edgeset* _es) : e{_e}, es{_es} {}
    edge_iterator(const edge_iterator& other) : e{other.e}, es{other.es} {}

   private:
    edge           e;
    const edgeset* es;
    friend class boost::iterator_core_access;

    void increment() {
      do
        ++e.i;
      while (!es->is_valid(e));
    }
    void decrement() {
      do
        --e.i;
      while (!es->is_valid(e));
    }

    bool equal(const edge_iterator& other) const { return e.i == other.e.i; }
    auto dereference() const { return e; }
  };

  //============================================================================
  struct edge_container {
    using iterator       = edge_iterator;
    using const_iterator = edge_iterator;
    const edgeset* m_edgeset;
    auto           begin() const {
      edge_iterator ei{edge{0}, m_edgeset};
      while (!m_edgeset->is_valid(*ei)) ++ei;
      return ei;
    }
    auto end() const {
      return edge_iterator{edge{m_edgeset->m_edges.size()}, m_edgeset};
    }
  };

  //============================================================================
  template <typename T>
  struct edge_prop : public property_type<T> {
    // inherit all constructors
    using property_type<T>::property_type;
    auto&       at(edge e) { return property_type<T>::at(e.i); }
    const auto& at(edge e) const { return property_type<T>::at(e.i); }
    auto&       operator[](edge e) { return property_type<T>::operator[](e.i); }
    const auto& operator[](edge e) const {
      return property_type<T>::operator[](e.i);
    }
    std::unique_ptr<property> clone() const override {
      return std::unique_ptr<edge_prop<T>>(new edge_prop<T>{*this});
    }
  };

  //============================================================================
  using vertex_edge_link_t = vertex_prop<std::vector<edge>>; 

  //============================================================================
 protected:
  std::vector<std::array<vertex, 2>>               m_edges;
  std::vector<edge>                                m_invalid_edges;
  std::map<std::string, std::unique_ptr<property>> m_edge_properties;
  vertex_edge_link_t* m_edges_of_vertices = nullptr;

  //============================================================================
 public:
  edgeset() {
      add_link_properties();
  }

  //----------------------------------------------------------------------------
  edgeset(std::initializer_list<pos_t>&& vertices)
      : parent_t(std::move(vertices)) {
      add_link_properties();
  }

#ifdef USE_TRIANGLE
  edgeset(const triangle::io& io) : parent_t(io) {
    add_link_properties();
  }
#endif

  // edgeset(const tetgenio& io) : parent_t{io} { add_link_properties(); }

 private:
  //----------------------------------------------------------------------------
  void add_link_properties() {
    m_edges_of_vertices = dynamic_cast<vertex_prop<std::vector<edge>>*>(
        &this->template add_vertex_property<std::vector<edge>>("v:edges"));
  }

  //----------------------------------------------------------------------------
  auto find_link_properties() {
    return dynamic_cast<vertex_prop<std::vector<edge>>*>(
        &this->template vertex_property<std::vector<edge>>("v:edges"));
  }

 public:
  //----------------------------------------------------------------------------
  edgeset(const edgeset& other)
      : parent_t{other},
        m_edges{other.m_edges},
        m_invalid_edges{other.m_invalid_edges} {
    m_edge_properties.clear();
    for (const auto& [name, prop] : other.m_edge_properties)
      m_edge_properties[name] = prop->clone();
    m_edges_of_vertices = find_link_properties();
  }

  //----------------------------------------------------------------------------

  edgeset(edgeset&& other)
      : parent_t{std::move(other)},
        m_edges{std::move(other.m_edges)},
        m_invalid_edges{std::move(other.m_invalid_edges)},
        m_edge_properties{std::move(other.m_edge_properties)},
        m_edges_of_vertices{find_link_properties()} {}

  //----------------------------------------------------------------------------

  auto& operator=(const edgeset& other) {
    parent_t::operator=(other);
    m_edges           = other.m_edges;
    m_invalid_edges   = other.m_invalid_edges;
    for (const auto& [name, prop] : other.m_edge_properties)
      m_edge_properties[name] = prop->clone();
    m_edges_of_vertices = find_link_properties();
    return *this;
  }

  //----------------------------------------------------------------------------

  auto& operator=(edgeset&& other) {
    parent_t::operator=(std::move(other));
    m_edges           = std::move(other.m_edges);
    m_invalid_edges   = std::move(other.m_invalid_edges);
    m_edge_properties = std::move(other.m_edge_properties);
    m_edges_of_vertices =
      find_link_properties();
    return *this;
  }

  //----------------------------------------------------------------------------
  auto&       at(const edge& e) { return m_edges[e.i]; }
  const auto& at(const edge& e) const { return m_edges[e.i]; }

  //----------------------------------------------------------------------------
  auto&       operator[](const edge& e) { return at(e); }
  const auto& operator[](const edge& e) const { return at(e); }

  //----------------------------------------------------------------------------
  auto edges() const { return edge_container{this}; }

  //----------------------------------------------------------------------------
  auto&       edges(vertex v) { return m_edges_of_vertices->at(v); }
  const auto& edges(vertex v) const { return m_edges_of_vertices->at(v); }

  //----------------------------------------------------------------------------
  auto&       vertices(edge e) { return at(e); }
  const auto& vertices(edge e) const { return at(e); }

  //----------------------------------------------------------------------------
  auto insert_edge(size_t v0, size_t v1) {
    return insert_edge(vertex{v0}, vertex{v1});
  }

  //----------------------------------------------------------------------------
  auto insert_edge(vertex v0, vertex v1) {
    std::array new_edge{v0, v1};
    order_independent_edge_equal eq;
    for (auto e : edges())
      if (eq(at(e), new_edge)) return e;

    edge e{m_edges.size()};
    m_edges.push_back(new_edge);
    for (auto v : vertices(e)) edges(v).push_back(e);
    for (auto& [key, prop] : m_edge_properties) prop->push_back();
    return e;
  }

  //----------------------------------------------------------------------------
  void remove(vertex v) {
    parent_t::remove(v);
    for (auto e : edges(v)) remove(e);
  }

  //----------------------------------------------------------------------------
  void remove(edge e, bool remove_orphaned_vertices = true) {
    using namespace boost;
    if (is_valid(e)) {
      // remove edge link from vertices
      for (auto v : vertices(e)) edges(v).erase(find(edges(v), e));

      if (remove_orphaned_vertices)
        for (auto v : vertices(e))
          if (num_edges(v) == 0) remove(v);

      if (find(m_invalid_edges, e) == m_invalid_edges.end())
        m_invalid_edges.push_back(e);
    }
  }

  //----------------------------------------------------------------------------
  //! tidies up invalid vertices and edges
  void tidy_up() {
    using namespace boost;

    // decrease edge-index of vertices whose indices are greater than an invalid
    // edge-index
    for (const auto invalid_e : m_invalid_edges)
      for (const auto v : vertices())
        for (auto& e : edges(v))
          if (e.i >= invalid_e.i) --e.i;

    // reindex edge's vertex indices
    for (auto invalid_v : this->m_invalid_points)
      for (auto e : edges())
        for (auto& v : vertices(e))
          if (v.i >= invalid_v.i) --v.i;

    // erase actual edges
    for (const auto invalid_e : m_invalid_edges) {
      m_edges.erase(m_edges.begin() + invalid_e.i);
      for (const auto& [key, prop] : m_edge_properties)
        prop->erase(invalid_e.i);

      // reindex deleted edge indices;
      for (auto& e_to_reindex : m_invalid_edges)
        if (e_to_reindex.i > invalid_e.i) --e_to_reindex.i;
    }

    m_invalid_edges.clear();

    // tidy up vertices
    parent_t::tidy_up();
  }

  //----------------------------------------------------------------------------
  constexpr bool is_valid(edge e) const {
    return boost::find(m_invalid_edges, e) == m_invalid_edges.end();
  }

  //----------------------------------------------------------------------------
  void clear_edges() {
    m_edges.clear();
    m_edges.shrink_to_fit();
    m_invalid_edges.clear();
    m_invalid_edges.shrink_to_fit();
    for (auto& [key, val] : m_edge_properties) val->clear();
  }

  //----------------------------------------------------------------------------
  void clear() {
    parent_t::clear();
    clear_edges();
  }

  //----------------------------------------------------------------------------
  auto num_edges() const { return m_edges.size() - m_invalid_edges.size(); }

  //----------------------------------------------------------------------------
  auto num_edges(vertex v) const { return edges(v).size(); }

  //----------------------------------------------------------------------------
  template <typename T>
  auto& edge_property(const std::string& name) {
    return *dynamic_cast<edge_prop<T>*>(m_edge_properties.at(name).get());
  }

  //----------------------------------------------------------------------------
  template <typename T>
  const auto& edge_property(const std::string& name) const {
    return *dynamic_cast<edge_prop<T>*>(m_edge_properties.at(name).get());
  }

  //----------------------------------------------------------------------------
  template <typename T>
  auto& add_edge_property(const std::string& name, const T& value = T{}) {
    auto [it, suc] = m_edge_properties.insert(
        std::pair{name, std::make_unique<edge_prop<T>>(value)});
    auto prop = dynamic_cast<edge_prop<T>*>(it->second.get());
    prop->resize(m_edges.size());
    return *prop;
  }

  //============================================================================
  struct order_independent_edge_equal {
    bool operator()(const std::array<vertex, 2>& lhs,
                    const std::array<vertex, 2>& rhs) const {
      return (lhs[0] == rhs[0] && lhs[1] == rhs[1]) ||
             (lhs[0] == rhs[1] && lhs[1] == rhs[0]);
    }
  };

  //----------------------------------------------------------------------------
  struct order_independent_edge_compare {
    bool operator()(const std::array<vertex, 2>& lhs,
                    const std::array<vertex, 2>& rhs) const {
      auto min_lhs = std::min(lhs[0], lhs[1]);
      auto min_rhs = std::min(rhs[0], rhs[1]);
      auto max_lhs = std::max(lhs[0], lhs[1]);
      auto max_rhs = std::max(rhs[0], rhs[1]);
      if (min_lhs == min_rhs)
        return max_lhs < max_rhs;
      else
        return min_lhs < min_rhs;
    }
  };

  //----------------------------------------------------------------------------
  struct order_dependent_edge_equal {
    auto operator()(const std::array<vertex, 2>& lhs,
                    const std::array<vertex, 2>& rhs) const {
      return lhs[0] == rhs[0] && lhs[1] == rhs[1];
    }
  };

  //----------------------------------------------------------------------------
  struct order_dependent_edge_compare {
    bool operator()(const std::array<vertex, 2>& lhs,
                    const std::array<vertex, 2>& rhs) const {
      if (lhs[0] == rhs[0])
        return lhs[1] < rhs[1];
      else
        return lhs[0] < rhs[0];
    }
  };
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
