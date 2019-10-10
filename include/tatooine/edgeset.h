#ifndef TATOOINE_EDGESET_H
#define TATOOINE_EDGESET_H

#include "pointset.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
class edgeset;

//============================================================================
template <typename Real, size_t N>
struct edge_iterator
    : public boost::iterator_facade<edge_iterator<Real, N>, edge,
                                    boost::bidirectional_traversal_tag, edge> {
  edge_iterator(edge _e, const edgeset<Real, N>* _es) : e{_e}, es{_es} {}
  edge_iterator(const edge_iterator& other) : e{other.e}, es{other.es} {}

 private:
  edge                    e;
  const edgeset<Real, N>* es;
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
template <typename Real, size_t N>
struct edge_container {
  using iterator       = edge_iterator<Real, N>;
  using const_iterator = edge_iterator<Real, N>;
  const edgeset<Real, N>* m_edgeset;
  auto                    begin() const {
    edge_iterator<Real, N> ei{edge{0}, m_edgeset};
    while (!m_edgeset->is_valid(*ei)) ++ei;
    return ei;
  }
  auto end() const {
    return iterator{
        edge{m_edgeset->num_edges() + m_edgeset->num_invalid_edges()},
        m_edgeset};
  }
};

//============================================================================
template <typename Real, size_t N>
class edgeset : public pointset<Real, N> {
 public:
  using this_t   = edgeset<Real, N>;
  using parent_t = pointset<Real, N>;

  using typename parent_t::pos_t;

  using parent_t::at;
  using parent_t::operator[];
  using parent_t::is_valid;
  using parent_t::remove;

  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  using parent_t::vertices;

  //============================================================================
  template <typename T>
  struct edge_property_t : public property_type<T> {
    // inherit all constructors
    using property_type<T>::property_type;
    auto&       at(edge e) { return property_type<T>::at(e.i); }
    const auto& at(edge e) const { return property_type<T>::at(e.i); }
    auto&       operator[](edge e) { return property_type<T>::operator[](e.i); }
    const auto& operator[](edge e) const {
      return property_type<T>::operator[](e.i);
    }
    std::unique_ptr<property> clone() const override {
      return std::unique_ptr<edge_property_t<T>>(new edge_property_t<T>{*this});
    }
  };

  //============================================================================
  using vertex_edge_link_t = vertex_property_t<std::vector<edge>>;

  //============================================================================
 protected:
  std::vector<std::array<vertex, 2>>               m_edges;
  std::vector<edge>                                m_invalid_edges;
  std::map<std::string, std::unique_ptr<property>> m_edge_properties;
  vertex_edge_link_t                               m_edges_of_vertices;

  //============================================================================
 public:
  edgeset() = default;

  //----------------------------------------------------------------------------
  edgeset(std::initializer_list<pos_t>&& vertices)
      : parent_t(std::move(vertices)) {}

 public:
  //----------------------------------------------------------------------------
  edgeset(const edgeset& other)
      : parent_t{other},
        m_edges{other.m_edges},
        m_invalid_edges{other.m_invalid_edges},
        m_edges_of_vertices{other.m_edges_of_vertices} {
    m_edge_properties.clear();
    for (const auto& [name, prop] : other.m_edge_properties) {
      m_edge_properties[name] = prop->clone();
    }
  }

  //----------------------------------------------------------------------------
  edgeset(edgeset&& other)
      : parent_t{std::move(other)},
        m_edges{std::move(other.m_edges)},
        m_invalid_edges{std::move(other.m_invalid_edges)},
        m_edge_properties{std::move(other.m_edge_properties)},
        m_edges_of_vertices{std::move(m_edges_of_vertices)} {}

  //----------------------------------------------------------------------------
  auto& operator=(const edgeset& other) {
    parent_t::operator=(other);
    m_edges           = other.m_edges;
    m_invalid_edges   = other.m_invalid_edges;
    for (const auto& [name, prop] : other.m_edge_properties) {
      m_edge_properties[name] = prop->clone();
    }
    m_edges_of_vertices = other.m_edges_of_vertices;
    return *this;
  }

  //----------------------------------------------------------------------------
  auto& operator=(edgeset&& other) {
    parent_t::operator  =(std::move(other));
    m_edges             = std::move(other.m_edges);
    m_invalid_edges     = std::move(other.m_invalid_edges);
    m_edge_properties   = std::move(other.m_edge_properties);
    m_edges_of_vertices = std::move(other.m_edges_of_vertices);
    return *this;
  }

  //----------------------------------------------------------------------------
  auto&       at(const edge& e) { return m_edges[e.i]; }
  const auto& at(const edge& e) const { return m_edges[e.i]; }

  //----------------------------------------------------------------------------
  auto&       operator[](const edge& e) { return at(e); }
  const auto& operator[](const edge& e) const { return at(e); }

  //----------------------------------------------------------------------------
  auto edges() const { return edge_container<Real, N>{this}; }

  //----------------------------------------------------------------------------
  auto&       edges(vertex v) { return m_edges_of_vertices.at(v); }
  const auto& edges(vertex v) const { return m_edges_of_vertices.at(v); }

  //----------------------------------------------------------------------------
  auto&       vertices(edge e) { return at(e); }
  const auto& vertices(edge e) const { return at(e); }

  //----------------------------------------------------------------------------
  template <typename... Ts, std::enable_if_t<sizeof...(Ts) == N>...,
            enable_if_arithmetic<Ts...>...>
  auto insert_vertex(Ts... ts) {
    auto v = parent_t::insert_vertex(ts...);
    m_edges_of_vertices.emplace_back();
    return v;
  }

  //----------------------------------------------------------------------------
  auto insert_vertex(const pos_t& x) {
    auto v = parent_t::insert_vertex(x);
    m_edges_of_vertices.emplace_back();
    return v;
  }

  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& x) {
    auto v = parent_t::insert_vertex(std::move(x));
    m_edges_of_vertices.emplace_back();
    return v;
  }

  //----------------------------------------------------------------------------
  auto insert_edge(size_t v0, size_t v1) {
    return insert_edge(vertex{v0}, vertex{v1});
  }

  //----------------------------------------------------------------------------
  auto insert_edge(vertex v0, vertex v1) {
    std::array                   new_edge{v0, v1};
    order_independent_edge_equal eq;
    for (auto e : edges()) {
      if (eq(at(e), new_edge)) { return e; }
    }

    edge e{m_edges.size()};
    m_edges.push_back(new_edge);
    for (auto v : vertices(e)) { edges(v).push_back(e); }
    for (auto& [key, prop] : m_edge_properties) { prop->push_back(); }
    return e;
  }

  //----------------------------------------------------------------------------
  void remove(vertex v) {
    for (auto e : edges(v)) { remove(e, false); }
    parent_t::remove(v);
  }

  //----------------------------------------------------------------------------
  void remove(edge e, bool remove_orphaned_vertices = true) {
    using namespace boost;
    if (is_valid(e)) {
      // remove edge link from vertices
      for (auto v : vertices(e)) { edges(v).erase(find(edges(v), e)); }

      if (remove_orphaned_vertices) {
        for (auto v : vertices(e)) {
          if (num_edges(v) == 0) { remove(v); }
        }
      }

      if (!contains(e, m_invalid_edges)) { m_invalid_edges.push_back(e); }
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
    for (auto invalid_v : this->m_invalid_vertices)
      for (auto e : edges())
        for (auto& v : vertices(e))
          if (v.i >= invalid_v.i) --v.i;

    // erase actual edges
    for (const auto invalid_e : m_invalid_edges) {
      m_edges.erase(m_edges.begin() + invalid_e.i);
      for (const auto& [key, prop] : m_edge_properties) {
        prop->erase(invalid_e.i);
      }

      // reindex deleted edge indices;
      for (auto& e_to_reindex : m_invalid_edges) {
        if (e_to_reindex.i > invalid_e.i) { --e_to_reindex.i; }
      }
    }

    m_invalid_edges.clear();

    for (const auto v : this->m_invalid_vertices) {
      // decrease deleted vertex indices;
      for (auto& v_to_decrease : this->m_invalid_vertices) {
        if (v_to_decrease.i > v.i) { --v_to_decrease.i; }
      }

      this->m_points.erase(this->m_points.begin() + v.i);
      for (const auto& [key, prop] : this->m_vertex_properties) {
        prop->erase(v.i);
      }
      m_edges_of_vertices.erase(v.i);
    }
    this->m_invalid_vertices.clear();

    // tidy up vertices
    parent_t::tidy_up();
  }

  //----------------------------------------------------------------------------
  constexpr bool is_valid(edge e) const {
    return !contains(e, m_invalid_edges);
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
  auto num_invalid_edges() const {
    return m_edges.size() - m_invalid_edges.size();
  }
  //----------------------------------------------------------------------------
  auto num_edges(vertex v) const { return edges(v).size(); }

  //----------------------------------------------------------------------------
  template <typename T>
  auto& edge_property(const std::string& name) {
    return *dynamic_cast<edge_property_t<T>*>(m_edge_properties.at(name).get());
  }

  //----------------------------------------------------------------------------
  template <typename T>
  const auto& edge_property(const std::string& name) const {
    return *dynamic_cast<edge_property_t<T>*>(m_edge_properties.at(name).get());
  }

  //----------------------------------------------------------------------------
  template <typename T>
  auto& add_edge_property(const std::string& name, const T& value = T{}) {
    auto [it, suc] = m_edge_properties.insert(
        std::pair{name, std::make_unique<edge_property_t<T>>(value)});
    auto prop = dynamic_cast<edge_property_t<T>*>(it->second.get());
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
