#ifndef TATOOINE_LINE_H
#define TATOOINE_LINE_H

#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/algorithm/reverse.hpp>
#include <boost/range/algorithm_ext/iota.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/numeric.hpp>
#include <cassert>
#include <deque>
#include <map>
#include <stdexcept>

#include "handle.h"
#include "linspace.h"
#include "property.h"
#include "tensor.h"
#include "vtk_legacy.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
struct line;

struct automatic_t {};
static constexpr automatic_t automatic;

struct forward_t {};
static constexpr forward_t forward;

struct backward_t {};
static constexpr backward_t backward;

struct central_t {};
static constexpr central_t central;

struct quadratic_t {};
static constexpr quadratic_t quadratic;

//==============================================================================
// Iterators
//==============================================================================
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
struct const_line_vertex_iterator
    : boost::iterator_facade<
          const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>,
          Value, boost::bidirectional_traversal_tag, Reference> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t =
      const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;

  //============================================================================
  // ctors
  //============================================================================
  const_line_vertex_iterator(Handle handle, const Line& l, bool prefer_calc)
      : m_handle{handle}, m_line{l}, m_prefer_calc{prefer_calc} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const_line_vertex_iterator(const const_line_vertex_iterator& other) = default;

  //============================================================================
  // members
  //============================================================================
 private:
  Handle      m_handle;
  const Line& m_line;
  bool        m_prefer_calc;

  //============================================================================
  // iterator_face
  //============================================================================
 private:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  void increment() { ++m_handle; }
  void decrement() { --m_handle; }
  //----------------------------------------------------------------------------
  auto equal(const const_line_vertex_iterator& other) const {
    return m_handle == other.m_handle;
  }
  //----------------------------------------------------------------------------
  const Reference dereference() const { return m_line.at(m_handle, m_prefer_calc); }
  //============================================================================
  // methods
  //============================================================================
 public:
  this_t next(size_t inc = 1) const {
    this_t n = *this;
    n.m_handle.i += inc;
    return n;
  }
  //----------------------------------------------------------------------------
  this_t prev(size_t dec = 1) const {
    this_t p = *this;
    p.m_handle.i -= dec;
    return p;
  }
  //----------------------------------------------------------------------------
  auto& advance(const size_t inc = 1) const {
    m_handle.i += inc;
    return *this;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
static auto next(const const_line_vertex_iterator<Line, Real, N, Handle, Value,
                                                  Reference>& it,
                 size_t                                       inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto prev(const const_line_vertex_iterator<Line, Real, N, Handle, Value,
                                           Reference>& it,
          size_t                                       dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto& advance(
    const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
    size_t inc = 1) {
  return it.advance(inc);
}
//==============================================================================
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference = Value&>
struct line_vertex_iterator
    : boost::iterator_facade<
          line_vertex_iterator<Line, Real, N, Handle, Value, Reference>, Value,
          boost::bidirectional_traversal_tag, Reference> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;

  //============================================================================
  // ctors
  //============================================================================
  line_vertex_iterator(Handle handle, Line& l) : m_handle{handle}, m_line{l} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line_vertex_iterator(const line_vertex_iterator& other) = default;

  //============================================================================
  // members
  //============================================================================
 private:
  Handle m_handle;
  Line&  m_line;
  bool   m_prefer_calc;

  //============================================================================
  // iterator_facade
  //============================================================================
 private:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  void increment() { ++m_handle; }
  void decrement() { --m_handle; }
  //----------------------------------------------------------------------------
  auto equal(const line_vertex_iterator& other) const {
    return m_handle == other.m_handle;
  }
  //----------------------------------------------------------------------------
  Reference dereference() { return m_line.at(m_handle, m_prefer_calc); }

  //============================================================================
  // methods
  //============================================================================
 public:
  this_t next(const size_t inc = 1) const {
    this_t n = *this;
    n.m_handle.i += inc;
    return n;
  }
  this_t prev(const size_t dec = 1) const {
    this_t p = *this;
    p.m_handle.i -= dec;
    return p;
  }
  auto& advance(const size_t inc = 1) const {
    m_handle.i += inc;
    return *this;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto next(
    const line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
    size_t inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto prev(
    const line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
    size_t dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto& advance(line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
              size_t inc = 1) {
  return it.advance(inc);
}
//==============================================================================
// Container
//==============================================================================
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference = Value&>
struct const_line_vertex_container {
  //============================================================================
  // typedefs
  //============================================================================
  using iterator =
      line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;
  using const_iterator =
      const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;

  //============================================================================
  // members
  //============================================================================
  const Line& m_line;
  bool m_prefer_calc;

  //============================================================================
  // methods
  //============================================================================
  auto begin() const { return const_iterator{Handle{0}, m_line, m_prefer_calc}; }
  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto end() const {
    return const_iterator{Handle{m_line.num_vertices()}, m_line, m_prefer_calc};
  }
  //--------------------------------------------------------------------------
  const auto& front() const { return m_line.at(Handle{0}, m_prefer_calc); }
  //--------------------------------------------------------------------------
  const auto& back() const {
    return m_line.at(Handle{m_line.num_vertices() - 1}, m_prefer_calc);
  }
  //--------------------------------------------------------------------------
  const auto& line() const { return m_line; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto begin(const const_line_vertex_container<Line, Real, N, Handle, Value,
                                             Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto end(const const_line_vertex_container<Line, Real, N, Handle, Value,
                                           Reference>& it) {
  return it.begin();
}
//============================================================================
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference = Value&>
struct line_vertex_container {
  //============================================================================
  // typedefs
  //============================================================================
  using iterator =
      line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;
  using const_iterator =
      const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;

  //============================================================================
  // members
  //============================================================================
  Line& m_line;
  bool m_prefer_calc;

  //============================================================================
  // methods
  //============================================================================
  auto begin() const { return const_iterator{Handle{0}, m_line, m_prefer_calc}; }
  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto begin() { return iterator{Handle{0}, m_line, m_prefer_calc}; }
  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto end() const {
    return const_iterator{Handle{m_line.num_vertices()}, m_line, m_prefer_calc};
  }
  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto end() { return iterator{Handle{m_line.num_vertices()}, m_line, m_prefer_calc}; }
  //----------------------------------------------------------------------------
  const auto& front() const { return m_line.at(Handle{0}, m_prefer_calc); }
  auto&       front() { return m_line.at(Handle{0}, m_prefer_calc); }
  //----------------------------------------------------------------------------
  const auto& back() const {
    return m_line.at(Handle{m_line.num_vertices() - 1}, m_prefer_calc);
  }
  auto& back() { return m_line.at(Handle{m_line.num_vertices() - 1}, m_prefer_calc); }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto begin(
    const line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto begin(line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto end(
    const line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value, typename Reference>
auto end(line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
//============================================================================
// line implementation
//============================================================================
template <typename Real, size_t N>
struct line {
  struct empty_exception : std::exception {};
  //============================================================================
  using real_t          = Real;
  using vec_t           = vec<Real, N>;
  using pos_t           = vec_t;
  using vec3            = vec<Real, 3>;
  using mat3            = mat<Real, 3, 3>;
  using this_t          = line<Real, N>;
  using pos_container_t = std::deque<pos_t>;

  //============================================================================
  // Handles
  //============================================================================
  struct vertex_idx : handle {
    using handle::handle;
    bool operator==(vertex_idx other) const { return this->i == other.i; }
    bool operator!=(vertex_idx other) const { return this->i != other.i; }
    bool operator<(vertex_idx other) const { return this->i < other.i; }
    static constexpr auto invalid() { return vertex_idx{handle::invalid_idx}; }
  };
  //----------------------------------------------------------------------------
  struct tangent_idx : handle {
    using handle::handle;
    bool operator==(tangent_idx other) const { return this->i == other.i; }
    bool operator!=(tangent_idx other) const { return this->i != other.i; }
    bool operator<(tangent_idx other) const { return this->i < other.i; }
    static constexpr auto invalid() { return tangent_idx{handle::invalid_idx}; }
  };

  template <typename T>
  using vertex_property_t = deque_property_impl<vertex_idx, T>;
  using vertex_property_container_t =
      std::map<std::string, std::unique_ptr<deque_property<vertex_idx>>>;

  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto num_dimensions() noexcept { return N; }

  //============================================================================
  // members
  //============================================================================
 private:
  pos_container_t m_vertices;
  bool            m_is_closed = false;

 protected:
  vertex_property_container_t      m_vertex_properties;
  vertex_property_t<vec<Real, N>>* m_tangents = nullptr;

  //============================================================================
 public:
  line() = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line(const line& other)
      : m_vertices{other.m_vertices}, m_is_closed{other.m_is_closed} {
    for (auto& [name, prop] : other.m_vertex_properties) {
      m_vertex_properties[name] = prop->clone();
    }
    if (other.m_tangents) {
      m_tangents = &vertex_property<vec<Real, N>>("tangents");
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line(line&& other) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line& operator=(const line& other) {
    m_vertices  = other.m_vertices;
    m_is_closed = other.m_is_closed;
    m_vertex_properties.clear();
    for (auto& [name, prop] : other.m_vertex_properties) {
      m_vertex_properties[name] = prop->clone();
    }
    if (other.m_tangents) {
      m_tangents = &vertex_property<vec<Real, N>>("tangents");
    }
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line& operator=(line&& other) = default;
  //----------------------------------------------------------------------------
  line(const pos_container_t& data, bool is_closed = false)
      : m_vertices{data}, m_is_closed{is_closed} {}
  //----------------------------------------------------------------------------
  line(pos_container_t&& data, bool is_closed = false)
      : m_vertices{std::move(data)}, m_is_closed{is_closed} {}
  //----------------------------------------------------------------------------
  line(std::initializer_list<pos_t>&& data)
      : m_vertices{std::move(data)}, m_is_closed{false} {}
  //----------------------------------------------------------------------------
  auto num_vertices() const { return m_vertices.size(); }
  //----------------------------------------------------------------------------
  auto empty() const { return m_vertices.empty(); }
  //----------------------------------------------------------------------------
  auto clear() { return m_vertices.clear(); }
  //============================================================================
  // vertex
  //============================================================================
  const auto& vertex_at(size_t i) const { return m_vertices[i]; }
  auto&       vertex_at(size_t i) { return m_vertices[i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& vertex_at(vertex_idx i) const { return m_vertices[i.i]; }
  auto&       vertex_at(vertex_idx i) { return m_vertices[i.i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& at(vertex_idx i) const { return m_vertices[i.i]; }
  auto&       at(vertex_idx i) { return m_vertices[i.i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& operator[](vertex_idx i) const { return m_vertices[i.i]; }
  auto&       operator[](vertex_idx i) { return m_vertices[i.i]; }
  //----------------------------------------------------------------------------
  const auto& front_vertex() const { return m_vertices.front(); }
  auto&       front_vertex() { return m_vertices.front(); }
  //----------------------------------------------------------------------------
  const auto& back_vertex() const { return m_vertices.back(); }
  auto&       back_vertex() { return m_vertices.back(); }
  //----------------------------------------------------------------------------
  template <typename... Components, enable_if_arithmetic<Components...> = true,
            std::enable_if_t<sizeof...(Components) == N, bool> = true>
  auto push_back(Components... comps) {
    m_vertices.push_back(pos_t{static_cast<Real>(comps)...});
    for (auto& [name, prop] : m_vertex_properties) { prop->push_back(); }
    return vertex_idx{m_vertices.size() - 1};
  }
  auto push_back(const pos_t& p) {
    m_vertices.push_back(p);
    for (auto& [name, prop] : m_vertex_properties) { prop->push_back(); }
    return vertex_idx{m_vertices.size() - 1};
  }
  auto push_back(pos_t&& p) {
    m_vertices.emplace_back(std::move(p));
    for (auto& [name, prop] : m_vertex_properties) { prop->push_back(); }
    return vertex_idx{m_vertices.size() - 1};
  }
  auto pop_back() { m_vertices.pop_back(); }
  //----------------------------------------------------------------------------
  template <typename... Components, enable_if_arithmetic<Components...> = true,
            std::enable_if_t<sizeof...(Components) == N, bool> = true>
  auto push_front(Components... comps) {
    m_vertices.push_front(pos_t{static_cast<Real>(comps)...});
    for (auto& [name, prop] : m_vertex_properties) { prop->push_front(); }
    return vertex_idx{0};
  }
  auto push_front(const pos_t& p) {
    m_vertices.push_front(p);
    for (auto& [name, prop] : m_vertex_properties) { prop->push_front(); }
    return vertex_idx{0};
  }
  auto push_front(pos_t&& p) {
    m_vertices.emplace_front(std::move(p));
    for (auto& [name, prop] : m_vertex_properties) { prop->push_front(); }
    return vertex_idx{0};
  }
  void pop_front() { m_vertices.pop_front(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& vertices() const { return m_vertices; }
  auto&       vertices() { return m_vertices; }
  //============================================================================
  // tangent
  //============================================================================
  /// calculates tangent at point t with backward differences
  auto tangent_at(const tangent_idx i, forward_t tag) const {
    return tangent_at(i.i, tag);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// calculates tangent at point i with forward differences
  auto tangent_at(const size_t i, forward_t /*tag*/) const {
    assert(num_vertices() > 1);
    if (is_closed() && i == num_vertices() - 1) {
      return (front_vertex() - back_vertex()) /
             distance(front_vertex(), back_vertex());
    }
    return (vertex_at(i + 1) - vertex_at(i)) /
           distance(vertex_at(i), vertex_at(i + 1));
  }

  //----------------------------------------------------------------------------
  /// calculates tangent at point t with backward differences
  auto tangent_at(const tangent_idx i, backward_t tag) const {
    return tangent_at(i.i, tag);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// calculates tangent at point i with central differences
  auto tangent_at(const size_t i, backward_t /*tag*/) const {
    assert(num_vertices() > 1);
    if (is_closed() && i == 0) {
      return (front_vertex() - back_vertex()) /
             distance(back_vertex(), front_vertex());
    }
    return (vertex_at(i) - vertex_at(i - 1)) /
           distance(vertex_at(i), vertex_at(i - 1));
  }

  //----------------------------------------------------------------------------
  /// calculates tangent at point i with central differences
  auto tangent_at(const tangent_idx i, central_t tag) const {
    return tangent_at(i.i, tag);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// calculates tangent at point i with central differences
  auto tangent_at(const size_t i, central_t /*tag*/) const {
    if (is_closed()) {
      if (i == 0) {
        return (vertex_at(1) - back_vertex()) /
               (distance(back_vertex(), front_vertex()) +
                distance(front_vertex(), vertex_at(i + 1)));
      } else if (i == num_vertices() - 1) {
        return (front_vertex() - vertex_at(i - 1)) /
               (distance(vertex_at(i - 1), back_vertex()) +
                distance(back_vertex(), front_vertex()));
      }
    }
    return (vertex_at(i + 1) - vertex_at(i - 1)) /
           (distance(vertex_at(i - 1), vertex_at(i)) +
            distance(vertex_at(i), vertex_at(i + 1)));
  }
  //----------------------------------------------------------------------------
  auto tangent_at(const tangent_idx i, automatic_t tag,
                  bool prefer_calc = false) const {
    return tangent_at(i.i, tag, prefer_calc);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto tangent_at(const size_t i, automatic_t /*tag*/,
                  bool         prefer_calc = false) const {
    if (m_tangents && !prefer_calc) { return m_tangents->at(i); }
    if (is_closed()) { return tangent_at(i, central); }
    if (i == 0) { return tangent_at(i, forward); }
    if (i == num_vertices() - 1) { return tangent_at(i, backward); }
    return tangent_at(i, central);
  }
  //----------------------------------------------------------------------------
  auto tangent_at(const tangent_idx i, bool prefer_calc = false) const {
    return tangent_at(i, automatic, prefer_calc);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto tangent_at(const size_t i, bool prefer_calc = false) const {
    return tangent_at(i, automatic, prefer_calc);
  }
  //----------------------------------------------------------------------------
  auto front_tangent(bool prefer_calc = false) const {
    return tangent_at(0, prefer_calc);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto back_tangent(bool prefer_calc = false) const {
    return tangent_at(num_vertices() - 1, prefer_calc);
  }
  //----------------------------------------------------------------------------
  auto at(tangent_idx i, bool prefer_calc = false) const {
    return tangent_at(i.i, prefer_calc);
  }
  auto at(tangent_idx i, forward_t tag) const { return tangent_at(i.i, tag); }
  auto at(tangent_idx i, backward_t tag) const { return tangent_at(i.i, tag); }
  auto at(tangent_idx i, central_t tag) const { return tangent_at(i.i, tag); }
  auto operator[](tangent_idx i) const { return tangent_at(i.i); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using tangent_iterator =
      const_line_vertex_iterator<this_t, Real, N, tangent_idx, vec<Real, N>,
                                 vec<Real, N>>;
  using const_tangent_container =
      const_line_vertex_container<this_t, Real, N, tangent_idx, vec<Real, N>,
                                  vec<Real, N>>;
  using tangent_container = line_vertex_container<this_t, Real, N, tangent_idx,
                                                  vec<Real, N>, vec<Real, N>>;
  auto tangents(bool prefer_calc = false) const {
    return const_tangent_container{*this, prefer_calc};
  }
  auto tangents(bool prefer_calc = false) {
    return tangent_container{*this, prefer_calc};
  }
  //----------------------------------------------------------------------------
  auto& tangents_property() {
    if (!m_tangents) {
      m_tangents = &add_vertex_property<vec<Real, N>>("tangents");
    }
    return *m_tangents;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& tangents_property() const {
    if (!m_tangents) {
      throw std::runtime_error{"no tangent property created"};
    }
    return *m_tangents;
  }
  //----------------------------------------------------------------------------
  auto& tangents_to_property(bool update = false) {
    if (m_tangents != nullptr && !update) { return tangents_property(); }
    auto& prop = tangents_property();
    boost::copy(this->tangents(true), prop.begin());
    return prop;
  }
  //============================================================================
  auto arc_length() const {
    Real len = 0;
    for (size_t i = 0; i < this->num_vertices() - 1; ++i) {
      len += distance(vertex_at(i), vertex_at(i + 1));
    }
    return len;
  }
  //----------------------------------------------------------------------------
  bool is_closed() const { return m_is_closed; }
  void set_closed(bool is_closed) { m_is_closed = is_closed; }
  //----------------------------------------------------------------------------
  template <typename T>
  auto& vertex_property(const std::string& name) {
    auto prop = m_vertex_properties.at(name).get();
    assert(typeid(T) == prop->type());
    return *dynamic_cast<vertex_property_t<T>*>(prop);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  const auto& vertex_property(const std::string& name) const {
    auto prop = m_vertex_properties.at(name).get();
    assert(typeid(T) == prop->type_info());
    return *dynamic_cast<const vertex_property_t<T>*>(prop);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto& add_vertex_property(const std::string& name, const T& value = T{}) {
    auto [it, suc] = m_vertex_properties.insert(
        std::pair{name, std::make_unique<vertex_property_t<T>>(value)});
    auto prop = dynamic_cast<vertex_property_t<T>*>(it->second.get());
    prop->resize(m_vertices.size());
    return *prop;
  }
  //----------------------------------------------------------------------------
  bool has_vertex_property(const std::string& name) const {
    return m_vertex_properties.find(name) != end(m_vertex_properties);
  }
  //----------------------------------------------------------------------------
  void write(const std::string& file);
  //----------------------------------------------------------------------------
  static void write(const std::vector<line<Real, N>>& line_set,
                    const std::string&                file);
  //----------------------------------------------------------------------------
  void write_vtk(const std::string& path,
                 const std::string& title = "tatooine line") const {
    vtk::legacy_file_writer writer(path, vtk::POLYDATA);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();

      // write points
      std::vector<std::array<Real, 3>> ps;
      ps.reserve(this->num_vertices());
      for (const auto& p : vertices()) {
        if constexpr (N == 3) {
          ps.push_back({p(0), p(1), p(2)});
        } else {
          ps.push_back({p(0), p(1), 0});
        }
      }
      writer.write_points(ps);

      // write lines
      std::vector<std::vector<size_t>> line_seq(
          1, std::vector<size_t>(this->num_vertices()));
      boost::iota(line_seq.front(), 0);
      if (this->is_closed()) { line_seq.front().push_back(0); }
      writer.write_lines(line_seq);

      writer.write_point_data(this->num_vertices());

      // write properties
      for (auto& [name, prop] : m_vertex_properties) {
        const auto& type = prop->type();
        if (type == typeid(float)) {
          write_prop_to_vtk<float>(writer, name, prop);
        } else if (type == typeid(vec<float, 2>)) {
          write_prop_to_vtk<vec<float, 2>>(writer, name, prop);
        } else if (type == typeid(vec<float, 3>)) {
          write_prop_to_vtk<vec<float, 3>>(writer, name, prop);
        } else if (type == typeid(vec<float, 4>)) {
          write_prop_to_vtk<vec<float, 4>>(writer, name, prop);

        } else if (type == typeid(double)) {
          write_prop_to_vtk<double>(writer, name, prop);
        } else if (type == typeid(vec<double, 2>)) {
          write_prop_to_vtk<vec<double, 2>>(writer, name, prop);
        } else if (type == typeid(vec<double, 3>)) {
          write_prop_to_vtk<vec<double, 3>>(writer, name, prop);
        } else if (type == typeid(vec<double, 4>)) {
          write_prop_to_vtk<vec<double, 4>>(writer, name, prop);
        }
      }
      writer.close();
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  static void write_prop_to_vtk(
      vtk::legacy_file_writer& writer, const std::string& name,
      const std::unique_ptr<deque_property<vertex_idx>>& prop) {
    const auto& deque =
        dynamic_cast<vertex_property_t<T>*>(prop.get())->container();

    writer.write_scalars(name, std::vector<T>(begin(deque), end(deque)));
  }
  //----------------------------------------------------------------------------
  template <size_t N_ = N, std::enable_if_t<N_ == 3>...>
  static auto read_vtk(const std::string& filepath) {
    struct reader : vtk::legacy_file_listener {
      std::vector<std::array<Real, 3>> points;
      std::vector<int>                 lines;

      void on_points(const std::vector<std::array<Real, 3>>& points_) override {
        points = points_;
      }
      void on_lines(const std::vector<int>& lines_) override {
        lines = lines_;
      }
    } listener;

    vtk::legacy_file file{filepath};
    file.add_listener(listener);
    file.read();

    std::vector<line<Real, 3>> lines;
    const auto&                vs = listener.points;
    for (size_t i = 0; i < listener.lines.size();) {
      auto&       l    = lines.emplace_back();
      const auto& size = listener.lines[i++];
      for (; i < size; ++i) { l.push_back({vs[i][0], vs[i][1], vs[i][2]}); }
    }
    return lines;
  }
};
//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto diff(const line<Real, N>& l) {
  return l.tangents();
}
//==============================================================================
// template <typename Real, size_t N>
// template <typename Pred>
// std::vector<line<Real, N>> line<Real, N>::filter(Pred&& pred) const {
//  std::vector<line<Real, N>> filtered_lines;
//  bool                         need_new_strip = true;
//
//  size_t i      = 0;
//  bool   closed = is_closed();
//  for (const auto [x, t] : *this) {
//    if (pred(x, t, i)) {
//      if (need_new_strip) {
//        filtered_lines.emplace_back();
//        need_new_strip = false;
//      }
//      filtered_lines.back().push_back(x, t);
//    } else {
//      closed         = false;
//      need_new_strip = true;
//      if (!filtered_lines.empty() && filtered_lines.back().num_vertices() <=
//      1)
//        filtered_lines.pop_back();
//    }
//    i++;
//  }
//
//  if (!filtered_lines.empty() && filtered_lines.back().num_vertices() <= 1)
//    filtered_lines.pop_back();
//  if (filtered_lines.num_vertices() == 1)
//  filtered_lines.front().set_is_closed(closed); return filtered_lines;
//}

namespace detail {
template <typename LineCont>
void write_line_container_to_vtk(const LineCont& lines, const std::string& path,
                                 const std::string& title) {
  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
  if (writer.is_open()) {
    size_t num_pts = 0;
    for (const auto& l : lines) num_pts += l.num_vertices();
    std::vector<std::array<typename LineCont::value_type::real_t, 3>> points;
    std::vector<std::vector<size_t>>                                  line_seqs;
    points.reserve(num_pts);
    line_seqs.reserve(lines.size());

    size_t cur_first = 0;
    for (const auto& l : lines) {
      // add points
      for (const auto& p : l.vertices()) {
        if constexpr (LineCont::value_type::num_dimensions() == 3) {
          points.push_back({p(0), p(1), p(2)});
        } else {
          points.push_back({p(0), p(1), 0});
        }
      }

      // add lines
      boost::iota(line_seqs.emplace_back(l.num_vertices()), cur_first);
      if (l.is_closed()) { line_seqs.back().push_back(cur_first); }
      cur_first += l.num_vertices();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_lines(line_seqs);
    writer.write_point_data(num_pts);
    writer.close();
  }
}
//------------------------------------------------------------------------------
template <typename Lines, typename MaxDist /*, typename MinAngle*/>
auto merge_line_container(Lines   lines,
                          MaxDist max_dist /*, MinAngle min_angle*/) {
  using line_t = typename std::decay_t<Lines>::value_type;
  std::list<line_t> merged;
  merged.emplace_back(std::move(lines.back()));
  lines.pop_back();

  while (!lines.empty()) {
    auto min_d   = std::numeric_limits<typename line_t::real_t>::max();
    auto best_it = std::end(lines);
    bool merged_take_front = false;
    bool it_take_front     = false;
    for (auto it = std::begin(lines); it != std::end(lines); ++it) {
      if (const auto d =
              distance(merged.back().front_vertex(), it->front_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = true;
        it_take_front     = true;
      }
      if (const auto d =
              distance(merged.back().back_vertex(), it->front_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = false;
        it_take_front     = true;
      }
      if (const auto d =
              distance(merged.back().front_vertex(), it->back_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = true;
        it_take_front     = false;
      }
      if (const auto d =
              distance(merged.back().back_vertex(), it->back_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = false;
        it_take_front     = false;
      }
    }

    if (best_it != end(lines)) {
      if (merged_take_front) {
        if (it_take_front) {
          for (const auto& v : best_it->vertices()) {
            merged.back().push_front(v);
          }
        } else {
          for (const auto& v :
               best_it->vertices() | boost::adaptors::reversed) {
            merged.back().push_front(v);
          }
        }
      } else {
        if (it_take_front) {
          for (const auto& v : best_it->vertices()) {
            merged.back().push_back(v);
          }
        } else {
          for (const auto& v :
               best_it->vertices() | boost::adaptors::reversed) {
            merged.back().push_back(v);
          }
        }
      }
      lines.erase(best_it);
    } else {
      merged.emplace_back(std::move(lines.back()));
      lines.pop_back();
    }
  }

  return merged;
}

//------------------------------------------------------------------------------
template <typename Lines, typename Real>
auto filter_length(Lines lines, Real length) {
  for (auto it = begin(lines); it != end(lines);) {
    auto l = it->arc_length();
    ++it;
    if (l < length) { lines.erase(prev(it)); }
  }
  return lines;
}
}  // namespace detail
//------------------------------------------------------------------------------
template <typename Real, size_t N>
void write_vtk(const std::vector<line<Real, N>>& lines, const std::string& path,
               const std::string& title = "tatooine lines") {
  detail::write_line_container_to_vtk(lines, path, title);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
void write_vtk(const std::list<line<Real, N>>& lines, const std::string& path,
               const std::string& title = "tatooine lines") {
  detail::write_line_container_to_vtk(lines, path, title);
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename MaxDist>
auto merge(const std::vector<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::merge_line_container(lines, max_dist);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename MaxDist>
auto merge(const std::list<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::merge_line_container(lines, max_dist);
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename MaxDist>
auto filter_length(const std::vector<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::filter_length(lines, max_dist);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename MaxDist>
auto filter_length(const std::list<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::filter_length(lines, max_dist);
}

//==============================================================================
template <typename Real, size_t N,
          template <typename> typename InterpolationKernel>
struct parameterized_line : line<Real, N> {
  using this_t   = parameterized_line<Real, N, InterpolationKernel>;
  using parent_t = line<Real, N>;
  using typename parent_t::empty_exception;
  using typename parent_t::pos_t;
  using typename parent_t::tangent_idx;
  using typename parent_t::vec_t;
  using typename parent_t::vertex_idx;
  struct time_not_found : std::exception {};
  using interpolation_t = InterpolationKernel<vec_t>;
  static constexpr bool interpolation_needs_first_derivative =
      interpolation_t::needs_first_derivative;

  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  using parent_t::num_vertices;
  using parent_t::tangent_at;
  using parent_t::vertex_at;
  using parent_t::front_vertex;
  using parent_t::back_vertex;
  using parent_t::vertices;
  using parent_t::operator[];

 private:
  vertex_property_t<Real>*    m_parameterization;
  std::deque<interpolation_t> m_interpolators;

 public:
  parameterized_line()
      : m_parameterization{&this->template add_vertex_property<Real>(
            "parameterization")} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  parameterized_line(const parameterized_line& other)
      : parent_t{other},
        m_parameterization{
            &this->template vertex_property<Real>("parameterization")},
        m_interpolators{other.m_interpolators} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  parameterized_line(parameterized_line&& other)
      : parent_t{std::move(other)},
        m_parameterization{
            &this->template vertex_property<Real>("parameterization")},
        m_interpolators{std::move(other.m_interpolators)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  parameterized_line(std::initializer_list<std::pair<pos_t, Real>>&& data)
      : m_parameterization{
            &this->template add_vertex_property<Real>("parameterization")} {
    for (auto& [pos, param] : data) {
      push_back(std::move(pos), param);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& operator=(const parameterized_line& other) {
    parent_t::operator=(other);
    m_parameterization =
        &this->template vertex_property<Real>("parameterization");
    m_interpolators = other.m_interpolators;
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& operator=(parameterized_line&& other) {
    parent_t::operator=(std::move(other));
    m_parameterization =
        &this->template vertex_property<Real>("parameterization");
    m_interpolators = std::move(other.m_interpolators);
    return *this;
  }
  //----------------------------------------------------------------------------
  const auto& parameterization() const { return *m_parameterization; }
  auto&       parameterization() { return *m_parameterization; }
  //----------------------------------------------------------------------------
  std::pair<pos_t&, Real&> front() {
    return {front_vertex(), front_parameterization()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<const pos_t&, const Real&> front() const {
    return {front_vertex(), front_parameterization()};
  }
  //----------------------------------------------------------------------------
  std::pair<pos_t&, Real&> back() {
    return {back_vertex(), back_parameterization()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<const pos_t&, const Real&> back() const {
    return {back_vertex(), back_parameterization()};
  }

  //----------------------------------------------------------------------------
  std::pair<const pos_t&, const Real&> at(vertex_idx i) const {
    return {vertex_at(i.i), parameterization_at(i.i)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<pos_t&, Real&> at(vertex_idx i) {
    return {vertex_at(i.i), parameterization_at(i.i)};
  }
  //----------------------------------------------------------------------------
  auto operator[](vertex_idx i) const { return at(i); }
  auto operator[](vertex_idx i) { return at(i); }
  //----------------------------------------------------------------------------
  auto& parameterization_at(size_t i) { return m_parameterization->at(i); }
  const auto& parameterization_at(size_t i) const {
    return m_parameterization->at(i);
  }
  //----------------------------------------------------------------------------
  auto&       front_parameterization() { return m_parameterization->front(); }
  const auto& front_parameterization() const {
    return m_parameterization->front();
  }
  //----------------------------------------------------------------------------
  auto&       back_parameterization() { return m_parameterization->back(); }
  const auto& back_parameterization() const {
    return m_parameterization->back();
  }
  //----------------------------------------------------------------------------
  auto push_back(const pos_t& p, Real t, bool auto_compute_interpolator = true) {
    auto i                    = parent_t::push_back(p);
    m_parameterization->at(i) = t;
    if (num_vertices() > 1) {
      if (auto_compute_interpolator) {
        if constexpr (interpolation_needs_first_derivative) {
          const auto h =
              back_parameterization() - parameterization_at(num_vertices() - 2);
          m_interpolators.emplace_back(
              vertex_at(num_vertices() - 2), back_vertex(),
              tangent_at(num_vertices() - 2) * h, back_tangent() * h);
          if (num_vertices() >= 3) {
            update_interpolator(num_vertices() - 3);
          }
        } else {
          m_interpolators.emplace_back(vertex_at(num_vertices() - 2),
                                       vertex_at(num_vertices() - 1));
          if (num_vertices() >= 3) {
            update_interpolator(num_vertices() - 3);
          }
        }
      } else {
        m_interpolators.emplace_back();
      }
    }
    return i;
  }
  //----------------------------------------------------------------------------
  auto push_back(pos_t&& p, Real t, bool auto_compute_interpolator = true) {
    auto i                    = parent_t::push_back(std::move(p));
    m_parameterization->at(i) = t;
    if (num_vertices() > 1) {
      if (auto_compute_interpolator) {
        if constexpr (interpolation_needs_first_derivative) {
          const auto h =
              back_parameterization() - parameterization_at(num_vertices() - 2);
          m_interpolators.emplace_back(
              vertex_at(num_vertices() - 2), back_vertex(),
              tangent_at(num_vertices() - 2) * h, back_tangent() * h);
          if (num_vertices() >= 3) {
            update_interpolator(num_vertices() - 3);
          }
        } else {
          m_interpolators.emplace_back(vertex_at(num_vertices() - 2),
                                       back_vertex());
          if (num_vertices() >= 3) {
            update_interpolator(num_vertices() - 3);
          }
        }
      } else {
        m_interpolators.emplace_back();
      }
    }
    return i;
  }
  //----------------------------------------------------------------------------
  void pop_back() {
    parent_t::pop_back();
    m_parameterization->pop_back();
    if (num_vertices() >= 2) { m_interpolators.pop_back(); }
  }
  //----------------------------------------------------------------------------
  auto push_front(const pos_t& p, Real t, bool auto_compute_interpolator = true) {
    auto i                    = parent_t::push_front(p);
    m_parameterization->at(i) = t;
    if (num_vertices() > 1) {
      if (auto_compute_interpolator) {
        if constexpr (interpolation_needs_first_derivative) {
          const auto h = front_parameterization() - parameterization_at(1);
          m_interpolators.emplace_front(front_vertex(), vertex_at(1),
                                        front_tangent() * h, tangent_at(1) * h);
          if (num_vertices() >= 3) { update_interpolator(1); }
        } else {
          m_interpolators.emplace_front(front_vertex(), vertex_at(1));
          if (num_vertices() >= 3) { update_interpolator(1); }
        }
      } else {
        m_interpolators.emplace_front();
      }
    }
    return i;
  }
  //----------------------------------------------------------------------------
  auto push_front(pos_t&& p, Real t, bool auto_compute_interpolator = true) {
    auto i                    = parent_t::push_front(std::move(p));
    m_parameterization->at(i) = t;
    if (num_vertices() > 1) {
      if (auto_compute_interpolator) {
        if constexpr (interpolation_needs_first_derivative) {
          const auto h = front_parameterization() - parameterization_at(1);
          m_interpolators.emplace_front(front_vertex(), vertex_at(1),
                                        front_tangent() * h, tangent_at(1) * h);
          if (num_vertices() >= 3) { update_interpolator(1); }
        } else {
          m_interpolators.emplace_front(front_vertex(), vertex_at(1));
          if (num_vertices() >= 3) { update_interpolator(1); }
        }
      } else {
        m_interpolators.emplace_front();
      }
    }
    return i;
  }
  //----------------------------------------------------------------------------
  void pop_front() {
    parent_t::pop_front();
    m_parameterization->pop_front();
    if (num_vertices() >= 2) { m_interpolators.pop_front(); }
  }
  //----------------------------------------------------------------------------
  void update_interpolators() {
    for (size_t i = 0; i < num_vertices() - 1; ++i) {
      update_interpolator(i);
    }
  }
  //----------------------------------------------------------------------------
  void update_interpolator(size_t i) {
    if constexpr (interpolation_needs_first_derivative) {
      auto h = parameterization_at(i + 1) - parameterization_at(i);
      m_interpolators[i] =
          interpolation_t{vertex_at(i), vertex_at(i + 1), tangent_at(i) * h,
                          tangent_at(i + 1) * h};
    } else {
      m_interpolators[i] =
          interpolation_t{vertex_at(i), vertex_at(i + 1)};
    }
  }
  //----------------------------------------------------------------------------
  auto binary_search_index(Real t) const {
    if (t < front_parameterization() && front_parameterization() - t < 1e-7) {
      t = front_parameterization();
    } else if (t > back_parameterization() &&
               t - back_parameterization() < 1e-7) {
      t = back_parameterization();
    }
    if (this->empty()) { throw empty_exception{}; }

    if (t < front_parameterization() || t > back_parameterization()) {
      throw time_not_found{};
    }

    // find the two points t is in between
    size_t left = 0, right = num_vertices() - 1;

    while (right - left > 1) {
      size_t center = (left + right) / 2;
      if (t < parameterization_at(center)) {
        right = center;
      } else {
        left = center;
      }
    }
    return left;
  }
  //----------------------------------------------------------------------------
  /// sample the line via interpolation
  auto sample(Real t) const {
    const auto left = binary_search_index(t);

    // interpolate
    const Real factor =
        std::min<Real>(1, std::max<Real>(0, (t - parameterization_at(left)) /
                                                (parameterization_at(left + 1) -
                                                 parameterization_at(left))));
    return m_interpolators[left](factor);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(const Real t) const {
    return sample(t);
  }
  //----------------------------------------------------------------------------
  /// sample tangents
  auto tangent(Real t) const {
    const auto left = binary_search_index(t);

    // interpolate
    const Real factor =
        (t - parameterization_at(left)) /
        (parameterization_at(left + 1) - parameterization_at(left));
    assert(0 <= factor && factor <= 1);
    return m_interpolators[left].curve().tangent(factor);
  }
  //----------------------------------------------------------------------------
  /// sample second_derivatives
  auto second_derivative(Real t) const {
    const auto left = binary_search_index(t);

    // interpolate
    const Real factor =
        (t - parameterization_at(left)) /
        (parameterization_at(left + 1) - parameterization_at(left));
    assert(0 <= factor && factor <= 1);
    return m_interpolators[left].curve().second_derivative(factor);
  }
  //----------------------------------------------------------------------------
  /// sample curvature
  Real curvature(Real t) const {
    if (num_vertices() <= 1) { return -1; }
    const auto left = binary_search_index(t);

    // interpolate
    const Real factor =
        std::min<Real>(1, std::max<Real>(0, (t - parameterization_at(left)) /
                                                (parameterization_at(left + 1) -
                                                 parameterization_at(left))));
    return m_interpolators[left].curve().curvature(factor);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// sample a line property via interpolation
  template <typename Prop>
  auto sample(Real t, const vertex_property_t<Prop>& prop) const {
    auto left = binary_search_index(t);

    // interpolate
    const Real h      = parameterization_at(left+1) - parameterization_at(left);
    const Real factor = (t - parameterization_at(left)) / h;
    assert(0 <= factor && factor <= 1);

    // if constexpr (interpolation_needs_first_derivative) {
    //  InterpolationKernel<Prop>{prop[left], prop[left+1], tangent_at(left) * h,
    //                            tangent_at(left+1) * h}(factor);
    //} else {
    //  return InterpolationKernel<Prop>{prop[left], prop[left+1]}(factor);
    //}
    return prop[left] * (1 - factor) + prop[left+1] * factor;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  template <typename Prop>
  void resample_property(
      const linspace<Real>& ts, this_t& resampled, const std::string& name,
      const std::unique_ptr<deque_property<vertex_idx>>& p) const {
    resample_property(ts, resampled, name,
                      *dynamic_cast<const vertex_property_t<Prop>*>(p.get()));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Prop>
  void resample_property(const linspace<Real>& ts, this_t& resampled,
                         const std::string&             name,
                         const vertex_property_t<Prop>& prop) const {
    auto&  resampled_prop = resampled.template add_vertex_property<Prop>(name);
    size_t i              = 0;
    for (auto t : ts) {
      if constexpr (std::is_same_v<vec_t, Prop>) {
        if (&prop == this->m_tangents) {
          resampled_prop[i++] = tangent(t);
          continue;
        }
      }
      resampled_prop[i++] = sample(t, prop);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Ts>
  void resample_properties(const linspace<Real>& ts, this_t& resampled) const {
    for (const auto& prop : this->m_vertex_properties) {
      (
          [&]() {
            if (prop.second->type() == typeid(Ts)) {
              resample_property<Ts>(ts, resampled, prop.first, prop.second);
            }
          }(),
          ...);
    }
    }

  public:
   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   auto resample(const linspace<Real>& ts) const {
     this_t resampled;
     for (auto t : ts) { resampled.push_back(sample(t), t); }
     resample_properties<double, float,
                         vec<double, 2>, vec<float, 2>,
                         vec<double, 3>, vec<float, 3>,
                         vec<double, 4>, vec<float, 4>>(ts, resampled);
     return resampled;
   }

  //============================================================================
  void uniform_parameterization(Real t0 = 0) {
    parameterization_at(0) = t0;
    for (size_t i = 1; i < this->num_vertices(); ++i) {
      parameterization_at(i) = parameterization_at(i - 1) + 1;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void chordal_parameterization(Real t0 = 0) {
    parameterization_at(0) = t0;
    for (size_t i = 1; i < this->num_vertices(); ++i) {
      parameterization_at(i) =
          parameterization_at(i - 1) + distance(vertex_at(i), vertex_at(i - 1));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void centripetal_parameterization(Real t0 = 0) {
    parameterization_at(0) = t0;
    for (size_t i = 1; i < this->num_vertices(); ++i) {
      parameterization_at(i) =
          parameterization_at(i - 1) +
          std::sqrt(distance(vertex_at(i), vertex_at(i - 1)));
    }
  }
  //----------------------------------------------------------------------------
  auto arc_length() const {
    Real len = 0;
    for (size_t i = 0; i < this->num_vertices() - 1; ++i) {
      len += m_interpolators[i].curve().arc_length(linspace<Real>{0, 1, 10});
    }
    return len;
  }

  //============================================================================
  // tangents
  //============================================================================
  /// computes tangent assuming the line is a quadratic curve
  auto tangent_at(const tangent_idx i, quadratic_t tag) const {
    return tangent_at(i.i, tag);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// computes tangent assuming the line is a quadratic curve
  auto tangent_at(const size_t i, quadratic_t /*tag*/) const {
    const auto& x0 = [&]() {
      if (i == 0) {
        return vertex_at(i);
      } else if (i == num_vertices() - 1) {
        return vertex_at(i - 2);
      }
      return vertex_at(i - 1);
    }();
    const auto& x1 = [&]() {
      if (i == 0) {
        return vertex_at(i + 1);
      } else if (i == num_vertices() - 1) {
        return vertex_at(i - 1);
      }
      return vertex_at(i);
    }();
    const auto& x2 = [&]() {
      if (i == 0) {
        return vertex_at(i + 2);
      } else if (i == num_vertices() - 1) {
        return vertex_at(i);
      }
      return vertex_at(i + 1);
    }();
    const auto& t0 = [&]() -> const auto& {
      if (i == 0) {
        return parameterization_at(i);
      } else if (i == num_vertices() - 1) {
        return parameterization_at(i - 2);
      }
      return parameterization_at(i - 1);
    }
    ();
    const auto& t1 = [&]() -> const auto& {
      if (i == 0) {
        return parameterization_at(i + 1);
      } else if (i == num_vertices() - 1) {
        return parameterization_at(i - 1);
      }
      return parameterization_at(i);
    }
    ();
    const auto& t2 = [&]() -> const auto& {
      if (i == 0) {
        return parameterization_at(i + 2);
      } else if (i == num_vertices() - 1) {
        return parameterization_at(i);
      }
      return parameterization_at(i + 1);
    }
    ();

    // for each dimension fit a quadratic curve through the neighbor points and
    // the point itself and compute the derivative
    vec_t      tangent;
    const mat3 A{{t0 * t0, t0, Real(1)},
                 {t1 * t1, t1, Real(1)},
                 {t2 * t2, t2, Real(1)}};
    for (size_t n = 0; n < N; ++n) {
      vec3 b{x0(n), x1(n), x2(n)};
      auto c     = gesv(A, b);
      tangent(n) = 2 * c(0) * parameterization_at(i) + c(1);
    }
    return tangent;
  }
  //----------------------------------------------------------------------------
  auto tangent_at(const tangent_idx i, automatic_t tag,
                  bool prefer_calc = false) const {
    return tangent_at(i.i, tag, prefer_calc);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto tangent_at(const size_t i, automatic_t /*tag*/,
                  bool         prefer_calc = false) const {
    if (this->m_tangents && !prefer_calc) { return this->m_tangents->at(i); }
    if (num_vertices() >= 3) {
      return tangent_at(i, quadratic);
    } else /* if (num_vertices() == 2)*/ {
      return this->tangent_at(i, forward);
    }
  }
  //----------------------------------------------------------------------------
  auto tangent_at(const tangent_idx i, bool prefer_calc = false) const {
    return tangent_at(i.i, automatic, prefer_calc);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto tangent_at(const size_t i, bool prefer_calc = false) const {
    return tangent_at(i, automatic, prefer_calc);
  }
  //----------------------------------------------------------------------------
  auto front_tangent(bool prefer_calc = false) const {
    return tangent_at(0, prefer_calc);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto back_tangent(bool prefer_calc = false) const {
    return tangent_at(num_vertices() - 1, prefer_calc);
  }
  //----------------------------------------------------------------------------
  auto at(tangent_idx i, bool prefer_calc = false) const {
    return tangent_at(i.i, prefer_calc);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](tangent_idx i) const { return tangent_at(i.i); }
  //----------------------------------------------------------------------------
  auto& tangents_to_property(bool update = false) {
    if (this->m_tangents != nullptr && !update) {
      return this->tangents_property();
    }
    auto& prop = this->tangents_property();
    boost::copy(this->tangents(true), prop.begin());
    return prop;
  }
  //----------------------------------------------------------------------------
  using tangent_iterator =
      const_line_vertex_iterator<this_t, Real, N, tangent_idx, vec<Real, N>,
                                 vec<Real, N>>;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto integrate_property(const vertex_property_t<Real>& prop) {
    std::vector<Real> seg_lens(num_vertices() - 1);
    for (size_t i = 0; i < num_vertices() - 1; ++i) {
      seg_lens[i] = distance(vertex_at(i), vertex_at(i + 1));
    }
    Real integral = 0;
    integral += seg_lens.front() * prop.front(); 
    integral += seg_lens.back() * prop.back(); 
    for (size_t i = 1; i < num_vertices() - 1; ++i) {
      integral += (seg_lens[i - 1] + seg_lens[i]) * prop[i];
    }
    integral /= boost::accumulate(seg_lens, Real(0)) * 2;
    return integral;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto integrate_curvature() const {
    std::vector<Real> seg_lens(num_vertices() - 1);
    for (size_t i = 0; i < num_vertices() - 1; ++i) {
      seg_lens[i] =
          m_interpolators[i].curve().arc_length(linspace<Real>{0, 1, 10});
    }
    Real integral = 0;
    integral += seg_lens.front() * curvature(front_parameterization()); 
    integral += seg_lens.back() * curvature(back_parameterization()); 
    for (size_t i = 1; i < num_vertices() - 1; ++i) {
      integral += (seg_lens[i - 1] + seg_lens[i]) * curvature(parameterization_at(i));
    }
    integral /= boost::accumulate(seg_lens, Real(0)) * 2;
    return integral;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto integrate_curvature(const linspace<Real>& ts) const {
    std::vector<Real> seg_lens(ts.size()-1);
    for (size_t i = 0; i < ts.size() - 1; ++i) {
      seg_lens[i] = distance(at(ts[i]), vertex_at(ts[i + 1]));
    }
    Real integral = 0;
    integral += seg_lens.front() * curvature(ts.front()); 
    integral += seg_lens.back() * curvature(ts.back()); 
    for (size_t i = 1; i < ts.size() - 1; ++i) {
      integral += (seg_lens[i - 1] + seg_lens[i]) * curvature(ts[i]);
    }
    integral /= boost::accumulate(seg_lens, Real(0)) * 2;
    return integral;
  }
};
//------------------------------------------------------------------------------
template <typename Real, size_t N, template <typename> typename InterpolationKernel>
auto diff(const parameterized_line<Real, N, InterpolationKernel>& l) {
  return l.tangents();
}
//==============================================================================
/// \brief      merge line strips
template <typename Real, size_t N>
void merge_lines(std::vector<line<Real, N>>& lines0,
                 std::vector<line<Real, N>>& lines1) {
  const Real eps = 1e-7;
  // move line1 pairs to line0 pairs
  const size_t size_before = size(lines0);
  lines0.resize(size(lines0) + size(lines1));
  std::move(begin(lines1), end(lines1), next(begin(lines0), size_before));
  lines1.clear();

  // merge line0 side
  for (auto line0 = begin(lines0); line0 != end(lines0); ++line0) {
    for (auto line1 = begin(lines0); line1 != end(lines0); ++line1) {
      if (line0 != line1 && !line0->empty() && !line1->empty()) {
        // [line0front, ..., LINE0BACK] -> [LINE1FRONT, ..., line1back]
        if (approx_equal(line0->back(), line1->front(), eps)) {
          for (size_t i = 1; i < line1->num_vertices(); ++i) {
            line0->push_back(line1->at(i));
          }
          *line1 = std::move(*line0);
          line0->clear();

          // [line1front, ..., LINE1BACK] -> [LINE0FRONT, ..., line0back]
        } else if (approx_equal(line1->back(), line0->front(), eps)) {
          for (size_t i = 1; i < line0->num_vertices(); ++i) {
            line1->push_back(line0->at(i));
          }
          line0->clear();

          // [LINE1FRONT, ..., line1back] -> [LINE0FRONT, ..., line0back]
        } else if (approx_equal(line1->front(), line0->front(), eps)) {
          boost::reverse(line1->vertices());
          // -> [line1back, ..., LINE1FRONT] -> [LINE0FRONT, ..., line0back]
          for (size_t i = 1; i < line0->num_vertices(); ++i) {
            line1->push_back(line0->at(i));
          }
          line0->clear();

          // [line0front, ..., LINE0BACK] -> [line1front,..., LINE1BACK]
        } else if (approx_equal(line0->back(), line1->back(), eps)) {
          boost::reverse(line0->vertices());
          // -> [line1front, ..., LINE1BACK] -> [LINE0BACK, ..., line0front]
          for (size_t i = 1; i < line0->num_vertices(); ++i) {
            line1->push_back(line0->at(i));
          }
          line0->clear();
        }
      }
    }
  }

  // move empty vectors of line0 side at end
  for (unsigned int i = 0; i < lines0.size(); i++) {
    for (unsigned int j = 0; j < i; j++) {
      if (lines0[j].empty() && !lines0[i].empty()) {
        lines0[j] = std::move(lines0[i]);
      }
    }
  }

  // remove empty vectors of line0 side
  for (int i = lines0.size() - 1; i >= 0; i--) {
    if (lines0[i].empty()) { lines0.pop_back(); }
  }
}

//----------------------------------------------------------------------------
template <typename Real, size_t N>
auto line_segments_to_line_strips(
    const std::vector<line<Real, N>>& line_segments) {
  std::vector<std::vector<line<Real, N>>> merged_strips(line_segments.size());

  auto seg_it = begin(line_segments);
  for (auto& merged_strip : merged_strips) {
    merged_strip.push_back({*seg_it});
    ++seg_it;
  }

  auto num_merge_steps =
      static_cast<size_t>(std::ceil(std::log2(line_segments.size())));

  for (size_t i = 0; i < num_merge_steps; i++) {
    size_t offset = std::pow(2, i);

    for (size_t j = 0; j < line_segments.size(); j += offset * 2) {
      auto left  = j;
      auto right = j + offset;
      if (right < line_segments.size()) {
        merge_lines(merged_strips[left], merged_strips[right]);
      }
    }
  }
  return merged_strips.front();
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto merge_lines(const std::vector<line<Real, N>>& lines) {
  std::vector<line<Real, N>> merged_lines;
  if (!lines.empty()) {
    auto line_strips = line_segments_to_line_strips(lines);

    for (const auto& line_strip : line_strips) {
      merged_lines.emplace_back();
      for (size_t i = 0; i < line_strip.num_vertices() - 1; i++) {
        merged_lines.back().push_back(line_strip[i]);
      }
      if (&line_strip.front() == &line_strip.back()) {
        merged_lines.back().set_closed(true);
      } else {
        merged_lines.back().push_back(line_strip.back());
      }
    }
  }
  return merged_lines;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
