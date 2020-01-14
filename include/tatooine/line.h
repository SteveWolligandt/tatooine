#ifndef TATOOINE_LINE_H
#define TATOOINE_LINE_H

#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/algorithm/reverse.hpp>
#include <boost/range/algorithm_ext/iota.hpp>
#include <cassert>
#include <deque>
#include <map>
#include <stdexcept>

#include "handle.h"
#include "interpolation.h"
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
static constexpr inline automatic_t automatic;

struct forward_t {};
static constexpr inline forward_t forward;

struct backward_t {};
static constexpr inline backward_t backward;

struct central_t {};
static constexpr inline central_t central;

struct quadratic_t {};
static constexpr inline quadratic_t quadratic;

//==============================================================================
// Iterators
//==============================================================================
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
struct const_line_vertex_iterator
    : boost::iterator_facade<
          const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>, Value,
          boost::bidirectional_traversal_tag, Reference> {
  const_line_vertex_iterator(Handle handle, const Line& l)
      : m_handle{handle}, m_line{l} {}
  const_line_vertex_iterator(const const_line_vertex_iterator& other)
      : m_handle{other.m_handle}, m_line{other.m_line} {}
  using this_t = const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;

 private:
  Handle      m_handle;
  const Line& m_line;

  friend class boost::iterator_core_access;

  void increment() { ++m_handle; }
  void decrement() { --m_handle; }

  auto equal(const const_line_vertex_iterator& other) const {
    return m_handle == other.m_handle;
  }
  const Reference dereference() const { return m_line[m_handle]; }

 public:
  this_t next(size_t inc = 1) const {
    this_t n = *this;
    n.m_handle.i += inc;
    return n;
  }
  this_t prev(size_t dec = 1) const {
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
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
static auto next(
    const const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
    size_t inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto prev(
    const const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
    size_t dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto& advance(const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
              size_t inc = 1) {
  return it.advance(inc);
}
//==============================================================================
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference = Value&>
struct line_vertex_iterator
    : boost::iterator_facade<
          line_vertex_iterator<Line, Real, N, Handle, Value, Reference>, Value,
          boost::bidirectional_traversal_tag, Reference> {
  line_vertex_iterator(Handle handle, Line& l)
      : m_handle{handle}, m_line{l} {}
  line_vertex_iterator(const line_vertex_iterator& other)
      : m_handle{other.m_handle}, m_line{other.m_line} {}
  using this_t = line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;

 private:
  Handle m_handle;
  Line&  m_line;

  friend class boost::iterator_core_access;

  void increment() { ++m_handle; }
  void decrement() { --m_handle; }

  auto equal(const line_vertex_iterator& other) const {
    return m_handle == other.m_handle;
  }
  Reference dereference() { return m_line[m_handle]; }

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
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto next(const line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
          size_t inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto prev(const line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
          size_t dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto& advance(line_vertex_iterator<Line, Real, N, Handle, Value, Reference>& it,
              size_t inc = 1) {
  return it.advance(inc);
}
//==============================================================================
// Container
//==============================================================================
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference = Value&>
struct const_line_vertex_container {
  using iterator = line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;
  using const_iterator =
      const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;
  //--------------------------------------------------------------------------
  const Line& m_line;
  //--------------------------------------------------------------------------
  auto begin() const { return const_iterator{Handle{0}, m_line}; }
  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto end() const {
    return const_iterator{Handle{m_line.num_vertices()}, m_line};
  }
  //--------------------------------------------------------------------------
  const auto& front() const { return m_line.at(Handle{0}); }
  //--------------------------------------------------------------------------
  const auto& back() const {
    return m_line.at(Handle{m_line.num_vertices() - 1});
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto begin(
    const const_line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto end(
    const const_line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
//============================================================================
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference = Value&>
struct line_vertex_container {
  using iterator = line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;
  using const_iterator =
      const_line_vertex_iterator<Line, Real, N, Handle, Value, Reference>;
  //----------------------------------------------------------------------------
  Line& m_line;
  //----------------------------------------------------------------------------
  auto begin() const { return const_iterator{Handle{0}, m_line}; }
  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto begin() { return iterator{Handle{0}, m_line}; }
  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto end() const {
    return const_iterator{Handle{m_line.num_vertices()}, m_line};
  }
  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto end() { return iterator{Handle{m_line.num_vertices()}, m_line}; }
  //----------------------------------------------------------------------------
  const auto& front() const { return m_line.at(Handle{0}); }
  auto&       front() { return m_line.at(Handle{0}); }
  //----------------------------------------------------------------------------
  const auto& back() const {
    return m_line.at(Handle{m_line.num_vertices() - 1});
  }
  auto& back() { return m_line.at(Handle{m_line.num_vertices() - 1}); }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto begin(const line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto begin(line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto end(const line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto end(line_vertex_container<Line, Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
//============================================================================
// Line implementation
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
  //----------------------------------------------------------------------------
  struct second_derivative_idx : handle {
    using handle::handle;
    bool operator==(second_derivative_idx other) const {
      return this->i == other.i;
    }
    bool operator!=(second_derivative_idx other) const {
      return this->i != other.i;
    }
    bool operator<(second_derivative_idx other) const { return this->i < other.i; }
    static constexpr auto invalid() {
      return second_derivative_idx{handle::invalid_idx};
    }
  };
  //----------------------------------------------------------------------------
  struct curvature_idx : handle {
    using handle::handle;
    bool operator==(curvature_idx other) const { return this->i == other.i; }
    bool operator!=(curvature_idx other) const { return this->i != other.i; }
    bool operator<(curvature_idx other) const { return this->i < other.i; }
    static constexpr auto invalid() { return curvature_idx{handle::invalid_idx}; }
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
 private:
  pos_container_t             m_vertices;
  bool                        m_is_closed = false;
  vertex_property_container_t m_vertex_properties;

  //============================================================================
 public:
  line() = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line(const line& other)
      : m_vertices{other.m_vertices}, m_is_closed{other.m_is_closed} {
    for (auto& [name, prop] : other.m_vertex_properties) {
      m_vertex_properties[name] = prop->clone();
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line(line&& other) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line& operator     =(const line& other) {
    m_vertices  = other.m_vertices;
    m_is_closed = other.m_is_closed;
    m_vertex_properties.clear();
    for (auto& [name, prop] : other.m_vertex_properties) {
      m_vertex_properties[name] = prop->clone();
    }
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
  const auto& vertex_at(vertex_idx v) const { return m_vertices[v.i]; }
  auto&       vertex_at(vertex_idx v) { return m_vertices[v.i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& at(vertex_idx v) const { return m_vertices[v.i]; }
  auto&       at(vertex_idx v) { return m_vertices[v.i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& operator[](vertex_idx v) const { return m_vertices[v.i]; }
  auto&       operator[](vertex_idx v) { return m_vertices[v.i]; }
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
  auto tangent_at(const tangent_idx t, forward_t tag) const {
    return tangent_at(t.i, tag);
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
  auto tangent_at(const tangent_idx t, backward_t tag) const {
    return tangent_at(t.i, tag);
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
  auto tangent_at(const tangent_idx t, central_t tag) const {
    return tangent_at(t.i, tag);
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
  auto tangent_at(const tangent_idx t, automatic_t /*tag*/) const {
    return tangent_at(t.i);
  }
  //----------------------------------------------------------------------------
  auto tangent_at(const size_t i, automatic_t /*tag*/) const {
    if (is_closed()) { return tangent_at(i, central); }
    if (i == 0) { return tangent_at(i, forward); }
    if (i == num_vertices() - 1) { return tangent_at(i, backward); }
    return tangent_at(i, central);
  }
  //----------------------------------------------------------------------------
  auto tangent_at(const tangent_idx t) const { return tangent_at(t, automatic); }
  auto tangent_at(const size_t i) const { return tangent_at(i, automatic); }
  //----------------------------------------------------------------------------
  auto front_tangent() const { return tangent_at(0); }
  auto back_tangent() const { return tangent_at(num_vertices() - 1); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(tangent_idx t) const { return tangent_at(t.i); }
  auto at(tangent_idx t, forward_t fw) const { return tangent_at(t.i, fw); }
  auto at(tangent_idx t, backward_t bw) const { return tangent_at(t.i, bw); }
  auto at(tangent_idx t, central_t ce) const { return tangent_at(t.i, ce); }
  auto operator[](tangent_idx t) const { return tangent_at(t.i); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using tangent_iterator =
      const_line_vertex_iterator<this_t, Real, N, tangent_idx, vec<Real, N>,
                                 vec<Real, N>>;
  using tangent_container =
      const_line_vertex_container<this_t, Real, N, tangent_idx, vec<Real, N>,
                                  vec<Real, N>>;
  auto tangents() const { return tangent_container{*this}; }
  //----------------------------------------------------------------------------
  auto& tangents_to_property() {
    auto& tangent_prop = [&]() -> auto& {
      if (!has_vertex_property("tangents")) {
        return this->template add_vertex_property<vec<Real, N>>("tangents");
      } else {
        return this->template vertex_property<vec<Real, N>>("tangents");
      }
    }();
    boost::copy(this->tangents(), tangent_prop.begin());
    return tangent_prop;
  }
  //============================================================================
  // second derivative
  //============================================================================
  /// calculates second derivative at point i with forward differences
  auto second_derivative_at(const second_derivative_idx d2,
                            forward_t                   fw) const {
    return second_derivative_at(d2.i, fw);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// calculates second derivative at point i with forward differences
  auto second_derivative_at(const size_t i, forward_t /*fw*/) const {
    assert(num_vertices() > 1);
    if (is_closed()) {
      if (i == num_vertices() - 1) {
        return (front_tangent() - back_tangent()) /
               distance(front_vertex(), back_vertex());
      }
    }
    return (tangent_at(i + 1) - tangent_at(i)) /
           distance(vertex_at(i), vertex_at(i + 1));
  }
  //----------------------------------------------------------------------------
  /// calculates second derivative at point i with backward differences
  auto second_derivative_at(const second_derivative_idx d2, backward_t bw) const {
    return second_derivative_at(d2.i, bw);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// calculates second derivative at point i with backward differences
  auto second_derivative_at(const size_t i, backward_t /*bw*/) const {
    assert(num_vertices() > 1);
    if (is_closed()) {
      if (i == 0) {
        return (front_tangent() - back_tangent()) /
               distance(back_vertex(), front_tangent());
      }
    }
    return (tangent_at(i) - tangent_at(i - 1)) /
           distance(vertex_at(i), vertex_at(i - 1));
  }
  //----------------------------------------------------------------------------
  /// calculates second derivative at point d2 with central differences
  auto second_derivative_at(const second_derivative_idx d2, central_t ce) const {
    return second_derivative_at(d2.i, ce);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// calculates second derivative at point i with central differences
  auto second_derivative_at(const size_t i, central_t /*ce*/) const {
    if (is_closed()) {
      if (i == 0) {
        return (tangent_at(1) - back_tangent()) /
               (distance(back_vertex(), front_vertex()) +
                distance(front_vertex(), vertex_at(1)));
      } else if (i == num_vertices() - 1) {
        return (front_tangent() - tangent_at(i - 1)) /
               (distance(vertex_at(i - 1), back_vertex()) +
                distance(back_vertex(), front_vertex()));
      }
    }
    return (tangent_at(i + 1) - tangent_at(i - 1)) /
           (distance(vertex_at(i - 1), vertex_at(i)) +
            distance(vertex_at(i), vertex_at(i + 1)));
  }
  //----------------------------------------------------------------------------
  auto second_derivative_at(const second_derivative_idx d2) const {
    return second_derivative_at(d2.i);
  }
  auto second_derivative_at(const size_t i) const {
    if (is_closed()) { return second_derivative_at(i, central); }
    if (i == 0) { return second_derivative_at(i, forward); }
    if (i == num_vertices() - 1) { return second_derivative_at(i, backward); }
    return second_derivative_at(i, central);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(second_derivative_idx d2) const { return second_derivative_at(d2.i); }
  auto at(second_derivative_idx d2, forward_t fw) const {
    return second_derivative_at(d2.i, fw);
  }
  auto at(second_derivative_idx d2, backward_t bw) const {
    return second_derivative_at(d2.i, bw);
  }
  auto at(second_derivative_idx d2, central_t ce) const {
    return second_derivative_at(d2.i, ce);
  }
  auto operator[](second_derivative_idx d2) const {
    return second_derivative_at(d2.i);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using second_derivative_iterator =
      const_line_vertex_iterator<this_t, Real, N, second_derivative_idx, vec<Real, N>,
                           vec<Real, N>>;
  using second_derivative_container =
      const_line_vertex_container<this_t, Real, N, second_derivative_idx, vec<Real, N>,
                            vec<Real, N>>;
  auto second_derivatives() const { return second_derivative_container{*this}; }
  //----------------------------------------------------------------------------
  auto& second_derivative_to_property() {
    auto& second_derivative_prop =
        add_vertex_property<vec<Real, N>>("second_derivative");
    boost::copy(second_derivatives(), second_derivative_prop.begin());
    return second_derivative_prop;
  }
  //============================================================================
  // curvature
  //============================================================================
  auto curvature_at(const curvature_idx c, forward_t fw) const {
    return curvature_at(c.i, fw);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto curvature_at(size_t i, forward_t fw) const {
    auto d1  = tangent_at(i, fw);
    auto d2  = second_derivative_at(i, fw);
    auto ld1 = ::tatooine::length(d1);
    return std::abs(d1(0) * d2(1) - d1(1) * d2(0)) / (ld1 * ld1 * ld1);
  }
  //----------------------------------------------------------------------------
  auto curvature_at(const curvature_idx c, backward_t bw) const {
    return curvature_at(c.i, bw);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto curvature_at(size_t i, backward_t bw) const {
    auto d1  = tangent_at(i, bw);
    auto d2  = second_derivative_at(i, bw);
    auto ld1 = ::tatooine::length(d1);
    return std::abs(d1(0) * d2(1) - d1(1) * d2(0)) / (ld1 * ld1 * ld1);
  }
  //----------------------------------------------------------------------------
  auto curvature_at(const curvature_idx c, central_t ce) const {
    return curvature_at(c.i, ce);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto curvature_at(size_t i, central_t ce) const {
    auto d1  = tangent_at(i, ce);
    auto d2 = second_derivative_at(i, ce);
    auto ld1 = ::tatooine::length(d1);
    return std::abs(d1(0) * d2(1) - d1(1) * d2(0)) / (ld1 * ld1 * ld1);
  }
  //----------------------------------------------------------------------------
  auto curvature_at(const curvature_idx c) const {
    return curvature_at(c.i, central);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto curvature_at(size_t i) const {
    if (is_closed()) { return curvature_at(i, central); }
    if (i == 0) { return curvature_at(i, forward); }
    if (i == num_vertices() - 1) { return curvature_at(i, backward); }
    return curvature_at(i, central);
  }
  //----------------------------------------------------------------------------
  auto front_curvature() const { return curvature_at(0); }
  auto back_curvature() const { return curvature_at(num_vertices() - 1); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(curvature_idx c, forward_t fw) const { return curvature_at(c.i, fw); }
  auto at(curvature_idx c, backward_t bw) const { return curvature_at(c.i, bw); }
  auto at(curvature_idx c, central_t ce) const { return curvature_at(c.i, ce); }
  auto at(curvature_idx c) const { return curvature_at(c.i); }
  auto operator[](curvature_idx c) const { return curvature_at(c.i); }
  //----------------------------------------------------------------------------
  using curvature_iterator =
      const_line_vertex_iterator<this_t, Real, N, curvature_idx, Real, Real>;
  using curvature_container =
      const_line_vertex_container<this_t, Real, N, curvature_idx, Real, Real>;
  auto curvatures() const { return curvature_container{*this}; }
  //----------------------------------------------------------------------------
  auto& curvatures_to_property() {
    auto& curvature_prop = add_vertex_property<Real>("curvatures");
    boost::copy(curvatures(), curvature_prop.begin());
    return curvature_prop;
  }
  //============================================================================
  auto length() {
    Real len = 0;
    for (size_t i = 0; i < this->num_vertices() - 1; ++i) {
      len += norm(vertex_at(i) - vertex_at(i + 1));
    }
    return len;
  }
  //----------------------------------------------------------------------------
  bool is_closed() const { return m_is_closed; }
  void set_closed(bool is_closed) { m_is_closed = is_closed; }
  ////----------------------------------------------------------------------------
  ///// filters the line and returns a vector of lines
  // template <typename Pred>
  // std::vector<line<Real, N>> filter(Pred&& pred) const;
  //
  ////----------------------------------------------------------------------------
  ///// filters out all points where the eigenvalue of the jacobian is not real
  // template <typename vf_t>
  // auto filter_only_real_eig_vals(const vf_t& vf) const {
  //  jacobian j{vf};
  //
  //  return filter([&](auto x, auto t, auto) {
  //    auto [eigvecs, eigvals] = eig(j(x, t));
  //    for (const auto& eigval : eigvals) {
  //      if (std::abs(std::imag(eigval)) > 1e-7) { return false; }
  //    }
  //    return true;
  //  });
  //}
  //----------------------------------------------------------------------------
  template <typename T>
  auto& vertex_property(const std::string& name) {
    auto prop = m_vertex_properties.at(name).get();
    assert(typeid(T) == prop->type_info());
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
      std::vector<std::vector<int>>    lines;

      void on_points(const std::vector<std::array<Real, 3>>& points_) override {
        points = points_;
      }
      void on_lines(const std::vector<std::vector<int>>& lines_) override {
        lines = lines_;
      }
    } listener;

    vtk::legacy_file file{filepath};
    file.add_listener(listener);
    file.read();

    std::list<line<Real, 3>> lines;
    const auto&              vs = listener.points;
    for (const auto& line : listener.lines) {
      auto& pv_line = lines.emplace_back();
      for (auto i : line) { pv_line.push_back({vs[i][0], vs[i][1], vs[i][2]}); }
    }
    return lines;
  }
};

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
    auto l = it->length();
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
template <typename Real, size_t N>
struct parameterized_line : line<Real, N> {
  using this_t   = parameterized_line<Real, N>;
  using parent_t = line<Real, N>;
  using typename parent_t::empty_exception;
  using typename parent_t::pos_t;
  using typename parent_t::vec_t;
  using typename parent_t::vertex_idx;
  using typename parent_t::tangent_idx;
  using typename parent_t::second_derivative_idx;
  using typename parent_t::curvature_idx;
  struct time_not_found : std::exception {};

  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  using parent_t::num_vertices;
  using parent_t::vertices;
  using parent_t::vertex_at;
  using parent_t::tangent_at;
  using parent_t::second_derivative_at;
  using parent_t::curvature_at;
  using parent_t::operator[];

 private:
  vertex_property_t<Real>* m_parameterization;

 public:
  parameterized_line()
      : m_parameterization{&this->template add_vertex_property<Real>("parameterization")} {}
  parameterized_line(const parameterized_line& other)
      : parent_t{other},
        m_parameterization{&this->template vertex_property<Real>("parameterization")} {}
  parameterized_line(parameterized_line&& other)
      : parent_t{std::move(other)},
        m_parameterization{&this->template vertex_property<Real>("parameterization")} {}
  auto& operator=(const parameterized_line& other) {
    parent_t::operator=(other);
    m_parameterization = &this->template vertex_property<Real>("parameterization");
    return *this;
  }
  auto& operator=(parameterized_line&& other) {
    parent_t::operator=(std::move(other));
    m_parameterization = &this->template vertex_property<Real>("parameterization");
    return *this;
  }
  //----------------------------------------------------------------------------
  parameterized_line(std::initializer_list<std::pair<pos_t, Real>>&& data)
      : m_parameterization{&this->template add_vertex_property<Real>("parameterization")} {
    for (auto& [pos, param] : data) {
      push_back(std::move(pos), param);
    }
  }
  //----------------------------------------------------------------------------
  const auto& parameterization() const { return *m_parameterization; }
  auto&       parameterization() { return *m_parameterization; }
  //----------------------------------------------------------------------------
  std::pair<pos_t&, Real&> front() {
    return {this->front_vertex(), front_parameterization()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<const pos_t&, const Real&> front() const {
    return {this->front_vertex(), front_parameterization()};
  }
  //----------------------------------------------------------------------------
  std::pair<pos_t&, Real&> back() {
    return {this->back_vertex(), back_parameterization()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<const pos_t&, const Real&> back() const {
    return {this->back_vertex(), back_parameterization()};
  }

  //----------------------------------------------------------------------------
  std::pair<const pos_t&, const Real&> at(vertex_idx v) const {
    return {vertex_at(v), parameterization_at(v.i)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<pos_t&, Real&> at(vertex_idx v) {
    return {vertex_at(v), parameterization_at(v.i)};
  }
  //----------------------------------------------------------------------------
  auto operator[](vertex_idx v) const { return at(v); }
  auto operator[](vertex_idx v) { return at(v); }
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
  void push_back(const pos_t& p, Real t) {
    auto i = parent_t::push_back(p);
    m_parameterization->at(i) = t;
  }
  //----------------------------------------------------------------------------
  void push_back(pos_t&& p, Real t) {
    auto i = parent_t::push_back(std::move(p));
    m_parameterization->at(i) = t;
  }
  //----------------------------------------------------------------------------
  void pop_back() {
    parent_t::pop_back();
    m_parameterization->pop_back();
  }
  //----------------------------------------------------------------------------
  void push_front(const pos_t& p, Real t) {
    auto i = parent_t::push_front(p);
    m_parameterization->at(i) = t;
  }
  //----------------------------------------------------------------------------
  void push_front(pos_t&& p, Real t) {
    auto i = parent_t::push_front(std::move(p));
    m_parameterization->at(i) = t;
  }
  //----------------------------------------------------------------------------
  void pop_front() {
    parent_t::pop_front();
    m_parameterization->pop_front();
  }

  //----------------------------------------------------------------------------
  /// sample the line via interpolation
  template <template <typename> typename Interpolator = interpolation::hermite>
  auto sample(Real t) const {
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

    // interpolate
    Real factor = (t - parameterization_at(left)) /
                  (parameterization_at(right) - parameterization_at(left));
    Interpolator<Real> interp;
    return interp.interpolate_iter(next(begin(vertices()), left),
                                   next(begin(vertices()), right),
                                   begin(vertices()), end(vertices()), factor);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <template <typename> typename Interpolator = interpolation::hermite>
  auto operator()(const Real t) const {
    return sample<Interpolator>(t);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <template <typename> typename Interpolator = interpolation::hermite>
  auto resample(const linspace<Real>& ts) const {
    this_t resampled;
    for (auto t : ts) { resampled.push_back(sample<Interpolator>(t), t); }
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
  /// computes tangent assuming the line is a quadratic curve
  auto tangent_at(const tangent_idx t, quadratic_t tag) const {
    return tangent_at(t.i, tag);
  }
  //----------------------------------------------------------------------------
  /// computes tangent assuming the line is a quadratic curve
  auto tangent_at(const size_t i, quadratic_t /*tag*/) const {
    const auto& x0 = [&]() {
      if (i == 0) { return vertex_at(i); }
      else if (i == num_vertices() - 1) { return vertex_at(i - 2); }
      return vertex_at(i - 1);
    }();
    const auto& x1 = [&]() {
      if (i == 0) { return vertex_at(i + 1); }
      else if (i == num_vertices() - 1) { return vertex_at(i - 1); }
      return vertex_at(i);
    }();
    const auto& x2 = [&]() {
      if (i == 0) { return vertex_at(i + 2); }
      else if (i == num_vertices() - 1) { return vertex_at(i); }
      return vertex_at(i + 1);
    }();
    const auto& t0 = [&]() -> const auto& {
      if (i == 0) { return parameterization_at(i); }
      else if (i == num_vertices() - 1) { return parameterization_at(i - 2); }
      return parameterization_at(i - 1);
    }();
    const auto& t1 = [&]() -> const auto& {
      if (i == 0) { return parameterization_at(i + 1); }
      else if (i == num_vertices() - 1) { return parameterization_at(i - 1); }
      return parameterization_at(i);
    }();
    const auto& t2 = [&]() -> const auto& {
      if (i == 0) { return parameterization_at(i + 2); }
      else if (i == num_vertices() - 1) { return parameterization_at(i); }
      return parameterization_at(i + 1);
    }();

    // for each dimension fit a quadratic curve through the neighbor points and
    // the point itself and compute the derivative
    vec_t          tangent;
    const mat3     A{{t0 * t0, t0, Real(1)},
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
  template <typename V, typename VReal>
  auto tangent_at(const tangent_idx t, const field<V, VReal, N, N>& v) const {
    return tangent_at(t.i, v);
  }
  //----------------------------------------------------------------------------
  template <typename V, typename VReal>
  auto tangent_at(const size_t i, const field<V, VReal, N, N>& v) const {
    return v(vertex_at(i), parameterization_at(i));
  }
  //----------------------------------------------------------------------------
  auto tangent_at(const tangent_idx t, automatic_t tag) const {
    return tangent_at(t.i, tag);
  }
  //----------------------------------------------------------------------------
  auto tangent_at(const size_t i, automatic_t /*tag*/) const {
    return tangent_at(i, quadratic);
  }
  //----------------------------------------------------------------------------
  auto tangent_at(const tangent_idx t) const { return tangent_at(t, automatic); }
  auto tangent_at(const size_t i) const { return tangent_at(i, automatic); }
  //----------------------------------------------------------------------------
  auto front_tangent() const { return tangent_at(0); }
  auto back_tangent() const { return tangent_at(num_vertices() - 1); }
  //----------------------------------------------------------------------------
  template <typename V, typename VReal>
  auto front_tangent(const field<V, VReal, N, N>& v) const {
    return tangent_at(0, v);
  }
  template <typename V, typename VReal>
  auto back_tangent(const field<V, VReal, N, N>& v) const {
    return tangent_at(num_vertices() - 1, v);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(tangent_idx t) const { return tangent_at(t.i); }
  template <typename V, typename VReal>
  auto at(tangent_idx t, const field<V, VReal, N, N>& v) const {
    return tangent_at(t.i, v);
  }
  auto at(tangent_idx t, forward_t fw) const { return tangent_at(t.i, fw); }
  auto at(tangent_idx t, backward_t bw) const { return tangent_at(t.i, bw); }
  auto at(tangent_idx t, central_t ce) const { return tangent_at(t.i, ce); }
  auto operator[](tangent_idx t) const { return tangent_at(t.i); }
  //----------------------------------------------------------------------------
  auto& tangents_to_property() {
    auto& tangent_prop = [&]() -> auto& {
      if (!this->has_vertex_property("tangents")) {
        return this->template add_vertex_property<vec<Real, N>>("tangents");
      } else {
        return this->template vertex_property<vec<Real, N>>("tangents");
      }
    }();
    boost::copy(this->tangents(), tangent_prop.begin());
    return tangent_prop;
  }
  //----------------------------------------------------------------------------
  template <typename V, typename VReal>
  auto& tangents_to_property(const field<V, VReal, N, N>& vf) {
    auto& tangent_prop = [&]() -> auto& {
      if (!this->has_vertex_property("tangents")) {
        return this->template add_vertex_property<vec<Real, N>>("tangents");
      } else {
        return this->template vertex_property<vec<Real, N>>("tangents");
      }
    }();
    for (size_t i = 0; i < num_vertices(); ++i) {
      tangent_prop[i] = vf(vertex_at(i), parameterization_at(i));
    }
    return tangent_prop;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using tangent_iterator =
      const_line_vertex_iterator<this_t, Real, N, tangent_idx, vec<Real, N>,
                                 vec<Real, N>>;
  using tangent_container =
      const_line_vertex_container<this_t, Real, N, tangent_idx, vec<Real, N>,
                                  vec<Real, N>>;
  auto tangents() const { return tangent_container{*this}; }
  /// TODO implement
  template <typename V, typename VReal>
  auto tangents(const field<V, VReal, N, N>& v) const {
    throw std::runtime_error("not implemented");
    return false;
  }
  //============================================================================
  // second derivative
  //============================================================================
  template <typename V, typename VReal>
  auto second_derivative_at(second_derivative_idx        i, quadratic_t /*tag*/,
                            const field<V, VReal, N, N>& v) const {
    return second_derivative_at(i.i, v);
  }
  template <typename V, typename VReal>
  auto second_derivative_at(size_t i, quadratic_t /*tag*/, const field<V, VReal, N, N>& v) const {
    assert(this->num_vertices() > 2);
    const auto x0 = [&]() {
      if (i == 0) { return tangent_at(i, v); }
      else if (i == num_vertices() - 1) { return tangent_at(i - 2, v); }
      return tangent_at(i - 1, v);
    }();
    const auto x1 = [&]() {
      if (i == 0) { return tangent_at(i + 1, v); }
      else if (i == num_vertices() - 1) { return tangent_at(i - 1, v); }
      return tangent_at(i, v);
    }();
    const auto x2 = [&]() {
      if (i == 0) { return tangent_at(i + 2, v); }
      else if (i == num_vertices() - 1) { return tangent_at(i, v); }
      return tangent_at(i + 1, v);
    }();
    const auto& t0 = [&]() -> const auto& {
      if (i == 0) { return parameterization_at(i); }
      else if (i == num_vertices() - 1) { return parameterization_at(i - 2); }
      return parameterization_at(i - 1);
    }();
    const auto& t1 = [&]() -> const auto& {
      if (i == 0) { return parameterization_at(i + 1); }
      else if (i == num_vertices() - 1) { return parameterization_at(i - 1); }
      return parameterization_at(i);
    }();
    const auto& t2 = [&]() -> const auto& {
      if (i == 0) { return parameterization_at(i + 2); }
      else if (i == num_vertices() - 1) { return parameterization_at(i); }
      return parameterization_at(i + 1);
    }();
    const auto& t = parameterization_at(i);
    // for each dimension fit a quadratic curve through the neighbor points and
    // the point itself and compute the derivative
    vec_t      dx;
    const mat3 A{{t0 * t0, t0, Real(1)},
                 {t1 * t1, t1, Real(1)},
                 {t2 * t2, t2, Real(1)}};
    for (size_t n = 0; n < N; ++n) {
      vec3 b{x0(n), x1(n), x2(n)};
      auto c     = gesv(A, b);
      dx(n) = 2 * c(0) * t + c(1);
    }
    return dx;
  }
  template <typename V, typename VReal>
  auto at(second_derivative_idx i, quadratic_t tag,
          const field<V, VReal, N, N>& v) const {
    return second_derivative_at(i.i,tag, v);
  }
  //----------------------------------------------------------------------------
  template <typename V, typename VReal>
  auto& second_derivative_to_property(const field<V, VReal, N, N>& v) {
    auto& snd_der_prop = [&]() -> auto& {
      if (!this->has_vertex_property("second_derivative")) {
        return this->template add_vertex_property<vec<Real, N>>("second_derivative");
      } else {
        return this->template vertex_property<vec<Real, N>>("second_derivative");
      }
    }
    ();
    for (size_t i = 0; i < num_vertices(); ++i) {
      snd_der_prop[i] = second_derivative_at(i, quadratic, v);
    }
    return snd_der_prop;
  }
  //============================================================================
  // curvature
  //============================================================================
  template <typename V, typename VReal>
  auto curvature_at(size_t i, const field<V, VReal, N, N>& v) const {
    auto d1  = tangent_at(i, v);
    auto d2  = second_derivative_at(i, quadratic, v);
    auto ld1 = ::tatooine::length(d1);
    return std::abs(d1(0) * d2(1) - d1(1) * d2(0)) / (ld1 * ld1 * ld1);
  }
  //----------------------------------------------------------------------------
  template <typename V, typename VReal>
  auto& curvature_to_property(const field<V, VReal, N, N>& v) {
    auto& curv_prop = [&]() -> auto& {
      if (!this->has_vertex_property("curvature")) {
        return this->template add_vertex_property<Real>("curvature");
      } else {
        return this->template vertex_property<Real>("curvature");
      }
    }();
    for (size_t i = 0; i < num_vertices(); ++i) {
      curv_prop[i] = curvature_at(i, v);
    }
    return curv_prop;
  }
};
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
