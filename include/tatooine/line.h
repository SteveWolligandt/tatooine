#ifndef TATOOINE_LINE_H
#define TATOOINE_LINE_H

#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/algorithm/reverse.hpp>
#include <boost/range/algorithm_ext/iota.hpp>
#include <cassert>
#include <deque>
#include <stdexcept>

#include "handle.h"
#include "interpolation.h"
#include "linspace.h"
#include "tensor.h"
#include "vtk_legacy.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
struct line;

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
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
struct const_line_vertex_iterator
    : boost::iterator_facade<
          const_line_vertex_iterator<Real, N, Handle, Value, Reference>, Value,
          boost::bidirectional_traversal_tag, Reference> {
  const_line_vertex_iterator(Handle handle, const line<Real, N>& l)
      : m_handle{handle}, m_line{l} {}
  const_line_vertex_iterator(const const_line_vertex_iterator& other)
      : m_handle{other.m_handle}, m_line{other.m_line} {}
  using this_t = const_line_vertex_iterator<Real, N, Handle, Value, Reference>;

 private:
  Handle               m_handle;
  const line<Real, N>& m_line;

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
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
static auto next(
    const const_line_vertex_iterator<Real, N, Handle, Value, Reference>& it,
    size_t inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto prev(
    const const_line_vertex_iterator<Real, N, Handle, Value, Reference>& it,
    size_t dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto& advance(const_line_vertex_iterator<Real, N, Handle, Value, Reference>& it,
              size_t inc = 1) {
  return it.advance(inc);
}
//==============================================================================
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference = Value&>
struct line_vertex_iterator
    : boost::iterator_facade<
          line_vertex_iterator<Real, N, Handle, Value, Reference>, Value,
          boost::bidirectional_traversal_tag, Reference> {
  line_vertex_iterator(Handle handle, line<Real, N>& l)
      : m_handle{handle}, m_line{l} {}
  line_vertex_iterator(const line_vertex_iterator& other)
      : m_handle{other.m_handle}, m_line{other.m_line} {}
  using this_t = line_vertex_iterator<Real, N, Handle, Value, Reference>;

 private:
  Handle         m_handle;
  line<Real, N>& m_line;

  friend class boost::iterator_core_access;

  void increment() { ++m_handle; }
  void decrement() { --m_handle; }

  auto equal(const line_vertex_iterator& other) const {
    return m_handle == other.m_handle;
  }
  Reference dereference() const { return m_line[m_handle]; }

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
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto next(const line_vertex_iterator<Real, N, Handle, Value, Reference>& it,
          size_t inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto prev(const line_vertex_iterator<Real, N, Handle, Value, Reference>& it,
          size_t dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto& advance(line_vertex_iterator<Real, N, Handle, Value, Reference>& it,
              size_t inc = 1) {
  return it.advance(inc);
}
//==============================================================================
// Container
//==============================================================================
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference = Value&>
struct const_line_vertex_container {
  using iterator = line_vertex_iterator<Real, N, Handle, Value, Reference>;
  using const_iterator =
      const_line_vertex_iterator<Real, N, Handle, Value, Reference>;
  //--------------------------------------------------------------------------
  const line<Real, N> &  m_line;
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
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto begin(
    const const_line_vertex_container<Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto end(
    const const_line_vertex_container<Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
//============================================================================
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference = Value&>
struct line_vertex_container {
  using iterator = line_vertex_iterator<Real, N, Handle, Value, Reference>;
  using const_iterator =
      const_line_vertex_iterator<Real, N, Handle, Value, Reference>;
  //----------------------------------------------------------------------------
  line<Real, N>& m_line;
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
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto begin(const line_vertex_container<Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto begin(line_vertex_container<Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto end(const line_vertex_container<Real, N, Handle, Value, Reference>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle, typename Value,
          typename Reference>
auto end(line_vertex_container<Real, N, Handle, Value, Reference>& it) {
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
  struct vertex : handle {
    using handle::handle;
    bool operator==(vertex other) const { return this->i == other.i; }
    bool operator!=(vertex other) const { return this->i != other.i; }
    bool operator<(vertex other) const { return this->i < other.i; }
    static constexpr auto invalid() { return vertex{handle::invalid_idx}; }
  };
  //----------------------------------------------------------------------------
  struct tangent : handle {
    using handle::handle;
    bool operator==(tangent other) const { return this->i == other.i; }
    bool operator!=(tangent other) const { return this->i != other.i; }
    bool operator<(tangent other) const { return this->i < other.i; }
    static constexpr auto invalid() { return tangent{handle::invalid_idx}; }
  };
  //----------------------------------------------------------------------------
  struct diff2 : handle {
    using handle::handle;
    bool operator==(diff2 other) const { return this->i == other.i; }
    bool operator!=(diff2 other) const { return this->i != other.i; }
    bool operator<(diff2 other) const { return this->i < other.i; }
    static constexpr auto invalid() { return diff2{handle::invalid_idx}; }
  };
  //----------------------------------------------------------------------------
  struct curvature : handle {
    using handle::handle;
    bool operator==(curvature other) const { return this->i == other.i; }
    bool operator!=(curvature other) const { return this->i != other.i; }
    bool operator<(curvature other) const { return this->i < other.i; }
    static constexpr auto invalid() { return curvature{handle::invalid_idx}; }
  };

  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto num_dimensions() noexcept { return N; }

  //============================================================================
 private:
  pos_container_t m_vertices;
  bool            m_is_closed = false;

  //============================================================================
 public:
  line()                  = default;
  line(const line& other) = default;
  line(line&& other)      = default;
  line& operator=(const line& other) = default;
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
  const auto& vertex_at(vertex v) const { return m_vertices[v.i]; }
  auto&       vertex_at(vertex v) { return m_vertices[v.i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& at(vertex v) const { return m_vertices[v.i]; }
  auto&       at(vertex v) { return m_vertices[v.i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& operator[](vertex v) const { return m_vertices[v.i]; }
  auto&       operator[](vertex v) { return m_vertices[v.i]; }
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
    return vertex{m_vertices.size() - 1};
  }
  auto push_back(const pos_t& p) {
    m_vertices.push_back(p);
    return vertex{m_vertices.size() - 1};
  }
  auto push_back(pos_t&& p) {
    m_vertices.emplace_back(std::move(p));
    return vertex{m_vertices.size() - 1};
  }
  auto pop_back() { m_vertices.pop_back(); }
  //----------------------------------------------------------------------------
  template <typename... Components, enable_if_arithmetic<Components...> = true,
            std::enable_if_t<sizeof...(Components) == N, bool> = true>
  auto push_front(Components... comps) {
    m_vertices.push_front(pos_t{static_cast<Real>(comps)...});
    return vertex{0};
  }
  auto push_front(const pos_t& p) {
    m_vertices.push_front(p);
    return vertex{0};
  }
  auto push_front(pos_t&& p) {
    m_vertices.emplace_front(std::move(p));
    return vertex{0};
  }
  void pop_front() { m_vertices.pop_front(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using vertex_iterator = line_vertex_iterator<Real, N, vertex, pos_t, pos_t&>;
  using const_vertex_iterator =
      const_line_vertex_iterator<Real, N, vertex, pos_t, const pos_t&>;
  using vertex_container =
      line_vertex_container<Real, N, vertex, pos_t, pos_t&>;
  using const_vertex_container =
      const_line_vertex_container<Real, N, vertex, pos_t, const pos_t&>;
  auto vertices() const { return const_vertex_container{*this}; }
  auto vertices() { return vertex_container{*this}; }
  //============================================================================
  // tangent
  //============================================================================
  /// calculates tangent at point t with backward differences
  auto tangent_at(const tangent t, forward_t fw) const {
    return tangent_at(t.i, fw);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// calculates tangent at point i with forward differences
  auto tangent_at(const size_t i, forward_t /*fw*/) const {
    assert(num_vertices() > 1);
    if (is_closed()) {
      if (i == num_vertices() - 1) {
        return (front_vertex() - back_vertex()) /
               distance(front_vertex(), back_vertex());
      }
    }
    return (vertex_at(i + 1) - vertex_at(i)) /
           distance(vertex_at(i), vertex_at(i + 1));
  }

  //----------------------------------------------------------------------------
  /// calculates tangent at point t with backward differences
  auto tangent_at(const tangent t, backward_t bw) const {
    return tangent_at(t.i, bw);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// calculates tangent at point i with central differences
  auto tangent_at(const size_t i, backward_t /*bw*/) const {
    assert(num_vertices() > 1);
    if (is_closed()) {
      if (i == 0) {
        return (front_vertex() - back_vertex()) /
               distance(back_vertex(), front_vertex());
      }
    }
    return (vertex_at(i) - vertex_at(i - 1)) /
           distance(vertex_at(i), vertex_at(i - 1));
  }

  //----------------------------------------------------------------------------
  /// calculates tangent at point i with central differences
  auto tangent_at(const tangent t, central_t ce) const {
    return tangent_at(t.i, ce);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// calculates tangent at point i with central differences
  auto tangent_at(const size_t i, central_t /*c*/) const {
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
  auto tangent_at(const size_t i) const {
    if (is_closed()) { return tangent_at(i, central); }
    if (i == 0) { return tangent_at(i, forward); }
    if (i == num_vertices() - 1) { return tangent_at(i, backward); }
    return tangent_at(i, central);
  }
  //----------------------------------------------------------------------------
  auto front_tangent() const { return tangent_at(0); }
  auto back_tangent() const { return tangent_at(num_vertices() - 1); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto at(tangent t) const { return tangent_at(t); }
  const auto at(tangent t, forward_t fw) const { return tangent_at(t, fw); }
  const auto at(tangent t, backward_t bw) const { return tangent_at(t, bw); }
  const auto at(tangent t, central_t ce) const { return tangent_at(t, ce); }
  const auto operator[](tangent t) const { return tangent_at(t); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using tangent_iterator =
      line_vertex_iterator<Real, N, tangent, vec<Real, N>, vec<Real, N>>;
  using const_tangent_iterator =
      const_line_vertex_iterator<Real, N, tangent, vec<Real, N>, vec<Real, N>>;
  using tangent_container =
      line_vertex_container<Real, N, tangent, vec<Real, N>, vec<Real, N>>;
  using const_tangent_container =
      const_line_vertex_container<Real, N, tangent, vec<Real, N>, vec<Real, N>>;
  auto tangents() const { return const_tangent_container{*this}; }
  auto tangents() { return tangent_container{*this}; }
  //============================================================================
  // second derivative
  //============================================================================
  /// calculates second derivative at point i with forward differences
  auto second_derivative_at(const diff2 d2, forward_t fw) const {
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
  auto second_derivative_at(const diff2 d2, backward_t bw) const {
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
  auto second_derivative_at(const diff2 d2, central_t ce) const {
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
  auto second_derivative_at(const diff2 d2) const {
    return second_derivative_at(d2.i);
  }
  auto second_derivative_at(const size_t i) const {
    if (is_closed()) { return second_derivative_at(i, central); }
    if (i == 0) { return second_derivative_at(i, forward); }
    if (i == num_vertices() - 1) { return second_derivative_at(i, backward); }
    return second_derivative_at(i, central);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto at(diff2 d2) const { return diff_at(d2); }
  const auto at(diff2 d2, forward_t fw) const { return diff_at(d2, fw); }
  const auto at(diff2 d2, backward_t bw) const { return diff_at(d2, bw); }
  const auto at(diff2 d2, central_t ce) const { return diff_at(d2, ce); }
  const auto operator[](diff2 d2) const { return diff_at(d2); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using diff2_iterator =
      line_vertex_iterator<Real, N, diff2, vec<Real, N>, vec<Real, N>>;
  using const_diff2_iterator =
      const_line_vertex_iterator<Real, N, diff2, vec<Real, N>, vec<Real, N>>;
  using diff2_container =
      line_vertex_container<Real, N, diff2, vec<Real, N>, vec<Real, N>>;
  using const_diff2_container =
      const_line_vertex_container<Real, N, diff2, vec<Real, N>, vec<Real, N>>;
  auto second_derivatives() const { return const_diff2_container{*this}; }
  auto second_derivatives() { return diff2_container{*this}; }
  //============================================================================
  // curvature
  //============================================================================
  auto curvature_at(const curvature c, forward_t fw) const {
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
  auto curvature_at(const curvature c, backward_t bw) const {
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
  auto curvature_at(const curvature c, central_t ce) const {
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
  auto curvature_at(const curvature c) const {
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
  const auto at(curvature t) const { return curvature_at(t); }
  const auto at(curvature t, forward_t fw) const { return curvature_at(t, fw); }
  const auto at(curvature t, backward_t bw) const {
    return curvature_at(t, bw);
  }
  const auto at(curvature t, central_t ce) const { return curvature_at(t, ce); }
  const auto operator[](curvature t) const { return curvature_at(t); }
  //----------------------------------------------------------------------------
  using curvature_iterator =
      line_vertex_iterator<Real, N, curvature, Real, Real>;
  using const_curvature_iterator =
      const_line_vertex_iterator<Real, N, curvature, Real, Real>;
  using curvature_container =
      line_vertex_container<Real, N, curvature, Real, Real>;
  using const_curvature_container =
      const_line_vertex_container<Real, N, curvature, Real, Real>;
  auto curvatures() const { return const_curvature_container{*this}; }
  auto curvatures() { return curvature_container{*this}; }
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
  void write(const std::string& file);
  //----------------------------------------------------------------------------
  static void write(const std::vector<line<Real, N>>& line_set,
                    const std::string&                file);
  //----------------------------------------------------------------------------
  void write_vtk(const std::string& path,
                 const std::string& title          = "tatooine line",
                 bool               write_tangents = false) const;
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

//------------------------------------------------------------------------------
template <typename Real, size_t N>
void line<Real, N>::write_vtk(const std::string& path, const std::string& title,
                              bool write_tangents) const {
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

    // write tangents
    if (write_tangents) {
      std::vector<std::vector<Real>> tangents;
      tangents.reserve(this->num_vertices());
      for (size_t i = 0; i < this->num_vertices(); ++i) {
        const auto t = tangent_at(i);
        tangents.push_back({t(0), t(1), t(2)});
      }
      writer.write_scalars("tangents", tangents);
    }

    writer.close();
  }
}

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
  struct time_not_found : std::exception {};

  using parent_t::num_vertices;
  using parent_t::tangent_at;
  using parent_t::vertex_at;
  using parent_t::vertices;

 private:
  std::deque<Real> m_parameterization;

 public:
  parameterized_line()                          = default;
  parameterized_line(const parameterized_line&) = default;
  parameterized_line(parameterized_line&&)      = default;
  parameterized_line& operator=(const parameterized_line&) = default;
  parameterized_line& operator=(parameterized_line&&) = default;
  //----------------------------------------------------------------------------
  parameterized_line(std::initializer_list<std::pair<pos_t, Real>>&& data) {
    for (auto& [pos, param] : data) {
      push_back(std::move(pos), std::move(param));
    }
  }
  //----------------------------------------------------------------------------
  const auto& parameterization() const { return m_parameterization; }
  auto&       parameterization() { return m_parameterization; }
  //----------------------------------------------------------------------------
  std::pair<pos_t&, Real&> front() {
    return {vertices().front(), m_parameterization.front()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<const pos_t&, const Real&> front() const {
    return {vertices().front(), m_parameterization.front()};
  }

  //----------------------------------------------------------------------------
  std::pair<pos_t&, Real&> back() {
    return {vertices().back(), m_parameterization.back()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<const pos_t&, const Real&> back() const {
    return {vertices().back(), m_parameterization.back()};
  }

  //----------------------------------------------------------------------------
  std::pair<const pos_t&, const Real&> at(size_t i) const {
    return {vertex_at(i), parameterization_at(i)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<pos_t&, Real&> at(size_t i) {
    return {vertex_at(i), parameterization_at(i)};
  }
  //----------------------------------------------------------------------------
  std::pair<const pos_t&, const Real&> operator[](size_t i) const {
    return {vertex_at(i), parameterization_at(i)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<pos_t&, Real&> operator[](size_t i) {
    return {vertex_at(i), parameterization_at(i)};
  }

  //----------------------------------------------------------------------------
  auto&       parameterization_at(size_t i) { return m_parameterization.at(i); }
  const auto& parameterization_at(size_t i) const {
    return m_parameterization.at(i);
  }
  //----------------------------------------------------------------------------
  auto&       front_parameterization() { return m_parameterization.front(); }
  const auto& front_parameterization() const {
    return m_parameterization.front();
  }

  //----------------------------------------------------------------------------
  auto&       back_parameterization() { return m_parameterization.back(); }
  const auto& back_parameterization() const {
    return m_parameterization.back();
  }

  //----------------------------------------------------------------------------
  void push_back(const pos_t& p, Real t) {
    parent_t::push_back(p);
    m_parameterization.push_back(t);
  }
  //----------------------------------------------------------------------------
  void push_back(pos_t&& p, Real t) {
    parent_t::push_back(std::move(p));
    m_parameterization.push_back(t);
  }
  //----------------------------------------------------------------------------
  void pop_back() {
    parent_t::pop_back();
    m_parameterization.pop_back();
  }

  //----------------------------------------------------------------------------
  void push_front(const pos_t& p, Real t) {
    parent_t::push_front(p);
    m_parameterization.push_front(t);
  }
  //----------------------------------------------------------------------------
  void push_front(pos_t&& p, Real t) {
    parent_t::push_front(std::move(p));
    m_parameterization.push_front(t);
  }
  //----------------------------------------------------------------------------
  void pop_front() {
    parent_t::pop_front();
    m_parameterization.pop_front();
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
  auto tangent_at(const size_t i, quadratic_t /*q*/) const {
    assert(this->num_vertices() > 1);
    // start or end point
    if (!this->is_closed()) {
      if (i == 0) { return at(1) - at(0); }
      if (i == this->num_vertices() - 1) { return at(i) - at(i - 1); }
    }

    // point in between
    // const auto& x0 = at(std::abs((i - 1) % this->num_vertices()));
    const auto& x0 = at(i - 1);
    const auto& x1 = at(i);
    const auto& x2 = at(i + 1);
    // const auto& x2 = at((i + 1) % this->num_vertices());
    const auto t = (parameterization_at(i) - parameterization_at(i - 1)) /
                   (parameterization_at(i + 1) - parameterization_at(i - 1));

    // for each component fit a quadratic curve through the neighbor points and
    // the point itself and compute the derivative
    vec_t      tangent;
    const mat3 A{{0.0, 0.0, 1.0}, {t * t, t, 1.0}, {1.0, 1.0, 1.0}};
    for (size_t j = 0; j < N; ++j) {
      vec3 b{x0(j), x1(j), x2(j)};
      auto coeffs = gesv(A, b);

      tangent(j) = 2 * coeffs(0) * t + coeffs(1);
    }
    return tangent;
  }

  //------------------------------------------------------------------------------
  void write_vtk(const std::string& path,
                 const std::string& title = "tatooine parameterized line",
                 bool               write_tangents = false) const {
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
      writer.write_lines(line_seq);

      writer.write_point_data(this->num_vertices());

      // write tangents
      if (write_tangents) {
        std::vector<std::vector<Real>> tangents;
        tangents.reserve(this->num_vertices());
        for (size_t i = 0; i < this->num_vertices(); ++i) {
          const auto t = tangent_at(i);
          tangents.push_back({t(0), t(1), t(2)});
        }
        writer.write_scalars("tangents", tangents);
      }

      // write parameterization
      std::vector<std::vector<Real>> parameterization;
      parameterization.reserve(this->num_vertices());
      for (auto t : m_parameterization) { parameterization.push_back({t}); }
      writer.write_scalars("parameterization", parameterization);

      writer.close();
    }
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
