#ifndef TATOOINE_GRID_H
#define TATOOINE_GRID_H
//==============================================================================
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <cassert>
#include <set>

#include "algorithm.h"
#include "boundingbox.h"
#include "grid_edge.h"
#include "grid_vertex.h"
#include "grid_vertex_edges.h"
#include "grid_vertex_neighbors.h"
#include "index_ordering.h"
#include "linspace.h"
#include "random.h"
#include "subgrid.h"
#include "type_traits.h"
#include "utility.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
class grid {
 public:
  using this_t            = grid<Real, N>;
  using linspace_iterator = typename linspace<Real>::iterator;
  using vertex            = grid_vertex<Real, N>;
  using edge              = grid_edge<Real, N>;
  using vertex_iterator   = grid_vertex_iterator<Real, N>;
  using edge_iterator     = grid_edge_iterator<Real, N>;

  struct vertex_container;

  //============================================================================
 private:
  std::array<linspace<Real>, N> m_dimensions;

  //============================================================================
 public:
  constexpr grid()                      = default;
  constexpr grid(const grid& other)     = default;
  constexpr grid(grid&& other) noexcept = default;
  //----------------------------------------------------------------------------
  template <typename OtherReal, size_t... Is>
  constexpr grid(const grid<OtherReal, N>& other,
                 std::index_sequence<Is...> /*is*/)
      : m_dimensions{other.dimension(Is)...} {}
  template <typename OtherReal>
  explicit constexpr grid(const grid<OtherReal, N>& other)
      : grid(other, std::make_index_sequence<N>{}) {}

  //----------------------------------------------------------------------------
  template <typename... Reals>
  explicit constexpr grid(const linspace<Reals>&... linspaces)
      : m_dimensions{linspace<Real>{linspaces}...} {
    static_assert(sizeof...(Reals) == N,
                  "number of linspaces does not match number of dimensions");
  }

  //----------------------------------------------------------------------------
  template <typename OtherReal, size_t... Is>
  constexpr grid(const boundingbox<OtherReal, N>& bb,
                 const std::array<size_t, N>&     res,
                 std::index_sequence<Is...> /*is*/)
      : m_dimensions{
            linspace<Real>{Real(bb.min(Is)), Real(bb.max(Is)), res[Is]}...} {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  template <typename OtherReal>
  constexpr grid(const boundingbox<OtherReal, N>& bb,
                 const std::array<size_t, N>&     res)
      : grid{bb, res, std::make_index_sequence<N>{}} {}

  //----------------------------------------------------------------------------
  ~grid() = default;

  //----------------------------------------------------------------------------
  constexpr auto operator=(const grid& other) -> grid& = default;
  constexpr auto operator=(grid&& other) noexcept -> grid& = default;

  //----------------------------------------------------------------------------
  template <typename OtherReal>
  constexpr auto operator=(const grid<OtherReal, N>& other) -> grid& {
    for (size_t i = 0; i < N; ++i) { m_dimensions[i] = other.dimension(i); }
    return *this;
  }
  //----------------------------------------------------------------------------
  constexpr auto dimension(size_t i) -> auto& { return m_dimensions[i]; }
  constexpr auto dimension(size_t i) const -> const auto& {
    return m_dimensions[i];
  }
  //----------------------------------------------------------------------------
  constexpr auto dimensions() -> auto& { return m_dimensions; }
  constexpr auto dimensions() const -> const auto& { return m_dimensions; }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto min(std::index_sequence<Is...> /*is*/) const {
    return vec<Real, N> {m_dimensions[Is].front()...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto min() const {
    return min(std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto max(std::index_sequence<Is...> /*is*/) const {
    return vec<Real, N> {m_dimensions[Is].back()...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto max() const {
    return max(std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto resolution(std::index_sequence<Is...> /*is*/) const {
    return vec<size_t, N> {m_dimensions[Is].size()...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto resolution() const {
    return resolution(std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  constexpr auto spacing() const {
    return spacing(std::make_index_sequence<N>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  template <size_t... Is>
  constexpr auto spacing(std::index_sequence<Is...> /*is*/) const {
    static_assert(sizeof...(Is) == N);
    return tatooine::vec{m_dimensions[Is].spacing()...};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto boundingbox() const {
    return boundingbox(std::make_index_sequence<N>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  template <size_t... Is>
  constexpr auto boundingbox(std::index_sequence<Is...> /*is*/) const {
    static_assert(sizeof...(Is) == N);
    return tatooine::boundingbox<Real, N>{
        vec<Real, N>{m_dimensions[Is].front()...},
        vec<Real, N>{m_dimensions[Is].back()...}};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto size() const { return size(std::make_index_sequence<N>{}); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 protected:
  template <size_t... Is>
  constexpr auto size(std::index_sequence<Is...> /*is*/) const {
    static_assert(sizeof...(Is) == N);
    return vec<size_t, N>{m_dimensions[Is].size()...};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto size(size_t i) const { return dimension(i).size(); }
  //----------------------------------------------------------------------------
  constexpr auto front(size_t i) const { return dimension(i).front(); }
  //----------------------------------------------------------------------------
  constexpr auto back(size_t i) const { return dimension(i).back(); }
  //----------------------------------------------------------------------------
  constexpr auto spacing(size_t i) const { return dimension(i).spacing(); }
  //----------------------------------------------------------------------------
  template <size_t... Is, typename... Reals,
            enable_if_arithmetic<Reals...> = true>
  constexpr auto in_domain(std::index_sequence<Is...> /*is*/,
                           Reals... xs) const {
    static_assert(sizeof...(Reals) == N,
                  "number of components does not match number of dimensions");
    static_assert(sizeof...(Is) == N,
                  "number of indices does not match number of dimensions");
#if has_cxx17_support()
    return ((m_dimensions[Is].front() <= xs &&
             xs <= m_dimensions[Is].back()) && ...);
#else
    constexpr std::array<Real, N> pos{static_cast<Real>(xs)...};
    for (size_t i = 0; i < N; ++i) {
      if (pos[i] < m_dimensions[i].front()) { return false; }
      if (pos[i] > m_dimensions[i].back()) { return false; }
    }
    return true;
#endif
  }

  //----------------------------------------------------------------------------
  template <typename... Reals, enable_if_arithmetic<Reals...> = true>
  constexpr auto in_domain(Reals... xs) const {
    static_assert(sizeof...(Reals) == N,
                  "number of components does not match number of dimensions");
    return in_domain(std::make_index_sequence<N>{}, std::forward<Reals>(xs)...);
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto in_domain(const std::array<Real, N>& x,
                           std::index_sequence<Is...> /*is*/) const {
    return in_domain(x[Is]...);
  }

  //----------------------------------------------------------------------------
  constexpr auto in_domain(const std::array<Real, N>& x) const {
    return in_domain(x, std::make_index_sequence<N>{});
  }

  //----------------------------------------------------------------------------
  constexpr auto num_points() const {
    return num_points(std::make_index_sequence<N>{});
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto num_points(std::index_sequence<Is...> /*is*/) const {
    static_assert(sizeof...(Is) == N);
#if has_cxx17_support()
    return (m_dimensions[Is].size() * ...);
#else
    Real f = 1;
    for (size_t i = 0; i < N; ++i) { f *= m_dimensions[i].size(); }
    return f;
#endif
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  auto front_vertex(std::index_sequence<Is...> /*is*/) {
    return vertex{m_dimensions[Is].begin()...};
  }
  //----------------------------------------------------------------------------
  auto front_vertex() { return front_vertex(std::make_index_sequence<N>()); }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  auto back_vertex(std::index_sequence<Is...> /*is*/) {
    return vertex{(--m_dimensions[Is].end())...};
  }
  //----------------------------------------------------------------------------
  auto back_vertex() { return back_vertex(std::make_index_sequence<N>()); }
  //----------------------------------------------------------------------------
  template <typename... Is, size_t... DIs, enable_if_integral<Is...> = true>
  auto at(std::index_sequence<DIs...>, Is... is) const -> vec<Real, N> {
    static_assert(sizeof...(DIs) == sizeof...(Is));
    static_assert(sizeof...(Is) == N);
    return {(m_dimensions[DIs][is])...};
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto at(Is... is) const {
    static_assert(sizeof...(Is) == N);
    return at(std::make_index_sequence<N>{}, is...);
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto operator()(Is... is) const {
    static_assert(sizeof...(Is) == N);
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto vertex_at(Is... is) const {
    static_assert(sizeof...(Is) == N);
    return vertex{*this, is...};
  }
  //----------------------------------------------------------------------------
  template <typename Is, enable_if_integral<Is> = true>
  auto vertex_at(const std::array<Is, N>& is) const {
    return invoke_unpacked([&](auto... is) { return vertex_at(is...); },
                           unpack(is));
  }
  //----------------------------------------------------------------------------
  constexpr auto num_vertices() const {
    size_t num = 1;
    for (const auto& dim : m_dimensions) { num *= dim.size(); }
    return num;
  }
  //----------------------------------------------------------------------------
  /// \return number of dimensions for one dimension dim
  constexpr auto edges() const {
    return grid_edge_container{this};
  }

  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto vertex_begin(std::index_sequence<Is...> /*is*/) const {
    return typename vertex::iterator{vertex(m_dimensions[Is].begin()...)};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto vertex_begin() const {
    return vertex_begin(std::make_index_sequence<N>{});
  }
 private:
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto vertex_end(std::index_sequence<Is...> /*is*/) const {
    return typename vertex::iterator{
        vertex(m_dimensions[Is].begin()..., m_dimensions.back().end())};
  }
 public:
  //----------------------------------------------------------------------------
  constexpr auto vertex_end() const {
    return vertex_end(std::make_index_sequence<N - 1>());
  }
  //----------------------------------------------------------------------------
  auto vertices() const { return vertex_container{this}; }
  //----------------------------------------------------------------------------
  auto vertices(const vertex& v) const {
    return grid_vertex_neighbors<Real, N>(v);
  }
  //----------------------------------------------------------------------------
  auto edges(const vertex& v) const { return grid_vertex_edges<Real, N>(v); }
  //----------------------------------------------------------------------------
  auto sub(const vertex& begin_vertex, const vertex& end_vertex) const {
    return subgrid<Real, N>(this, begin_vertex, end_vertex);
  }
  //----------------------------------------------------------------------------
  /// checks if an edge e has vertex v as point
  auto contains(const vertex& v, const edge& e) {
    return v == e.first || v == e.second;
  }
  //----------------------------------------------------------------------------
  /// checks if v0 and v1 are direct or diagonal neighbors
  auto are_neighbors(const vertex& v0, const vertex& v1) {
    auto v0_it = begin(v0.iterators);
    auto v1_it = begin(v1.iterators);
    for (; v0_it != end(v0.iterators); ++v0_it, ++v1_it) {
      if (distance(*v0_it, *v1_it) > 1) { return false; }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  /// checks if v0 and v1 are direct neighbors
  auto are_direct_neighbors(const vertex& v0, const vertex& v1) {
    bool off   = false;
    auto v0_it = begin(v0.iterators);
    auto v1_it = begin(v1.iterators);
    for (; v0_it != end(v0.iterators); ++v0_it, ++v1_it) {
      auto dist = std::abs(distance(*v0_it, *v1_it));
      if (dist > 1) { return false; }
      if (dist == 1 && !off) { off = true; }
      if (dist == 1 && off) { return false; }
    }
    return true;
  }
 private:
  //----------------------------------------------------------------------------
  template <size_t... Is, typename RandEng>
  constexpr auto random_vertex(std::index_sequence<Is...> /*is*/,
                               RandEng& eng) const {
    return vertex{linspace_iterator{
        &m_dimensions[Is],
        random_uniform<size_t, RandEng>{0, m_dimensions[Is].size() - 1, eng}()}...};
  }

 public:
  //----------------------------------------------------------------------------
  template <typename RandEng>
  auto random_vertex(RandEng& eng) -> vertex {
    return random_vertex(std::make_index_sequence<N>(), eng);
  }

  //----------------------------------------------------------------------------
  template <typename RandEng>
  auto random_vertex_neighbor_gaussian(const vertex& v, const Real _stddev,
                                       RandEng& eng) {
    auto neighbor = v;
    bool ok       = false;
    auto stddev   = _stddev;
    do {
      ok = true;
      for (size_t i = 0; i < N; ++i) {
        auto r = random_normal<Real>{}(
            0, std::min<Real>(stddev, neighbor[i].linspace().size() / 2), eng);
        // stddev -= r;
        neighbor[i].i() += static_cast<size_t>(r);
        if (neighbor[i].i() < 0 ||
            neighbor[i].i() >= neighbor[i].linspace().size()) {
          ok       = false;
          neighbor = v;
          stddev   = _stddev;
          break;
        }
      }
    } while (!ok);
    return neighbor;
  }
  //----------------------------------------------------------------------------
  template <typename OtherReal, size_t... Is>
  auto add_dimension(const linspace<OtherReal>& additional_dimension,
                     std::index_sequence<Is...> /*is*/) const {
    return grid<Real, N + 1>{m_dimensions[Is]..., additional_dimension};
  }

  //----------------------------------------------------------------------------
  template <typename OtherReal>
  auto add_dimension(const linspace<OtherReal>& additional_dimension) const {
    return add_dimension(additional_dimension, std::make_index_sequence<N>{});
  }

 // //----------------------------------------------------------------------------
 // private:
 //  template <size_t ReducedN>
 //  auto& remove_dimension(grid<Real, ReducedN>& reduced, size_t [>i<]) const {
 //    return reduced;
 //  }
 // // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 //  template <size_t ReducedN, typename... Omits,
 //            enable_if_integral<Omits...> = true>
 //  auto& remove_dimension(grid<Real, ReducedN>& reduced, size_t i, size_t omit,
 //                        Omits... omits) const {
 //    if (i != omit) {
 //      reduced.dimension(i) = m_dimensions[i];
 //      ++i;
 //    }
 //    return remove_dimension<ReducedN, Omits...>(reduced, i, omits...);
 //  }
 // // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 //public:
 // template <typename... Omits, enable_if_integral<Omits...> = true>
 // auto remove_dimension(Omits... omits) const {
 //   grid<Real, N - sizeof...(Omits)> reduced;
 //   return remove_dimension(reduced, 0, omits...);
 // }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if has_cxx17_support()
template <typename... Reals>
grid(const linspace<Reals>&...)->grid<promote_t<Reals...>, sizeof...(Reals)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, size_t... Is>
grid(const boundingbox<Real, N>& bb, const std::array<size_t, N>& res,
     std::index_sequence<Is...>)
    ->grid<Real, N>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
grid(const boundingbox<Real, N>& bb, const std::array<size_t, N>& res)
    ->grid<Real, N>;
#endif

//==============================================================================
template <typename Real, size_t N>
struct grid<Real, N>::vertex_container {
  const grid* g;
  auto        begin() const { return g->vertex_begin(); }
  auto        end() const { return g->vertex_end(); }
  auto        size() const { return g->num_vertices(); }
};

//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto operator+(const grid<Real, N>&  grid,
               const linspace<Real>& additional_dimension) {
  return grid.add_dimension(additional_dimension);
}

//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto operator+(const linspace<Real>& additional_dimension,
               const grid<Real, N>&  grid) {
  return grid.add_dimension(additional_dimension);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
