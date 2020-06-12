#ifndef TATOOINE_GRID_H
#define TATOOINE_GRID_H
//==============================================================================
#include <tatooine/algorithm.h>
#include <tatooine/boundingbox.h>
#include <tatooine/concepts.h>
//#include <tatooine/grid_edge.h>
#include <tatooine/grid_vertex_iterator.h>
//#include <tatooine/grid_vertex_edges.h>
//#include <tatooine/grid_vertex_neighbors.h>
#include <tatooine/index_ordering.h>
#include <tatooine/linspace.h>
#include <tatooine/random.h>
//#include <tatooine/subgrid.h>
#include <tatooine/utility.h>

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <cassert>
#include <set>
#include <tuple>
//==============================================================================
namespace tatooine {
//==============================================================================
template <indexable_space... Dimensions>
class grid {
 public:
  static constexpr auto num_dimensions() { return sizeof...(Dimensions); }
  using this_t          = grid<Dimensions...>;
  using real_t          = promote_t<typename Dimensions::value_type...>;
  using vec_t           = vec<real_t, num_dimensions()>;
  using pos_t           = vec_t;
  using vertex_iterator = grid_vertex_iterator<Dimensions...>;
  // using edge            = grid_edge<real_t, num_dimensions()>;
  // using edge_iterator   = grid_edge_iterator<real_t, num_dimensions()>;
  using seq_t = std::make_index_sequence<num_dimensions()>;

  struct vertex_container;

  //============================================================================
 private:
  std::tuple<std::decay_t<Dimensions>...> m_dimensions;

  //============================================================================
 public:
  constexpr grid()                      = default;
  constexpr grid(grid const& other)     = default;
  constexpr grid(grid&& other) noexcept = default;
  //----------------------------------------------------------------------------
  // template <typename OtherReal, size_t... Is>
  // constexpr grid(grid<OtherReal, num_dimensions()> const& other,
  //               std::index_sequence<Is...> [>is<])
  //    : m_dimensions{other.dimension(Is)...} {}
  // template <typename OtherReal>
  // explicit constexpr grid(grid<OtherReal, num_dimensions()> const& other)
  //    : grid(other, seq_t{}) {}

  //----------------------------------------------------------------------------
  template <indexable_space... _Dimensions>
  explicit constexpr grid(_Dimensions&&... dimensions)
      : m_dimensions{std::forward<Dimensions>(dimensions)...} {
    static_assert(sizeof...(Dimensions) == num_dimensions(),
                  "number of given dimensions does not match number of "
                  "specified dimensions");
  }

  //----------------------------------------------------------------------------
  template <typename Real, size_t... Is>
  constexpr grid(boundingbox<Real, num_dimensions()> const&  bb,
                 std::array<size_t, num_dimensions()> const& res,
                 std::index_sequence<Is...> /*is*/)
      : m_dimensions{linspace<real_t>{real_t(bb.min(Is)), real_t(bb.max(Is)),
                                      res[Is]}...} {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  template <typename Real>
  constexpr grid(boundingbox<Real, num_dimensions()> const&  bb,
                 std::array<size_t, num_dimensions()> const& res)
      : grid{bb, res, seq_t{}} {}
  //----------------------------------------------------------------------------
  ~grid() = default;
  //============================================================================
  constexpr auto operator=(grid const& other) -> grid& = default;
  constexpr auto operator=(grid&& other) noexcept -> grid& = default;

  //----------------------------------------------------------------------------
  // template <typename Real>
  // constexpr auto operator=(grid<Real, num_dimensions()>const& other) ->
  // grid& {
  //  for (size_t i = 0; i < num_dimensions(); ++i) { m_dimensions[i] =
  //  other.dimension(i); } return *this;
  //}
  //----------------------------------------------------------------------------
  template <size_t i>
  constexpr auto dimension() -> auto& {
    return std::get<i>(m_dimensions);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t i>
  constexpr auto dimension() const -> auto const& {
    return std::get<i>(m_dimensions);
  }
  //----------------------------------------------------------------------------
  constexpr auto dimensions() -> auto& { return m_dimensions; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto dimensions() const -> auto const& { return m_dimensions; }
  //----------------------------------------------------------------------------
  constexpr auto front_dimension() -> auto& {
    return std::get<0>(m_dimensions);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto front_dimension() const -> auto const& {
    return std::get<0>(m_dimensions);
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto min(std::index_sequence<Is...> /*is*/) const {
    return vec<real_t, num_dimensions()>{static_cast<real_t>(front<Is>())...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto min() const { return min(seq_t{}); }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto max(std::index_sequence<Is...> /*is*/) const {
    return vec<real_t, num_dimensions()>{static_cast<real_t>(back<Is>())...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto max() const { return max(seq_t{}); }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto resolution(std::index_sequence<Is...> /*is*/) const {
    return vec<size_t, num_dimensions()>{size<Is>()...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto resolution() const { return resolution(seq_t{}); }
  //----------------------------------------------------------------------------
 public:
  constexpr auto boundingbox() const { return boundingbox(seq_t{}); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  template <size_t... Is>
  constexpr auto boundingbox(std::index_sequence<Is...> /*is*/) const {
    static_assert(sizeof...(Is) == num_dimensions());
    return tatooine::boundingbox<real_t, num_dimensions()>{
        vec<real_t, num_dimensions()>{static_cast<real_t>(front<Is>())...},
        vec<real_t, num_dimensions()>{static_cast<real_t>(back<Is>())...}};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto size() const { return size(seq_t{}); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 protected:
  template <size_t... Is>
  constexpr auto size(std::index_sequence<Is...> /*is*/) const {
    static_assert(sizeof...(Is) == num_dimensions());
    return vec<size_t, num_dimensions()>{size<Is>()...};
  }
  //----------------------------------------------------------------------------
 public:
  template <size_t I>
  constexpr auto size() const {
    return size(dimension<I>());
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto front() const {
    return dimension<I>().front();
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto back() const {
    return dimension<I>().back();
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto in_domain(std::index_sequence<Is...> /*is*/,
                           real_number auto const... xs) const {
    static_assert(sizeof...(xs) == num_dimensions(),
                  "number of components does not match number of dimensions");
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return ((std::get<Is>(m_dimensions).front() <= xs &&
             xs <= std::get<Is>(m_dimensions).back()) &&
            ...);
  }

  //----------------------------------------------------------------------------
  constexpr auto in_domain(real_number auto const... xs) const {
    static_assert(sizeof...(xs) == num_dimensions(),
                  "number of components does not match number of dimensions");
    return in_domain(seq_t{}, xs...);
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto in_domain(std::array<real_t, num_dimensions()> const& x,
                           std::index_sequence<Is...> /*is*/) const {
    return in_domain(x[Is]...);
  }

  //----------------------------------------------------------------------------
  constexpr auto in_domain(
      std::array<real_t, num_dimensions()> const& x) const {
    return in_domain(x, seq_t{});
  }
  //----------------------------------------------------------------------------
  template <size_t... DIs>
  auto vertex_at(std::index_sequence<DIs...>, integral auto const... is) const
      -> vec<real_t, num_dimensions()> {
    static_assert(sizeof...(DIs) == sizeof...(is));
    static_assert(sizeof...(is) == num_dimensions());
    return pos_t{static_cast<real_t>((std::get<DIs>(m_dimensions)[is]))...};
  }
  //----------------------------------------------------------------------------
  auto vertex_at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions());
    return vertex_at(seq_t{}, is...);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions());
    return vertex_at(is...);
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto num_vertices(std::index_sequence<Is...> /*seq*/) const {
    return (size<Is>() * ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto num_vertices() const { return num_vertices(seq_t{}); }
  //----------------------------------------------------------------------------
  /// \return number of dimensions for one dimension dim
  // constexpr auto edges() const { return grid_edge_container{this}; }

  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto vertex_begin(std::index_sequence<Is...> /*is*/) const {
    return vertex_iterator{this, std::array{((void)Is, size_t(0))...}};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto vertex_begin() const { return vertex_begin(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto vertex_end(std::index_sequence<Is...> /*is*/) const {
    return vertex_iterator{
        this, vertex_iterator{this, std::array{((void)Is, size_t(0))...,
                                               size<num_dimensions() - 1>()}}};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto vertex_end() const {
    return vertex_end(std::make_index_sequence<num_dimensions() - 1>());
  }
  //----------------------------------------------------------------------------
  auto vertices() const { return vertex_container{this}; }
  //----------------------------------------------------------------------------
  //auto neighbors(vertex const& v) const {
  //  return grid_vertex_neighbors<real_t, num_dimensions()>(v);
  //}
  //----------------------------------------------------------------------------
  // auto edges(vertex const& v) const { return grid_vertex_edges<real_t,
  // num_dimensions()>(v); }
  ////----------------------------------------------------------------------------
  // auto sub(vertex const& begin_vertex, vertex const& end_vertex) const {
  //  return subgrid<real_t, num_dimensions()>(this, begin_vertex, end_vertex);
  //}
  ////----------------------------------------------------------------------------
  ///// checks if an edge e has vertex v as point
  // auto contains(vertex const& v, edge const& e) {
  //  return v == e.first || v == e.second;
  //}
  //----------------------------------------------------------------------------
  /// checks if v0 and v1 are direct or diagonal neighbors
  //auto are_neighbors(vertex const& v0, vertex const& v1) {
  //  auto v0_it = begin(v0.iterators);
  //  auto v1_it = begin(v1.iterators);
  //  for (; v0_it != end(v0.iterators); ++v0_it, ++v1_it) {
  //    if (distance(*v0_it, *v1_it) > 1) { return false; }
  //  }
  //  return true;
  //}
  //----------------------------------------------------------------------------
  /// checks if v0 and v1 are direct neighbors
  //auto are_direct_neighbors(vertex const& v0, vertex const& v1) {
  //  bool off   = false;
  //  auto v0_it = begin(v0.iterators);
  //  auto v1_it = begin(v1.iterators);
  //  for (; v0_it != end(v0.iterators); ++v0_it, ++v1_it) {
  //    auto dist = std::abs(distance(*v0_it, *v1_it));
  //    if (dist > 1) { return false; }
  //    if (dist == 1 && !off) { off = true; }
  //    if (dist == 1 && off) { return false; }
  //  }
  //  return true;
  //}

  //----------------------------------------------------------------------------
 private:
  //template <size_t... Is, typename RandEng>
  //constexpr auto random_vertex(std::index_sequence<Is...> [>is<],
  //                             RandEng& eng) const {
  //  return vertex{linspace_iterator{
  //      &std::get<Is>(m_dimensions),
  //      random_uniform<size_t, RandEng>{0, size(std::get<Is>(m_dimensions)) - 1,
  //                                      eng}()}...};
  //}

  //----------------------------------------------------------------------------
 public:
  //template <typename RandEng>
  //auto random_vertex(RandEng& eng) -> vertex {
  //  return random_vertex(seq_t{}, eng);
  //}

  //----------------------------------------------------------------------------
  //template <typename RandEng>
  //auto random_vertex_neighbor_gaussian(vertex const& v, real_t const _stddev,
  //                                     RandEng& eng) {
  //  auto neighbor = v;
  //  bool ok       = false;
  //  auto stddev   = _stddev;
  //  do {
  //    ok = true;
  //    for (size_t i = 0; i < num_dimensions(); ++i) {
  //      auto r = random_normal<real_t>{}(
  //          0, std::min<real_t>(stddev, neighbor[i].linspace().size() / 2),
  //          eng);
  //      // stddev -= r;
  //      neighbor[i].i() += static_cast<size_t>(r);
  //      if (neighbor[i].i() < 0 ||
  //          neighbor[i].i() >= neighbor[i].linspace().size()) {
  //        ok       = false;
  //        neighbor = v;
  //        stddev   = _stddev;
  //        break;
  //      }
  //    }
  //  } while (!ok);
  //  return neighbor;
  //}
  //----------------------------------------------------------------------------
  template <indexable_space AdditionalDimension, size_t... Is>
  auto add_dimension(AdditionalDimension&& additional_dimension,
                     std::index_sequence<Is...> /*is*/) const {
    return grid<Dimensions..., std::decay_t<AdditionalDimension>>{
        dimension<Is>()...,
        std::forward<AdditionalDimension>(additional_dimension)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <indexable_space AdditionalDimension>
  auto add_dimension(indexable_space auto&& additional_dimension) const {
    return add_dimension(
        std::forward<AdditionalDimension>(additional_dimension), seq_t{});
  }

  // //----------------------------------------------------------------------------
  // private:
  //  template <size_t ReducedN>
  //  auto& remove_dimension(grid<real_t, ReducedN>& reduced, size_t [>i<])
  //  const {
  //    return reduced;
  //  }
  // // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // - -
  //  template <size_t ReducedN, integral... Omits>
  //  auto& remove_dimension(grid<real_t, ReducedN>& reduced, size_t i, size_t
  //  omit,
  //                        Omits... omits) const {
  //    if (i != omit) {
  //      reduced.dimension(i) = m_dimensions[i];
  //      ++i;
  //    }
  //    return remove_dimension<ReducedN, Omits...>(reduced, i, omits...);
  //  }
  // // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // - -
  // public:
  // template <integral... Omits>
  // auto remove_dimension(Omits... omits) const {
  //   grid<real_t, num_dimensions() - sizeof...(Omits)> reduced;
  //   return remove_dimension(reduced, 0, omits...);
  // }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <indexable_space... Dimensions>
grid(Dimensions&&...) -> grid<std::decay_t<Dimensions>...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, size_t... Is>
grid(boundingbox<Real, N> const& bb,  std::array<size_t, N> const& res,
     std::index_sequence<Is...>)
    -> grid<decltype(((void)Is, std::declval<linspace<Real>()>))...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Real, size_t num_dimensions()>
//grid(boundingbox<Real, num_dimensions()> const& bb,
//     std::array<size_t, num_dimensions()> const&  res)
//    -> grid<Real, num_dimensions()>;

//==============================================================================
template <indexable_space... Dimensions>
struct grid<Dimensions...>::vertex_container {
  grid const* const g;
  auto        begin() const { return g->vertex_begin(); }
  auto        end() const { return g->vertex_end(); }
  auto        size() const { return g->num_vertices(); }
};

//------------------------------------------------------------------------------
template <indexable_space... Dimensions, indexable_space AdditionalDimension>
auto operator+(
    grid<Dimensions...> const& grid,
    AdditionalDimension&& additional_dimension) {
  return grid.add_dimension(
      std::forward<AdditionalDimension>(additional_dimension));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <indexable_space... Dimensions, indexable_space AdditionalDimension>
auto operator+(AdditionalDimension&& additional_dimension,
               grid<Dimensions...> const&            grid) {
  return grid.add_dimension(
      std::forward<AdditionalDimension>(additional_dimension));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
