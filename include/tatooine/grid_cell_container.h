#ifndef TATOOINE_GRID_CELL_CONTAINER_H
#define TATOOINE_GRID_CELL_CONTAINER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/grid.h>
//#include <tatooine/grid_cell_handle.h>
//#include <tatooine/grid_cell_iterator.h>
//==============================================================================
namespace tatooine {
//==============================================================================
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
class grid;
//==============================================================================
template <typename... Dimensions>
struct grid_cell_container {
 private:
  grid<Dimensions...> const& m_grid;

 public:
  //using iterator          = grid_cell_iterator<Dimensions...>;
  //using const_iterator    = iterator;
  //using handle    = grid_cell_handle<Dimensions...>;
  //using pos_t = typename grid<Dimensions...>::pos_t;
  using seq_t = typename grid<Dimensions...>::seq_t;
  static constexpr auto num_dimensions() { return sizeof...(Dimensions); }
  //----------------------------------------------------------------------------
  grid_cell_container(grid<Dimensions...> const& g) : m_grid{g} {}
  //----------------------------------------------------------------------------
//#ifdef __cpp_concepts
//  template <size_t... DIs, integral Int>
//#else
//  template <size_t... DIs, typename Int, enable_if_integral<Int> = true>
//#endif
//  auto at(std::index_sequence<DIs...>,
//          std::array<Int, num_dimensions()> const& is) const
//      -> vec<real_t, num_dimensions()> {
//    return pos_t{static_cast<real_t>(m_grid.template dimension<DIs>()[is[DIs]])...};
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//#ifdef __cpp_concepts
//  template <size_t... DIs, integral... Is>
//#else
//  template <size_t... DIs, typename... Is, enable_if_integral<Is...> = true>
//#endif
//  auto at(std::index_sequence<DIs...>, Is const... is) const
//      -> vec<real_t, num_dimensions()> {
//    static_assert(sizeof...(DIs) == sizeof...(is));
//    static_assert(sizeof...(is) == num_dimensions());
//    return pos_t{static_cast<real_t>(m_grid.template dimension<DIs>()[is])...};
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// public:
//#ifdef __cpp_concepts
//  template <integral... Is>
//#else
//  template <typename... Is, enable_if_integral<Is...> = true>
//#endif
//  auto at(Is const... is) const {
//    static_assert(sizeof...(is) == num_dimensions());
//    return at(seq_t{}, is...);
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//#ifdef __cpp_concepts
//  template <integral Int>
//#else
//  template <typename Int, enable_if_integral<Int> = true>
//#endif
//  auto at(std::array<Int, num_dimensions()> const& is) const {
//    return at(seq_t{}, is);
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  auto at(handle const& h) const {
//    return at(seq_t{}, h.indices());
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  auto operator[](handle const& h) const {
//    return at(seq_t{}, h.indices());
//  }
//  //----------------------------------------------------------------------------
//#ifdef __cpp_concepts
//  template <integral... Is>
//#else
//  template <typename... Is, enable_if_integral<Is...> = true>
//#endif
//  auto operator()(Is const... is) const {
//    static_assert(sizeof...(is) == num_dimensions());
//    return at(is...);
//  }
//
//  //----------------------------------------------------------------------------
// private:
//  template <size_t... Is>
//  constexpr auto begin(std::index_sequence<Is...> [>seq<]) const {
//    return iterator{&m_grid, handle{std::array{((void)Is, size_t(0))...}, 0}};
//  }
//  //----------------------------------------------------------------------------
// public:
//  constexpr auto begin() const { return begin(seq_t{}); }
//  //----------------------------------------------------------------------------
// private:
//  template <size_t... Is>
//  constexpr auto end(std::index_sequence<Is...> [>seq<]) const {
//    auto it =  iterator{
//        &m_grid, handle{std::array{((void)Is, size_t(0))...}, size()}};
//    it->indices()[num_dimensions() - 1] =
//        m_grid.template size<num_dimensions() - 1>();
//    return it;
//  }
//  //----------------------------------------------------------------------------
// public:
//  constexpr auto end() const { return end(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto size(std::index_sequence<Is...> /*seq*/) const {
    return ((m_grid.template size<Is>() - 1) * ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto size() const { return size(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
  /// Sequential iteration implementation
#ifdef __cpp_concepts
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration,
            size_t... Ds>
#else
  template <typename Iteration, size_t... Ds,
            enable_if<is_invocable<Iteration,
                                   decltype(((void)std::declval<Dimensions>(),
                                             size_t{}))...> > = true>
#endif
  auto iterate_indices(Iteration&& iteration, tag::sequential_t,
                       std::index_sequence<Ds...>) const -> decltype(auto) {
    return for_loop(
        std::forward<Iteration>(iteration), tag::sequential,
        std::pair{size_t(0),
                  static_cast<size_t>(m_grid.template size<Ds>() - 1)}...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// Sequential iteration
#ifdef __cpp_concepts
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
#else
  template <typename Iteration,
            enable_if<is_invocable<Iteration,
                                   decltype(((void)std::declval<Dimensions>(),
                                             size_t{}))...> > = true>
#endif
  auto iterate_indices(Iteration&& iteration, tag::sequential_t) const
      -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration), tag::sequential,
                           std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
 private:
  /// Parallel iteration implementation
#ifdef __cpp_concepts
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration,
            size_t... Ds>
#else
  template <typename Iteration, size_t... Ds,
            enable_if<is_invocable<Iteration,
                                   decltype(((void)std::declval<Dimensions>(),
                                             size_t{}))...> > = true>
#endif
  auto iterate_indices(Iteration&& iteration, tag::parallel_t,
                       std::index_sequence<Ds...>) const -> decltype(auto) {
    return for_loop(
        std::forward<Iteration>(iteration), tag::parallel,
        std::pair{size_t(0),
                  static_cast<size_t>(m_grid.template size<Ds>() - 1)}...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// Parallel iteration
#ifdef __cpp_concepts
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
#else
  template <typename Iteration,
            enable_if<is_invocable<Iteration,
                                   decltype(((void)std::declval<Dimensions>(),
                                             size_t{}))...> > = true>
#endif
  auto iterate_indices(Iteration&& iteration, tag::parallel_t) const
      -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration), tag::parallel,
                           std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  /// Default iteration
#ifdef __cpp_concepts
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
#else
  template <typename Iteration,
            enable_if<is_invocable<Iteration,
                                   decltype(((void)std::declval<Dimensions>(),
                                             size_t{}))...> > = true>
#endif
  auto iterate_indices(Iteration&& iteration) const -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration), tag::sequential,
                           std::make_index_sequence<num_dimensions()>{});
  }
};
//------------------------------------------------------------------------------
//#ifdef __cpp_concepts
//template <indexable_space... Dimensions>
//#else
//template <typename... Dimensions>
//#endif
//auto begin(grid_cell_container<Dimensions...> const& c) {
//  return c.begin();
//}
////------------------------------------------------------------------------------
//#ifdef __cpp_concepts
//template <indexable_space... Dimensions>
//#else
//template <typename... Dimensions>
//#endif
//auto end(grid_cell_container<Dimensions...> const& c) {
//  return c.end();
//}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
auto size(grid_cell_container<Dimensions...> const& c) {
  return c.size();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace std::ranges {
//==============================================================================
//#ifdef __cpp_concepts
//template <tatooine::indexable_space... Dimensions>
//#else
//template <typename... Dimensions>
//#endif
//constexpr auto begin(tatooine::grid_cell_container<Dimensions...>& r) {
//  r.begin();
//}
//#ifdef __cpp_concepts
//template <tatooine::indexable_space... Dimensions>
//#else
//template <typename... Dimensions>
//#endif
//constexpr auto end(tatooine::grid_cell_container<Dimensions...>& r) {
//  r.end();
//}
//#ifdef __cpp_concepts
//template <tatooine::indexable_space... Dimensions>
//#else
//template <typename... Dimensions>
//#endif
//constexpr auto begin(tatooine::grid_cell_container<Dimensions...> const& r) {
//  r.begin();
//}
//#ifdef __cpp_concepts
//template <tatooine::indexable_space... Dimensions>
//#else
//template <typename... Dimensions>
//#endif
//constexpr auto end(tatooine::grid_cell_container<Dimensions...> const& r) {
//  r.end();
//}
//==============================================================================
}  // namespace std::ranges
//==============================================================================
#endif
