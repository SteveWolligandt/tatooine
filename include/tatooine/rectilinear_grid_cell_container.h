#ifndef TATOOINE_RECTILINEAR_GRID_CELL_CONTAINER_H
#define TATOOINE_RECTILINEAR_GRID_CELL_CONTAINER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/rectilinear_grid.h>
//#include <tatooine/grid_cell_handle.h>
//#include <tatooine/grid_cell_iterator.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <indexable_space... Dimensions>
class rectilinear_grid;
//==============================================================================
template <typename... Dimensions>
struct rectilinear_grid_cell_container {
 private:
  rectilinear_grid<Dimensions...> const& m_grid;

 public:
  //using iterator          = rectilinear_grid_cell_iterator<Dimensions...>;
  //using const_iterator    = iterator;
  //using handle    = rectilinear_grid_cell_handle<Dimensions...>;
  //using pos_t = typename rectilinear_grid<Dimensions...>::pos_t;
  using seq_t = typename rectilinear_grid<Dimensions...>::seq_t;
  static constexpr auto num_dimensions() { return sizeof...(Dimensions); }
  //----------------------------------------------------------------------------
  rectilinear_grid_cell_container(rectilinear_grid<Dimensions...> const& g) : m_grid{g} {}
  //----------------------------------------------------------------------------
//  template <size_t... DIs, integral Int>
//  auto at(std::index_sequence<DIs...>,
//          std::array<Int, num_dimensions()> const& is) const
//      -> vec<real_t, num_dimensions()> {
//    return pos_t{static_cast<real_t>(m_grid.template dimension<DIs>()[is[DIs]])...};
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  template <size_t... DIs, integral... Is>
//  auto at(std::index_sequence<DIs...>, Is const... is) const
//      -> vec<real_t, num_dimensions()> {
//    static_assert(sizeof...(DIs) == sizeof...(is));
//    static_assert(sizeof...(is) == num_dimensions());
//    return pos_t{static_cast<real_t>(m_grid.template dimension<DIs>()[is])...};
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// public:
//  template <integral... Is>
//  auto at(Is const... is) const {
//    static_assert(sizeof...(is) == num_dimensions());
//    return at(seq_t{}, is...);
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  template <integral Int>
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
//  template <integral... Is>
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
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration,
            size_t... Ds>
  auto iterate_indices(Iteration&& iteration, execution_policy::sequential_t,
                       std::index_sequence<Ds...>) const -> decltype(auto) {
    return for_loop(
        std::forward<Iteration>(iteration), execution_policy::sequential,
        std::pair{size_t(0),
                  static_cast<size_t>(m_grid.template size<Ds>() - 1)}...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// Sequential iteration
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
  auto iterate_indices(Iteration&& iteration, execution_policy::sequential_t) const
      -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration), execution_policy::sequential,
                           std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
 private:
  /// Parallel iteration implementation
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration,
            size_t... Ds>
  auto iterate_indices(Iteration&& iteration, execution_policy::parallel_t,
                       std::index_sequence<Ds...>) const -> decltype(auto) {
    return for_loop(
        std::forward<Iteration>(iteration), execution_policy::parallel,
        std::pair{size_t(0),
                  static_cast<size_t>(m_grid.template size<Ds>() - 1)}...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// Parallel iteration
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
  auto iterate_indices(Iteration&& iteration, execution_policy::parallel_t) const
      -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration), execution_policy::parallel,
                           std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  /// Default iteration
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
  auto iterate_indices(Iteration&& iteration) const -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration), execution_policy::sequential,
                           std::make_index_sequence<num_dimensions()>{});
  }
};
//------------------------------------------------------------------------------
//template <indexable_space... Dimensions>
//auto begin(rectilinear_grid_cell_container<Dimensions...> const& c) {
//  return c.begin();
//}
////------------------------------------------------------------------------------
//template <indexable_space... Dimensions>
//auto end(rectilinear_grid_cell_container<Dimensions...> const& c) {
//  return c.end();
//}
//------------------------------------------------------------------------------
template <indexable_space... Dimensions>
auto size(rectilinear_grid_cell_container<Dimensions...> const& c) {
  return c.size();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace std::ranges {
//==============================================================================
//template <tatooine::indexable_space... Dimensions>
//constexpr auto begin(tatooine::rectilinear_grid_cell_container<Dimensions...>& r) {
//  r.begin();
//}
//template <tatooine::indexable_space... Dimensions>
//constexpr auto end(tatooine::rectilinear_grid_cell_container<Dimensions...>& r) {
//  r.end();
//}
//template <tatooine::indexable_space... Dimensions>
//constexpr auto begin(tatooine::rectilinear_grid_cell_container<Dimensions...> const& r) {
//  r.begin();
//}
//template <tatooine::indexable_space... Dimensions>
//constexpr auto end(tatooine::rectilinear_grid_cell_container<Dimensions...> const& r) {
//  r.end();
//}
//==============================================================================
}  // namespace std::ranges
//==============================================================================
#endif
