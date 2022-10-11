#ifndef TATOOINE_DETAIL_RECTILINEAR_GRID_CELL_CONTAINER_H
#define TATOOINE_DETAIL_RECTILINEAR_GRID_CELL_CONTAINER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/detail/rectilinear_grid/dimension.h>
//==============================================================================
namespace tatooine {
template <floating_point_range... Dimensions>
requires(sizeof...(Dimensions) > 1)
class rectilinear_grid;
}
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
//==============================================================================
template <typename... Dimensions>
struct cell_container {
  using grid_t = tatooine::rectilinear_grid<Dimensions...>;

 private:
  grid_t const& m_grid;

 public:
  //using iterator          = cell_iterator<Dimensions...>;
  //using const_iterator    = iterator;
  //using handle    = cell_handle<Dimensions...>;
  //using pos_type = typename grid_t::pos_type;
  using seq_t = typename grid_t::seq_t;
  static constexpr auto num_dimensions() -> std::size_t { return sizeof...(Dimensions); }
  //----------------------------------------------------------------------------
  explicit cell_container(grid_t const& g) : m_grid{g} {}
  //----------------------------------------------------------------------------
//  template <size_t... DIs, integral Int>
//  auto at(std::index_sequence<DIs...>,
//          std::array<Int, num_dimensions()> const& is) const
//      -> vec<real_type, num_dimensions()> {
//    return pos_type{static_cast<real_type>(m_grid.template dimension<DIs>()[is[DIs]])...};
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  template <size_t... DIs, integral... Is>
//  auto at(std::index_sequence<DIs...>, Is const... is) const
//      -> vec<real_type, num_dimensions()> {
//    static_assert(sizeof...(DIs) == sizeof...(is));
//    static_assert(sizeof...(is) == num_dimensions());
//    return pos_type{static_cast<real_type>(m_grid.template dimension<DIs>()[is])...};
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
  auto iterate_indices(Iteration&&                    iteration,
                       execution_policy::sequential_t exec,
                       std::index_sequence<Ds...> /*seq*/) const
      -> decltype(auto) {
    return tatooine::for_loop(
        std::forward<Iteration>(iteration), exec,
        std::pair{size_t(0),
                  static_cast<size_t>(m_grid.template size<Ds>() - 1)}...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// Sequential iteration
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
  auto iterate_indices(Iteration&& iteration, execution_policy::sequential_t exec) const
      -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration), exec,
                           std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
 private:
  /// Parallel iteration implementation
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration,
            size_t... Ds>
  auto iterate_indices(Iteration&& iteration, execution_policy::parallel_t exec,
                       std::index_sequence<Ds...> /*seq*/) const -> decltype(auto) {
    return tatooine::for_loop(
        std::forward<Iteration>(iteration), exec,
        std::pair{size_t(0),
                  static_cast<size_t>(m_grid.template size<Ds>() - 1)}...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// Parallel iteration
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
  auto iterate_indices(Iteration&& iteration, execution_policy::parallel_t exec) const
      -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration), exec,
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
//template <dimension... Dimensions>
//auto begin(cell_container<Dimensions...> const& c) {
//  return c.begin();
//}
////------------------------------------------------------------------------------
//template <dimension... Dimensions>
//auto end(cell_container<Dimensions...> const& c) {
//  return c.end();
//}
//------------------------------------------------------------------------------
template <dimension... Dimensions>
auto size(cell_container<Dimensions...> const& c) {
  return c.size();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
