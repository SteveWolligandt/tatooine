#ifndef TATOOINE_DETAIL_RECTILINEAR_GRID_VERTEX_CONTAINER_H
#define TATOOINE_DETAIL_RECTILINEAR_GRID_VERTEX_CONTAINER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/detail/rectilinear_grid/vertex_handle.h>
#include <tatooine/detail/rectilinear_grid/vertex_iterator.h>
#include <tatooine/for_loop.h>
#include <tatooine/rectilinear_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <detail::rectilinear_grid::dimension... Dimensions>
requires(sizeof...(Dimensions) > 1)
class rectilinear_grid;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
template <typename... Dimensions>
struct vertex_container {
  using grid_type      = tatooine::rectilinear_grid<Dimensions...>;
  using iterator       = vertex_iterator<Dimensions...>;
  using const_iterator = iterator;
  using handle         = typename grid_type::vertex_handle;
  using pos_type       = typename grid_type::pos_type;
  using sequence_type  = typename grid_type::sequence_type;
  static constexpr auto num_dimensions() -> std::size_t { return sizeof...(Dimensions); }
  //----------------------------------------------------------------------------
 private:
  grid_type const& m_grid;
  //----------------------------------------------------------------------------
 public:
  vertex_container(grid_type const& g) : m_grid{g} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions());
    return m_grid.vertex_at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(handle const& h) const { return m_grid.vertex_at(h); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](handle const& h) const { return m_grid.vertex_at(h); }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions());
    return at(is...);
  }

  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  constexpr auto begin(std::index_sequence<Is...> /*seq*/) const {
    return iterator{&m_grid,
                    handle{std::array{((void)Is, std::size_t(0))...}, 0}};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto begin() const { return begin(sequence_type{}); }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  constexpr auto end(std::index_sequence<Is...> /*seq*/) const {
    auto it = iterator{
        &m_grid, handle{std::array{((void)Is, std::size_t(0))...}, size()}};
    it->indices()[num_dimensions() - 1] =
        m_grid.template size<num_dimensions() - 1>();
    return it;
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto end() const { return end(sequence_type{}); }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  constexpr auto size(std::index_sequence<Is...> /*seq*/) const {
    return (m_grid.template size<Is>() * ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto size() const { return size(sequence_type{}); }
  //----------------------------------------------------------------------------
 private:
  template <
      invocable<decltype(((void)std::declval<Dimensions>(), std::size_t{}))...>
          Iteration,
      std::size_t... Ds>
  auto iterate_indices(Iteration&&                    iteration,
                       execution_policy_tag auto const exec,
                       std::index_sequence<Ds...>) const -> decltype(auto) {
    return tatooine::for_loop(
        std::forward<Iteration>(iteration), exec,
        static_cast<std::size_t>(m_grid.template size<Ds>())...);
  }
  //----------------------------------------------------------------------------
 public:
  template <
      invocable<decltype(((void)std::declval<Dimensions>(), std::size_t{}))...>
          Iteration>
  auto iterate_indices(Iteration&&                     iteration,
                       execution_policy_tag auto const exec) const
      -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration), exec,
                           std::make_index_sequence<num_dimensions()>{});
  }
  //------------------------------------------------------------------------------
  template <
      invocable<decltype(((void)std::declval<Dimensions>(), std::size_t{}))...>
          Iteration>
  auto iterate_indices(Iteration&& iteration) const -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration),
                           execution_policy::sequential,
                           std::make_index_sequence<num_dimensions()>{});
  }
  //------------------------------------------------------------------------------
  template <invocable<pos_type> Iteration>
  auto iterate_positions(Iteration&& iteration) const -> decltype(auto) {
    return iterate_indices(
        [this, iteration = std::forward<Iteration>(iteration)](
            auto const... is) { iteration(this->at(is...)); },
        execution_policy::sequential,
        std::make_index_sequence<num_dimensions()>{});
  }
  //------------------------------------------------------------------------------
  template <invocable<pos_type> Iteration>
  auto iterate_positions(Iteration&&                     iteration,
                         execution_policy_tag auto const exec) const
      -> decltype(auto) {
    return iterate_indices(
        [this, iteration = std::forward<Iteration>(iteration)](
            auto const... is) { iteration(this->at(is...)); },
        exec, std::make_index_sequence<num_dimensions()>{});
  }
};
//------------------------------------------------------------------------------
template <dimension... Dimensions>
auto begin(vertex_container<Dimensions...> const& c) {
  return c.begin();
}
//------------------------------------------------------------------------------
template <dimension... Dimensions>
auto end(vertex_container<Dimensions...> const& c) {
  return c.end();
}
//------------------------------------------------------------------------------
template <dimension... Dimensions>
auto size(vertex_container<Dimensions...> const& c) {
  return c.size();
}
//==============================================================================
}  // namespace tatooine::detail::rectilinear_grid
//==============================================================================
#endif
