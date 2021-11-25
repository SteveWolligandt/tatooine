#ifndef TATOOINE_RECTILINEAR_GRID_VERTEX_CONTAINER_H
#define TATOOINE_RECTILINEAR_GRID_VERTEX_CONTAINER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rectilinear_grid_vertex_handle.h>
#include <tatooine/rectilinear_grid_vertex_iterator.h>
//==============================================================================
namespace tatooine {
//==============================================================================
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
class rectilinear_grid;
//==============================================================================
template <typename... Dimensions>
struct rectilinear_grid_vertex_container {
  using grid_t         = rectilinear_grid<Dimensions...>;
  using iterator       = rectilinear_grid_vertex_iterator<Dimensions...>;
  using const_iterator = iterator;
  using handle         = typename grid_t::vertex_handle;
  using pos_t          = typename grid_t::pos_t;
  using seq_t          = typename grid_t::seq_t;
  static constexpr auto num_dimensions() { return sizeof...(Dimensions); }
  //----------------------------------------------------------------------------
 private:
  grid_t const& m_grid;
  //----------------------------------------------------------------------------
 public:
  rectilinear_grid_vertex_container(grid_t const& g) : m_grid{g} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto at(Is const... is) const {
    static_assert(sizeof...(is) == num_dimensions());
    return m_grid.vertex_at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(handle const& h) const { return m_grid.vertex_at(h); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](handle const& h) const { return m_grid.vertex_at(h); }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto operator()(Is const... is) const {
    static_assert(sizeof...(is) == num_dimensions());
    return at(is...);
  }

  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto begin(std::index_sequence<Is...> /*seq*/) const {
    return iterator{&m_grid, handle{std::array{((void)Is, size_t(0))...}, 0}};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto begin() const { return begin(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto end(std::index_sequence<Is...> /*seq*/) const {
    auto it =
        iterator{&m_grid, handle{std::array{((void)Is, size_t(0))...}, size()}};
    it->indices()[num_dimensions() - 1] =
        m_grid.template size<num_dimensions() - 1>();
    return it;
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto end() const { return end(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto size(std::index_sequence<Is...> /*seq*/) const {
    return (m_grid.template size<Is>() * ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto size() const { return size(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
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
  auto iterate_indices(Iteration&& iteration, execution_policy::sequential_t,
                       std::index_sequence<Ds...>) const -> decltype(auto) {
    return for_loop(
        std::forward<Iteration>(iteration), execution_policy::sequential,
        std::pair{size_t(0),
                  static_cast<size_t>(m_grid.template size<Ds>())}...);
  }
  //----------------------------------------------------------------------------
 public:
#ifdef __cpp_concepts
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
#else
  template <typename Iteration,
            enable_if<is_invocable<Iteration,
                                   decltype(((void)std::declval<Dimensions>(),
                                             size_t{}))...> > = true>
#endif
  auto iterate_indices(Iteration&& iteration,
                       execution_policy::sequential_t) const -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration),
                           execution_policy::sequential,
                           std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
 private:
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
  auto iterate_indices(Iteration&& iteration, execution_policy::parallel_t,
                       std::index_sequence<Ds...>) const -> decltype(auto) {
    return for_loop(std::forward<Iteration>(iteration),
                    execution_policy::parallel, m_grid.template size<Ds>()...);
  }
  //----------------------------------------------------------------------------
 public:
#ifdef __cpp_concepts
  template <invocable<decltype(((void)std::declval<Dimensions>(), size_t{}))...>
                Iteration>
#else
  template <typename Iteration,
            enable_if<is_invocable<Iteration,
                                   decltype(((void)std::declval<Dimensions>(),
                                             size_t{}))...> > = true>
#endif
  auto iterate_indices(Iteration&& iteration,
                       execution_policy::parallel_t) const -> decltype(auto) {
    return iterate_indices(std::forward<Iteration>(iteration),
                           execution_policy::parallel,
                           std::make_index_sequence<num_dimensions()>{});
  }
//------------------------------------------------------------------------------
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
    return iterate_indices(std::forward<Iteration>(iteration),
                           execution_policy::sequential,
                           std::make_index_sequence<num_dimensions()>{});
  }
};
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
auto begin(rectilinear_grid_vertex_container<Dimensions...> const& c) {
  return c.begin();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
auto end(rectilinear_grid_vertex_container<Dimensions...> const& c) {
  return c.end();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
auto size(rectilinear_grid_vertex_container<Dimensions...> const& c) {
  return c.size();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
