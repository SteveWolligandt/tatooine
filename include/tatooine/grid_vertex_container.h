#ifndef TATOOINE_GRID_VERTEX_CONTAINER_H
#define TATOOINE_GRID_VERTEX_CONTAINER_H
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
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
struct grid_vertex_container {
  grid<Dimensions...> const& g;

#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto at(Is const... is) const {
    static_assert(sizeof...(Dimensions) == sizeof...(is),
                  "Number of indices does not match number of dimensions.");
    return g.vertex_at(is...);
  }
  auto begin() const { return g.vertex_begin(); }
  auto end() const { return g.vertex_end(); }
  auto size() const { return g.num_vertices(); }
};
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
auto begin(grid_vertex_container<Dimensions...> const& c) {
  return c.begin();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
auto end(grid_vertex_container<Dimensions...> const& c) {
  return c.end();
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
auto size(grid_vertex_container<Dimensions...> const& c) {
  return c.size();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace std::ranges {
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
constexpr auto begin(tatooine::grid_vertex_container<Dimensions...>& r) {
  r.begin();
}
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
constexpr auto end(tatooine::grid_vertex_container<Dimensions...>& r) {
  r.end();
}
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
constexpr auto begin(tatooine::grid_vertex_container<Dimensions...> const& r) {
  r.begin();
}
#ifdef __cpp_concepts
template <indexable_space... Dimensions>
#else
template <typename... Dimensions>
#endif
constexpr auto end(tatooine::grid_vertex_container<Dimensions...> const& r) {
  r.end();
}
}  // namespace std::ranges
#endif
