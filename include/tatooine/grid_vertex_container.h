#ifndef TATOOINE_GRID_VERTEX_CONTAINER_H
#define TATOOINE_GRID_VERTEX_CONTAINER_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <indexable_space... Dimensions>
class grid;
//==============================================================================
template <indexable_space... Dimensions>
struct grid_vertex_container {
  grid<Dimensions...> const& g;

  auto at(integral auto... is) const {
    static_assert(sizeof...(Dimensions) == sizeof...(is),
                  "Number of indices does not match number of dimensions.");
    return g.vertex_at(is...);
  }
  auto begin() const { return g.vertex_begin(); }
  auto end() const { return g.vertex_end(); }
  auto size() const { return g.num_vertices(); }
};
//------------------------------------------------------------------------------
template <indexable_space... Dimensions>
auto begin(grid_vertex_container<Dimensions...> const& c) {
  return c.begin();
}
//------------------------------------------------------------------------------
template <indexable_space... Dimensions>
auto end(grid_vertex_container<Dimensions...> const& c) {
  return c.end();
}
//------------------------------------------------------------------------------
template <indexable_space... Dimensions>
auto size(grid_vertex_container<Dimensions...> const& c) {
  return c.size();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace std::ranges {
template <tatooine::indexable_space... Dimensions>
constexpr auto begin(tatooine::grid_vertex_container<Dimensions...>& r) {
  r.begin();
}
template <tatooine::indexable_space... Dimensions>
constexpr auto end(tatooine::grid_vertex_container<Dimensions...>& r) {
  r.end();
}
template <tatooine::indexable_space... Dimensions>
constexpr auto begin(tatooine::grid_vertex_container<Dimensions...> const& r) {
  r.begin();
}
template <tatooine::indexable_space... Dimensions>
constexpr auto end(tatooine::grid_vertex_container<Dimensions...> const& r) {
  r.end();
}
}  // namespace std::ranges
#endif
