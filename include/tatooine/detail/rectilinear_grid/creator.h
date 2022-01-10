#ifndef TATOOINE_DETAIL_RECTILINEAR_GRID_CREATOR_H
#define TATOOINE_DETAIL_RECTILINEAR_GRID_CREATOR_H
//==============================================================================
#include <tatooine/detail/rectilinear_grid/dimension.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <detail::rectilinear_grid::dimension... Dimensions>
class rectilinear_grid;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
//==============================================================================
template <dimension IndexableSpace, std::size_t N>
struct creator {
 private:
  template <typename... Args, std::size_t... Seq>
  static constexpr auto create(std::index_sequence<Seq...> /*seq*/,
                               Args&&... args) {
    return tatooine::rectilinear_grid<decltype((static_cast<void>(Seq),
                                      IndexableSpace{}))...>{
        std::forward<Args>(args)...};
  }
  template <typename... Args>
  static constexpr auto create(Args&&... args) {
    return create(std::make_index_sequence<N>{}, std::forward<Args>(args)...);
  }

 public:
  using type = decltype(create());
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <dimension IndexableSpace, std::size_t N>
using creator_t =
    typename detail::rectilinear_grid::creator<IndexableSpace, N>::type;
//==============================================================================
}  // namespace tatooine::detail::rectilinear_grid
//==============================================================================
#endif
