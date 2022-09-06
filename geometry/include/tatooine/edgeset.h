#ifndef TATOOINE_EDGESET_H
#define TATOOINE_EDGESET_H
//==============================================================================
#include <tatooine/unstructured_simplicial_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct edgeset : unstructured_simplicial_grid<Real, NumDimensions, 1> {
  using this_type   = edgeset<Real, NumDimensions>;
  using parent_type = unstructured_simplicial_grid<Real, NumDimensions, 1>;
  using parent_type::parent_type;
  using typename parent_type::vertex_handle;
  using edge_handle = typename parent_type::simplex_handle;
  template <typename... Handles>
  auto insert_edge(vertex_handle const v0, vertex_handle const v1)  {
    return this->insert_simplex(v0, v1);
  }
  auto edge_at(edge_handle const h) { return this->simplex_at(h); }
  auto edge_at(edge_handle const h) const { return this->simplex_at(h); }
  auto edges() const { return this->simplices(); }
  auto are_connected(vertex_handle const v0, vertex_handle const v1) const {
    for (auto e : edges()) {
      auto [ev0, ev1] = this->at(e);
      if ((ev0 == v0 && ev1 == v1) || (ev0 == v1 && ev1 == v0)) {
        return true;
      }
    }
    return false;
  }
};
//==============================================================================
template <std::size_t NumDimensions>
using Edgeset = edgeset<real_number, NumDimensions>;
template <floating_point Real>
using Edgeset2 = edgeset<Real, 2>;
template <floating_point Real>
using Edgeset3 = edgeset<Real, 3>;
template <floating_point Real>
using Edgeset4 = edgeset<Real, 4>;
template <floating_point Real>
using Edgeset5 = edgeset<Real, 5>;
using edgeset2 = Edgeset<2>;
using edgeset3 = Edgeset<3>;
using edgeset4 = Edgeset<4>;
using edgeset5 = Edgeset<5>;
//==============================================================================
template <typename T>
struct is_edgeset_impl : std::false_type {};
template <floating_point Real, std::size_t NumDimensions>
struct is_edgeset_impl<edgeset<Real, NumDimensions>> : std::true_type {};
template <typename T>
static constexpr auto is_edgeset = is_edgeset_impl<T>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
//#include <tatooine/detail/edgeset/vtp_writer.h>
//namespace tatooine {
////==============================================================================
//static constexpr auto is_edgeset = is_edgeset_impl<T>::value;
//auto write_vtk(range auto const& grids, std::filesystem::path const& path,
//               std::string const& title = "tatooine grids") requires
//    is_edgeset<typename std::decay_t<decltype(grids)>::value_type> {
//  detail::write_edgeset_container_to_vtk(grids, path, title);
//}
////------------------------------------------------------------------------------
//auto write_vtp(range auto const&            grids,
//               std::filesystem::path const& path) requires
//    is_edgeset<typename std::decay_t<decltype(grids)>::value_type> {
//  detail::write_edgeset_container_to_vtp(grids, path);
//}
////------------------------------------------------------------------------------
//auto write(range auto const& grids, std::filesystem::path const& path) requires
//    is_edgeset<typename std::decay_t<decltype(grids)>::value_type> {
//  auto const ext = path.extension();
//  if (ext == ".vtp") {
//    detail::write_edgeset_container_to_vtp(grids, path);
//  } else if (ext == ".vtk") {
//    detail::write_edgeset_container_to_vtk(grids, path);
//  }
//}
//==============================================================================
//}  // namespace tatooine
//==============================================================================
#endif
