#ifndef TATOOINE_UNSTRUCTURED_TRIANGULAR_GRID_H
#define TATOOINE_UNSTRUCTURED_TRIANGULAR_GRID_H
//==============================================================================
#include <tatooine/unstructured_simplicial_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
using unstructured_triangular_grid = unstructured_simplicial_grid<Real, N, 2>;
template <size_t N>
using UnstructuredTriangularGrid = unstructured_triangular_grid<real_t, N>;
template <typename Real>
using UnstructuredTriangularGrid2   = unstructured_triangular_grid<Real, 2>;
template <typename Real>
using UnstructuredTriangularGrid3   = unstructured_triangular_grid<Real, 3>;
template <typename Real>
using UnstructuredTriangularGrid4   = unstructured_triangular_grid<Real, 4>;
template <typename Real>
using UnstructuredTriangularGrid5   = unstructured_triangular_grid<Real, 5>;
using unstructured_triangular_grid2 = UnstructuredTriangularGrid<2>;
using unstructured_triangular_grid3 = UnstructuredTriangularGrid<3>;
using unstructured_triangular_grid4 = UnstructuredTriangularGrid<4>;
using unstructured_triangular_grid5 = UnstructuredTriangularGrid<5>;
//==============================================================================
namespace detail {
//==============================================================================
template <typename MeshCont>
auto write_unstructured_triangular_grid_container_to_vtk(
    MeshCont const& meshes, std::filesystem::path const& path,
    std::string const& title) {
  vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
  if (writer.is_open()) {
    size_t num_pts   = 0;
    size_t cur_first = 0;
    for (auto const& m : meshes) {
      num_pts += m.vertices().size();
    }
    std::vector<std::array<typename MeshCont::value_type::real_t, 3>> points;
    std::vector<std::vector<size_t>>                                  faces;
    points.reserve(num_pts);
    faces.reserve(meshes.size());

    for (auto const& m : meshes) {
      // add points
      for (auto const& v : m.vertices()) {
        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
      }

      // add faces
      for (auto c : m.cells()) {
        faces.emplace_back();
        auto [v0, v1, v2] = m[c];
        faces.back().push_back(cur_first + v0.i);
        faces.back().push_back(cur_first + v1.i);
        faces.back().push_back(cur_first + v2.i);
      }
      cur_first += m.vertices().size();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_polygons(faces);
    // writer.write_point_data(num_pts);
    writer.close();
  }
}
}  // namespace detail
//==============================================================================
template <typename Real>
auto write_vtk(std::vector<unstructured_triangular_grid<Real, 3>> const& meshes,
               std::string const&                                        path,
               std::string const& title = "tatooine meshes") {
  detail::write_unstructured_triangular_grid_container_to_vtk(meshes, path,
                                                              title);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif

