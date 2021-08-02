#ifndef TATOOINE_UNSTRUCTURED_TRIANGULAR_GRID_H
#define TATOOINE_UNSTRUCTURED_TRIANGULAR_GRID_H
//==============================================================================
#include <tatooine/unstructured_simplex_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
using unstructured_triangular_grid = unstructured_simplex_grid<Real, N, 2>;
template <size_t N>
using UnstructuredTriangularGird     = unstructured_triangular_grid<real_t, N>;
using unstructured_triangular_grid_2 = UnstructuredTriangularGird<2>;
using unstructured_triangular_grid_3 = UnstructuredTriangularGird<3>;
using unstructured_triangular_grid_4 = UnstructuredTriangularGird<4>;
using unstructured_triangular_grid_5 = UnstructuredTriangularGird<5>;
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
      for (auto t : m.faces()) {
        faces.emplace_back();
        faces.back().push_back(cur_first + m[t][0].i);
        faces.back().push_back(cur_first + m[t][1].i);
        faces.back().push_back(cur_first + m[t][2].i);
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

