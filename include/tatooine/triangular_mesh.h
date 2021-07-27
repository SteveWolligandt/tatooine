#ifndef TATOOINE_TRIANGULAR_MESH_H
#define TATOOINE_TRIANGULAR_MESH_H
//==============================================================================
#include <tatooine/simplex_mesh.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
using triangular_mesh = simplex_mesh<Real, N, 2>;
template <size_t N>
using TriangularMesh   = triangular_mesh<real_t, N>;
using triangular_mesh_2 = TriangularMesh<2>;
using triangular_mesh_3 = TriangularMesh<3>;
using triangular_mesh_4 = TriangularMesh<4>;
using triangular_mesh_5 = TriangularMesh<5>;
//==============================================================================
namespace detail {
//==============================================================================
template <typename MeshCont>
auto write_mesh_container_to_vtk(MeshCont const&              meshes,
                                 std::filesystem::path const& path,
                                 std::string const&           title) {
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
auto write_vtk(std::vector<triangular_mesh<Real, 3>> const& meshes,
               std::string const&                           path,
               std::string const& title = "tatooine meshes") {
  detail::write_mesh_container_to_vtk(meshes, path, title);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif

