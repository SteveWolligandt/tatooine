#ifndef TATOOINE_SIMPLE_TRI_MESH_H
#define TATOOINE_SIMPLE_TRI_MESH_H

#include <vector>

#include "vtk_legacy.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
class simple_tri_mesh {
 public:
  using vec_t                             = vec<Real, N>;
  using this_t                            = simple_tri_mesh<Real, N>;
  using pos_t                             = vec_t;

  //============================================================================
 protected:
  std::vector<vec_t>                 m_vertices;
  std::vector<std::array<size_t, 3>> m_faces;

 public:
  //============================================================================
  constexpr simple_tri_mesh() = default;

  //============================================================================
 public:
  simple_tri_mesh(const simple_tri_mesh& other) = default;
  simple_tri_mesh(simple_tri_mesh&& other)      = default;
  simple_tri_mesh& operator=(const simple_tri_mesh& other) = default;
  simple_tri_mesh& operator=(simple_tri_mesh&& other) = default;
  //----------------------------------------------------------------------------
  auto insert_vertex(const pos_t& v) {
    m_vertices.push_back(v);
    return m_vertices.size() - 1;
  }
  //----------------------------------------------------------------------------
  auto insert_face(size_t v0, size_t v1, size_t v2) {
    m_faces.push_back(std::array{v0, v1, v2});
    return m_faces.size() - 1;
  }
  //----------------------------------------------------------------------------
  auto insert_face(const std::array<size_t, 3>& f) {
    m_faces.push_back(f);
    return m_faces.size() - 1;
  }
  //----------------------------------------------------------------------------
  void clear() {
    m_vertices.clear();
    m_faces.clear();
  }
  //----------------------------------------------------------------------------
  const auto& faces() const { return m_faces; }
  const auto& vertices() const { return m_vertices; }
  //----------------------------------------------------------------------------
  template <size_t _N = N, typename = std::enable_if_t<_N == 2 || _N == 3>>
  void write_obj(const std::string& path) {
    std::ofstream fout(path);
    if (fout) {
      for (const auto& v : m_vertices) {
        if constexpr (N == 2) {
          fout << "v " << v(0) << ' ' << v(1) << " 0\n";
        } else if constexpr (N == 3) {
          fout << "v " << v(0) << ' ' << v(1) << " " << v(2) << '\n';
        }
      }
      for (const auto& f : m_faces) {
        fout << "f " << f[0] + 1 << ' ' << f[1] + 1 << ' ' << f[2] + 1
             << '\n';
      }
    }
  }
  //----------------------------------------------------------------------------
  template <size_t _N = N, std::enable_if_t<_N == 3, bool> = true>
  void write_vtk(const std::string& path,
                 const std::string& title = "tatooine simple_tri_mesh") const {
    vtk::legacy_file_writer writer(path, vtk::POLYDATA);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      writer.write_points(m_vertices);
      writer.write_polygons(m_faces);
      writer.close();
    }
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif

