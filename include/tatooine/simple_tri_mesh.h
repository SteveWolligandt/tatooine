#ifndef TATOOINE_SIMPLE_TRI_MESH_H
#define TATOOINE_SIMPLE_TRI_MESH_H
//==============================================================================
#include <vector>
#include <boost/range/algorithm/copy.hpp>

#include "property.h"
#include "vtk_legacy.h"
#include "pointset.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
class simple_tri_mesh : public pointset<Real, N>{
 public:
  using this_t   = simple_tri_mesh<Real, N>;
  using parent_t = pointset<Real, N>;
  using typename parent_t::vertex;
  using parent_t::at;
  using parent_t::operator[];
  using parent_t::is_valid;
  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  //----------------------------------------------------------------------------
  struct face : handle {
    using handle::handle;
    constexpr bool operator==(face other) const { return this->i == other.i; }
    constexpr bool operator!=(face other) const { return this->i != other.i; }
    constexpr bool operator<(face other) const { return this->i < other.i; }
    static constexpr auto invalid() { return face{handle::invalid_idx}; }
  };
  //----------------------------------------------------------------------------
  struct face_iterator
      : boost::iterator_facade<face_iterator, face,
                               boost::bidirectional_traversal_tag, face> {
    face_iterator(face _f, const simple_tri_mesh* mesh) : f{_f}, m_mesh{mesh} {}
    face_iterator(const face_iterator& other) : f{other.f}, m_mesh{other.m_mesh} {}

   private:
    face                   f;
    const simple_tri_mesh* m_mesh;

    friend class boost::iterator_core_access;

    void increment() {
      do
        ++f;
      while (!m_mesh->is_valid(f));
    }
    void decrement() {
      do
        --f;
      while (!m_mesh->is_valid(f));
    }

    auto equal(const face_iterator& other) const { return f == other.f; }
    auto dereference() const { return f; }
  };
  //----------------------------------------------------------------------------
  struct face_container {
    using iterator       = face_iterator;
    using const_iterator = face_iterator;

    const simple_tri_mesh* m_mesh;

    auto begin() const {
      face_iterator vi{face{0}, m_mesh};
      if (!m_mesh->is_valid(*vi)) { ++vi; }
      return vi;
    }

    auto end() const {
      return face_iterator{face{m_mesh->m_faces.size()}, m_mesh};
    }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  using face_property_t = vector_property_impl<face, T>;
  //============================================================================
 protected:
  std::vector<std::array<vertex, 3>> m_faces;
  std::vector<face>                  m_invalid_faces;

  //============================================================================
 private:
  std::map<std::string, std::unique_ptr<vector_property<face>>>
      m_face_properties;
 public:
  //============================================================================
  constexpr simple_tri_mesh() = default;
  //============================================================================
 public:
  simple_tri_mesh(const simple_tri_mesh& other) : parent_t{other}, m_faces{other.m_faces} {
    for (const auto& [key, fprop] : other.m_face_properties) {
      m_face_properties.insert(std::pair{key, fprop->clone()});
    }
  }
  simple_tri_mesh& operator=(const simple_tri_mesh& other) {
    parent_t::operator=(other);
    m_faces    = other.m_faces;
    for (const auto& [key, fprop] : other.m_face_properties) {
      m_face_properties.insert(std::pair{key, fprop->clone()});
    }
  }
  simple_tri_mesh(simple_tri_mesh&& other) = default;
  simple_tri_mesh& operator=(simple_tri_mesh&& other) = default;
  //----------------------------------------------------------------------------
  simple_tri_mesh(const std::string& file) {
    read(file);
  }
  //============================================================================
  const auto& operator[](face f) const { return m_faces[f.i]; }
  auto&       operator[](face f) { return m_faces[f.i]; }
  //----------------------------------------------------------------------------
  const auto& at(face f) const {return m_faces[f.i];}
  auto& at(face f) {return m_faces[f.i];}
  //----------------------------------------------------------------------------
  auto insert_face(vertex v0, vertex v1, vertex v2) {
    m_faces.push_back(std::array{v0, v1, v2});
    for (auto& [key, prop] : m_face_properties) { prop->push_back(); }
    return face{m_faces.size() - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_face(size_t v0, size_t v1, size_t v2) {
    return insert_face(vertex{v0}, vertex{v1}, vertex{v2});
  }
  //----------------------------------------------------------------------------
  auto insert_face(const std::array<vertex, 3>& f) {
    m_faces.push_back(f);
    for (auto& [key, prop] : m_face_properties) { prop->push_back(); }
    return face{m_faces.size() - 1};
  }
  //----------------------------------------------------------------------------
  void clear() {
    parent_t::clear();
    m_faces.clear();
  }
  //----------------------------------------------------------------------------
  auto faces() const { return face_container{this}; }
  //----------------------------------------------------------------------------
  auto num_faces() const { return m_faces.size(); }
  //----------------------------------------------------------------------------
  template <size_t _N = N, typename = std::enable_if_t<_N == 2 || _N == 3>>
  void write_obj(const std::string& path) {
    std::ofstream fout(path);
    if (fout) {
      for (auto v : this->vertices()) {
        if constexpr (N == 2) {
          fout << "v " << at(v)(0) << ' ' << at(v)(1) << " 0\n";
        } else if constexpr (N == 3) {
          fout << "v " << at(v)(0) << ' ' << at(v)(1) << " " << at(v)(2) << '\n';
        }
      }
      for (const auto& f : m_faces) {
        fout << "f " << f[0] + 1 << ' ' << f[1] + 1 << ' ' << f[2] + 1 << '\n';
      }
    }
  }
  //----------------------------------------------------------------------------
  template <size_t _N = N, std::enable_if_t<_N == 2 || _N == 3, bool> = true>
  bool write_vtk(const std::string& path,
                 const std::string& title = "tatooine simple_tri_mesh") const {
    using boost::copy;
    using boost::adaptors::transformed;
    vtk::legacy_file_writer writer(path, vtk::POLYDATA);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      if constexpr (N == 2) {
        auto three_dims = [](const vec<Real, 2>& v2) {
          return vec<Real, 3>{v2(0), v2(1), 0};
        };
        std::vector<vec<Real, 3>> v3s(this->m_vertices.size());
        auto three_dimensional = transformed(three_dims);
        copy(this->m_vertices | three_dimensional, begin(v3s));
        writer.write_points(v3s);

      } else if constexpr (N == 3) {
        writer.write_points(this->m_vertices);
      }

      std::vector<std::vector<size_t>> polygons;
      polygons.reserve(num_faces());
      for (const auto& face : m_faces) {
        polygons.push_back(std::vector{face[0].i, face[1].i, face[2].i});
      }
      writer.write_polygons(polygons);

      // write vertex data
      writer.write_point_data(this->num_vertices());
      for (const auto& [name, prop] : this->m_vertex_properties) {
        if (prop->type() == typeid(vec<Real, 4>)) {
        } else if (prop->type() == typeid(vec<Real, 3>)) {
        } else if (prop->type() == typeid(vec<Real, 2>)) {
          const auto& casted_prop =
              *dynamic_cast<const vertex_property_t<vec<Real, 2>>*>(prop.get());
          writer.write_scalars(name, casted_prop.container());
        } else if (prop->type() == typeid(Real)) {
          const auto& casted_prop =
              *dynamic_cast<const vertex_property_t<Real>*>(prop.get());
          writer.write_scalars(name, casted_prop.container());
        }
      }

      writer.close();
      return true;
    } else {
      return false;
    }
  }
  //----------------------------------------------------------------------------
  void read(const std::string& path) {
    auto ext = path.substr(path.find_last_of(".") + 1);
    if constexpr (N == 2 || N == 3) {
      if (ext == "vtk") { read_vtk(path); }
    }
  }
  //----------------------------------------------------------------------------
  template <size_t _N = N, std::enable_if_t<_N == 2 || _N == 3, bool> = true>
  void read_vtk(const std::string& path) {
    struct listener_t : vtk::legacy_file_listener {
      simple_tri_mesh& mesh;
      listener_t(simple_tri_mesh& _mesh) : mesh(_mesh) {}
      void on_dataset_type(vtk::DatasetType t) override {
        if (t != vtk::POLYDATA) {
          throw std::runtime_error{
              "[simple_tri_mesh] need polydata when reading vtk legacy"};
        }
      }

      void on_points(const std::vector<std::array<float, 3>>& ps) override {
        for (auto& p : ps) { mesh.insert_vertex(p[0], p[1]); }
      }
      void on_points(const std::vector<std::array<double, 3>>& ps) override {
        for (auto& p : ps) { mesh.insert_vertex(p[0], p[1]); }
      }
      void on_polygons(const std::vector<int>& ps) override {
        for (size_t i = 0; i < ps.size();) {
          const auto& size = ps[i++];
          if (size == 3) {
            mesh.insert_face(size_t(ps[i]), size_t(ps[i + 1]), size_t(ps[i + 2]));
          }
          i += size;
        }
      }
      //void on_scalars(const std::string& data_name,
      //                const std::string& [>lookup_table_name<],
      //                size_t num_comps, const std::vector<float>& scalars,
      //                vtk::ReaderData) override {}
      void on_scalars(const std::string& data_name,
                      const std::string& /*lookup_table_name*/,
                      size_t num_comps, const std::vector<double>& scalars,
                      vtk::ReaderData data) override {
        if (data == vtk::POINT_DATA) {
          if (num_comps == 1) {
            auto& prop = mesh.template add_vertex_property<double>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) { prop[i] = scalars[i]; }
          } else if (num_comps == 2) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 2>>(data_name);

            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[i][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 3) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 3>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[i][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 4) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 4>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[i][j] = scalars[i * num_comps + j];
              }
            }
          }
        }
      }
    } listener{*this};
    vtk::legacy_file f{path};
    f.add_listener(listener);
    f.read();
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto& face_property(const std::string& name) {
    auto& prop = m_face_properties.at(name);
    assert(typeid(T) == prop.type_info());
    return *dynamic_cast<face_property_t<T>*>(prop);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  const auto& face_property(const std::string& name) const {
    const auto& prop = m_face_properties.at(name);
    assert(typeid(T) == prop.type_info());
    return *dynamic_cast<face_property_t<T>*>(prop);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto& add_face_property(const std::string& name, const T& value = T{}) {
    auto [it, suc] = m_face_properties.insert(
        std::pair{name, std::make_unique<face_property_t<T>>(value)});
    auto fprop = dynamic_cast<face_property_t<T>*>(it->second.get());
    fprop->resize(m_faces.size());
    return *fprop;
  }
  //----------------------------------------------------------------------------
  constexpr bool is_valid(face f) const {
    return boost::find(m_invalid_faces, f) == end(m_invalid_faces);
  }
};
namespace detail {
template <typename MeshCont>
void write_mesh_container_to_vtk(const MeshCont& meshes, const std::string& path,
                                 const std::string& title) {
  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
  if (writer.is_open()) {
    size_t num_pts = 0;
    size_t cur_first = 0;
    for (auto const& m : meshes) { num_pts += m.num_vertices(); }
    std::vector<std::array<typename MeshCont::value_type::real_t, 3>> points;
    std::vector<std::vector<size_t>>                                  faces;
    points.reserve(num_pts);
    faces.reserve(meshes.size());

    for (const auto& m : meshes) {
      // add points
      for (const auto& v : m.vertices()) {
        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
      }

      // add faces
      for (auto f : m.faces()) {
        faces.emplace_back();
        faces.back().push_back(cur_first + m[f][0].i);
        faces.back().push_back(cur_first + m[f][1].i);
        faces.back().push_back(cur_first + m[f][2].i);
      }
      cur_first += m.num_vertices();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_polygons(faces);
    //writer.write_point_data(num_pts);
    writer.close();
  }
}
}  // namespace detail
//==============================================================================
template <typename Real>
void write_vtk(const std::vector<simple_tri_mesh<Real, 3>>& meshes, const std::string& path,
               const std::string& title = "tatooine meshes") {
  detail::write_mesh_container_to_vtk(meshes, path, title);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif

