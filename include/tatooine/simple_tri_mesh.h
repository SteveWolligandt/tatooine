#ifndef TATOOINE_SIMPLE_TRI_MESH_H
#define TATOOINE_SIMPLE_TRI_MESH_H
//==============================================================================
#include <vector>

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
      m_face_properties.insert(std::pair{key, fprop.clone()});
    }
  }
  simple_tri_mesh& operator=(const simple_tri_mesh& other) {
    parent_t::operator=(other);
    m_faces    = other.m_faces;
    for (const auto& [key, fprop] : other.m_face_properties) {
      m_face_properties.insert(std::pair{key, fprop.clone()});
    }
  }
  simple_tri_mesh(simple_tri_mesh&& other) = default;
  simple_tri_mesh& operator=(simple_tri_mesh&& other) = default;
  //============================================================================
  const auto& operator[](face f) const {return m_faces[f.i];}
  auto& operator[](face f) {return m_faces[f.i];}
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
      writer.close();
      return true;
    } else {
      return false;
    }
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

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif

