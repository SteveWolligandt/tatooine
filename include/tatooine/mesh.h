#ifndef TATOOINE_MESH_H
#define TATOOINE_MESH_H

#include <tatooine/edgeset.h>
#include <boost/range/algorithm.hpp>
#include <list>
#include <set>
#include "utility.h"
#include "vtk_legacy.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
class mesh : public edgeset<Real, N> {
 public:
  using this_type   = mesh<Real, N>;
  using parent_type = edgeset<Real, N>;

  using typename parent_type::edge;
  using typename parent_type::handle;
  using typename parent_type::order_independent_edge_compare;
  using typename parent_type::pos_type;
  using typename parent_type::vertex;

  using parent_type::at;
  using parent_type::operator[];
  using parent_type::edges;
  using parent_type::is_valid;
  using parent_type::num_edges;
  using parent_type::num_vertices;
  using parent_type::remove;
  using parent_type::vertices;
  using parent_type::insert_edge;

  //============================================================================
 public:
  struct face : handle {
    face() = default;
    face(size_t i) : handle{i} {}
    face(const face&)            = default;
    face(face&&)                 = default;
    face& operator=(const face&) = default;
    face& operator=(face&&)      = default;
    bool operator==(const face& other) const { return this->i == other.i; }
    bool operator!=(const face& other) const { return this->i != other.i; }
    bool operator<(const face& other) const { return this->i < other.i; }
    static constexpr auto invalid() { return face{handle::invalid_idx}; }
  };

  //============================================================================
  struct face_iterator
      : public boost::iterator_facade<
            face_iterator, face, boost::bidirectional_traversal_tag, face> {
    face_iterator(face _f, const mesh* _m) : f{_f}, m{_m} {}
    face_iterator(const face_iterator& other) : f{other.f}, m{other.m} {}

   private:
    face        f;
    const mesh* m;
    friend class boost::iterator_core_access;

    void increment() {
      do
        ++f.i;
      while (!m->is_valid(f));
    }
    void decrement() {
      do
        --f.i;
      while (!m->is_valid(f));
    }

    bool equal(const face_iterator& other) const { return f.i == other.f.i; }
    auto dereference() const { return f; }
  };

  //============================================================================
  struct face_container {
    using iterator       = face_iterator;
    using const_iterator = face_iterator;
    const mesh* m_mesh;

    auto begin() const {
      face_iterator fi{face{0}, m_mesh};
      if (!m_mesh->is_valid(*fi)) ++fi;
      return fi;
    }
    auto end() const { return face_iterator{face{m_mesh->m_faces.size()}, m_mesh}; }
  };

  //============================================================================
  template <typename T>
  using vertex_prop = typename parent_type::template vertex_prop<T>;

  //----------------------------------------------------------------------------
  template <typename T>
  using edge_prop = typename parent_type::template edge_prop<T>;

  //----------------------------------------------------------------------------
  template <typename T>
  struct face_prop : public property_type<T> {
    using property_type<T>::property_type;
    auto&       at(face f) { return property_type<T>::at(f.i); }
    const auto& at(face f) const { return property_type<T>::at(f.i); }
    auto&       operator[](face f) { return property_type<T>::operator[](f.i); }
    const auto& operator[](face f) const {
      return property_type<T>::operator[](f.i);
    }
    std::unique_ptr<property> clone() const override {
      return std::unique_ptr<face_prop<T>>(new face_prop<T>{*this});
    }
  };

  //============================================================================
 protected:
  std::vector<std::vector<vertex>>                 m_faces;
  std::vector<face>                                m_invalid_faces;
  std::map<std::string, std::unique_ptr<property>> m_face_properties;

  vertex_prop<std::vector<face>>* m_faces_of_vertices = nullptr;
  edge_prop<std::vector<face>>*   m_faces_of_edges    = nullptr;

  face_prop<std::vector<edge>>* m_edges_of_faces = nullptr;

 public:
  //============================================================================
  constexpr mesh() { add_link_properties(); }

  //----------------------------------------------------------------------------
  constexpr mesh(std::initializer_list<pos_type>&& vertices)
      : parent_type(std::move(vertices)) {
    add_link_properties();
  }

  //----------------------------------------------------------------------------
#ifdef USE_TRIANGLE
  mesh(const triangle::io& io) : parent_type{io} {
    add_link_properties();
    for (int i = 0; i < io.numberoftriangles; ++i)
      insert_face(io.trianglelist[i * 3], io.trianglelist[i * 3 + 1],
                  io.trianglelist[i * 3 + 2]);
  }
#endif

  // mesh(const tetgenio& io) : parent_type{io} { add_link_properties(); }

  //============================================================================
 public:
  mesh(const mesh& other)
      : parent_type(other),
        m_faces(other.m_faces),
        m_invalid_faces(other.m_invalid_faces) {
    m_face_properties.clear();
    for (const auto& [name, prop] : other.m_face_properties)
      m_face_properties[name] = prop->clone();
    find_link_properties();
  }

  //----------------------------------------------------------------------------
  mesh(mesh&& other)
      : parent_type(std::move(other)),
        m_faces(std::move(other.m_faces)),
        m_invalid_faces(std::move(other.m_invalid_faces)),
        m_face_properties(std::move(other.m_face_properties)) {
    find_link_properties();
  }

  //----------------------------------------------------------------------------
  auto& operator=(const mesh& other) {
    parent_type::operator=(other);
    m_faces           = other.m_faces;
    m_invalid_faces   = other.m_invalid_faces;
    for (const auto& [name, prop] : other.m_face_properties)
      m_face_properties[name] = prop->clone();
    find_link_properties();
    return *this;
  }

  //----------------------------------------------------------------------------
  auto& operator=(mesh&& other) {
    parent_type::operator=(std::move(other));
    m_faces           = std::move(other.m_faces);
    m_invalid_faces   = std::move(other.m_invalid_faces);
    m_face_properties = std::move(other.m_face_properties);
    find_link_properties();
    return *this;
  }

  //============================================================================
 private:
  void add_link_properties() {
    m_faces_of_vertices = dynamic_cast<vertex_prop<std::vector<face>>*>(
        &this->template add_vertex_property<std::vector<face>>("v:faces"));
    m_faces_of_edges = dynamic_cast<edge_prop<std::vector<face>>*>(
        &this->template add_edge_property<std::vector<face>>("e:faces"));

    m_edges_of_faces = dynamic_cast<face_prop<std::vector<edge>>*>(
        &this->template add_face_property<std::vector<edge>>("f:edges"));
  }

  //----------------------------------------------------------------------------
  void find_link_properties() {
    m_faces_of_vertices = dynamic_cast<vertex_prop<std::vector<face>>*>(
        &this->template vertex_property<std::vector<face>>("v:faces"));
    m_faces_of_edges = dynamic_cast<edge_prop<std::vector<face>>*>(
        &this->template edge_property<std::vector<face>>("e:faces"));
    m_edges_of_faces = dynamic_cast<face_prop<std::vector<edge>>*>(
        &face_property<std::vector<edge>>("f:edges"));
  }

 public:
  //============================================================================
  constexpr auto&       at(face f) { return m_faces[f.i]; }
  constexpr const auto& at(face f) const { return m_faces[f.i]; }

  //----------------------------------------------------------------------------
  constexpr auto&       operator[](face f) { return at(f); }
  constexpr const auto& operator[](face f) const { return at(f); }

  //----------------------------------------------------------------------------
  template <typename... Vs,
            enable_if<(std::is_same_v<vertex, Vs> && ...)> = true>
  constexpr auto insert_face(Vs... vs) {
    return insert_face(std::vector<vertex>{vs...});
  }

  //----------------------------------------------------------------------------
  constexpr auto insert_face(std::vector<vertex> new_face) {
    // rotate vertex indices so that first vertex has smallest index
    boost::rotate(new_face, boost::min_element(new_face));
    rotation_independent_face_equal eq;
    for (auto f : faces())
      if (eq(at(f), new_face)) return f;

    face f{m_faces.size()};
    m_faces.push_back(new_face);
    for (auto& [key, prop] : m_face_properties) prop->push_back();

    auto inserted_edges = insert_edges(f);

    for (auto v : new_face) { faces(v).push_back(f); }

    for (auto e : inserted_edges) {
      faces(e).push_back(f);
      edges(f).push_back(e);
    }

    return f;
  }

  //----------------------------------------------------------------------------
  void remove(vertex v) {
    using namespace boost;
    parent_type::remove(v);
    for (auto f : faces(v))
      if (find(m_invalid_faces, f) == m_invalid_faces.end()) remove(f);
  }

  //----------------------------------------------------------------------------
  void remove(edge e, bool remove_orphaned_vertices = true) {
    using namespace boost;
    parent_type::remove(e, remove_orphaned_vertices);
    for (auto f : faces(e))
      if (find(m_invalid_faces, f) == m_invalid_faces.end()) remove(f);
  }

  //----------------------------------------------------------------------------
  constexpr void remove(face f, bool remove_orphaned_vertices = true,
                        bool remove_orphaned_edges = true) {
    using namespace boost;
    if (is_valid(f)) {
      if (find(m_invalid_faces, f) == m_invalid_faces.end())
        m_invalid_faces.push_back(f);
      if (remove_orphaned_vertices)
        for (auto v : vertices(f))
          if (num_faces(v) <= 1) remove(v);

      if (remove_orphaned_edges)
        for (auto e : edges(f))
          if (num_faces(e) <= 1) remove(e, remove_orphaned_vertices);

      // remove face link from vertices
      for (auto v : vertices(f)) faces(v).erase(find(faces(v), f));

      // remove face link from edges
      for (auto e : edges(f)) faces(e).erase(find(faces(e), f));
      edges(f).clear();
    }
  }

#ifdef USE_TRIANGLE
  //----------------------------------------------------------------------------
  template <size_t _N = N, enable_if<_N == 2> = true>
  void triangulate_face(const std::vector<vertex>& polygon) {
    if (polygon.size() == 3)
      insert_face(polygon[0], polygon[1], polygon[2]);
    else if (polygon.size() == 4) {
      insert_face(polygon[0], polygon[1], polygon[2]);
      insert_face(polygon[0], polygon[2], polygon[3]);
    } else {
      triangle::api api;
      api.behaviour().firstnumber = 0;
      api.behaviour().poly        = true;
      api.behaviour().usesegments = true;
      auto contour                = to_triangle_io(polygon);
      contour.numberofsegments    = polygon.size();
      contour.segmentlist         = new int[contour.numberofsegments * 2];
      for (size_t i = 0; i < polygon.size(); ++i) {
        contour.segmentlist[i * 2]     = i;
        contour.segmentlist[i * 2 + 1] = (i + 1) % polygon.size();
      }

      api.mesh_create(contour);
      auto triangulated_contour = api.mesh_copy();
      // assert(contour.numberofpoints == triangulated_contour.numberofpoints);
      for (int i = 0; i < contour.numberofpoints * 2; ++i)
        assert(contour.pointlist[i] == triangulated_contour.pointlist[i]);
      for (int i = 0; i < triangulated_contour.numberoftriangles; ++i)
        if (triangulated_contour.trianglelist[i * 3] < polygon.size() &&
            triangulated_contour.trianglelist[i * 3 + 1] < polygon.size() &&
            triangulated_contour.trianglelist[i * 3 + 2] < polygon.size())
          insert_face(polygon[triangulated_contour.trianglelist[i * 3]],
                      polygon[triangulated_contour.trianglelist[i * 3 + 1]],
                      polygon[triangulated_contour.trianglelist[i * 3 + 2]]);
    }
  }

  //----------------------------------------------------------------------------
  template <size_t _N = N, enable_if<_N == 2> = true>
  inline auto triangulate_face(face f) {
    std::vector<vertex> polygon;
    polygon.reserve(num_vertices(f));
    for (auto v : vertices(f)) polygon.push_back(v);
    triangulate_face(polygon);
  }
#endif

  //----------------------------------------------------------------------------
  //! tidies up invalid vertices, edges and faces
  void tidy_up() {
    using namespace boost;
    for (auto invalid_f : m_invalid_faces) {
      // decrease face-index of vertices whose indices are greater than an
      // invalid face index
      for (auto& faces : *m_faces_of_vertices)
        for (auto& f : faces)
          if (f.i > invalid_f.i) --f.i;

      // decrease face-index of edges whose indices are greater than an invalid
      // face-index
      for (auto& faces : *m_faces_of_edges)
        for (auto& f : faces)
          if (f.i > invalid_f.i) --f.i;
    }
    // decrease edge-index of faces whose indices are greater than an invalid
    // edge-index
    for (auto invalid_e : this->m_invalid_edges)
      for (auto& edges : *m_edges_of_faces)
        for (auto& e : edges)
          if (e.i >= invalid_e.i) --e.i;

    // reindex face's vertex indices
    for (const auto v : this->m_invalid_points)
      for (auto f : faces())
        for (auto& f_v : at(f))
          if (f_v.i > v.i) --f_v.i;

    // erase actual faces
    for (const auto f : m_invalid_faces) {
      // reindex deleted faces indices;
      for (auto& f_to_reindex : m_invalid_faces)
        if (f_to_reindex.i > f.i) --f_to_reindex.i;

      m_faces.erase(m_faces.begin() + f.i);
      for (const auto& [key, prop] : m_face_properties) { prop->erase(f.i); }
    }
    m_invalid_faces.clear();

    // tidy up vertices
    parent_type::tidy_up();
  }

  //----------------------------------------------------------------------------
  constexpr bool is_valid(face f) const {
    return boost::find(m_invalid_faces, f) == m_invalid_faces.end();
  }

  //----------------------------------------------------------------------------
  constexpr auto insert_edges(face f) {
    const auto&       vertices = at(f);
    std::vector<edge> edges;
    edges.reserve(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
      edges.push_back(
          insert_edge(vertices[i], vertices[(i + 1) % vertices.size()]));
    return edges;
  }

  //----------------------------------------------------------------------------
  constexpr void clear_faces() {
    m_faces.clear();
    m_faces.shrink_to_fit();
    m_invalid_faces.clear();
    m_invalid_faces.shrink_to_fit();
  }

  //----------------------------------------------------------------------------
  void clear() {
    parent_type::clear();
    clear_faces();
  }

  //----------------------------------------------------------------------------
  constexpr auto num_faces(vertex v) const { return faces(v).size(); }
  constexpr auto num_faces(edge e) const { return faces(e).size(); }
  constexpr auto num_faces() const {
    return m_faces.size() - m_invalid_faces.size();
  }
  constexpr auto num_edges(face f) const { return edges(f).size(); }

  //----------------------------------------------------------------------------
  constexpr auto faces() const { return face_container{this}; }
  auto&          faces(vertex v) { return m_faces_of_vertices->at(v); }
  const auto&    faces(vertex v) const { return m_faces_of_vertices->at(v); }
  auto&          faces(edge e) { return m_faces_of_edges->at(e); }
  const auto&    faces(edge e) const { return m_faces_of_edges->at(e); }
  auto&          edges(face f) { return m_edges_of_faces->at(f); }
  const auto&    edges(face f) const { return m_edges_of_faces->at(f); }
  auto&          vertices(face f) { return at(f); }
  const auto&    vertices(face f) const { return at(f); }
  auto           neighbor_faces(face f) const {
    std::vector<face> neighbors;
    for (auto e : edges(f)) {
      for (auto nf : faces(e)) {
        if (nf != f) { neighbors.push_back(nf); }
      }
    }
    return neighbors;
  }

  //----------------------------------------------------------------------------
  constexpr bool has_vertex(face f, vertex v) const {
    return boost::find(at(f), v) != end(at(f));
  }

  //----------------------------------------------------------------------------
  constexpr bool face_has_edge(face f, edge e) const {
    return has_vertex(f, this->at(e)[0]) &&
           has_vertex(f, this->at(e)[1]);
  }

  //----------------------------------------------------------------------------
  constexpr auto num_vertices(face f) const { return at(f).size(); }

#ifdef USE_TRIANGLE
  //----------------------------------------------------------------------------
  auto to_triangle_io() const {
    auto io = parent_type::to_triangle_io();

    // Define input points
    io.numberoftriangles = num_faces();
    io.numberofcorners   = 3;

    // copy faces
    io.trianglelist = new int[io.numberoftriangles * io.numberofcorners];
    size_t i        = 0;
    for (auto f : faces()) {
      const auto& face = at(f);
      for (unsigned int j = 0; j < 3; ++j) io.pointlist[i + j] = face[i].i;
      i += 3;
    }
    return io;
  }

  //----------------------------------------------------------------------------
  //! using specified vertices of mesh.
  //! automatically searches for faces using of of the vertices
  auto to_triangle_io(const std::vector<vertex>& vertices) const {
    auto io = parent_type::to_triangle_io(vertices);

    std::set<face> fs;
    for (auto v : vertices)
      for (auto f : faces(v)) fs.insert(f);

    // Define input points
    io.numberoftriangles = fs.size();
    io.numberofcorners   = 3;

    // copy faces
    io.trianglelist = new int[io.numberoftriangles * io.numberofcorners];
    size_t i        = 0;
    for (auto f : fs) {
      const auto& face = at(f);
      // faces are not indexed by global pointlist
      for (unsigned int j = 0; j < 3; ++j) io.trianglelist[i + j] = face[j].i;
      i += 3;
    }

    // reindex points to local list of vertices
    for (int i = 0; i < io.numberoftriangleattributes * 3; ++i)
      for (size_t j = 0; j < vertices.size(); ++j)
        if (io.trianglelist[i] == int(vertices[j].i)) {
          io.trianglelist[i] = j;
          break;
        }
    return io;
  }
#endif

  //----------------------------------------------------------------------------
  // void to_tetgen_io(tetgenio& in) const {
  //   parent_type::to_tetgen_io(in);
  //
  //   // Define input points
  //   in.numberoffacets = num_faces();
  //
  //   // copy faces
  //   in.facetlist = new tetgen::io::facet[in.numberoffacets];
  //   size_t i     = 0;
  //   for (auto f : faces()) {
  //     auto& facet = in.facetlist[i++];
  //     facet.numberofholes    = 0;
  //     facet.holelist         = nullptr;
  //     facet.numberofpolygons = 1;
  //     facet.polygonlist      = new tetgen::io::polygon[1];
  //     auto& poly = facet.polygonlist[0];
  //     poly.numberofvertices = num_vertices(f);
  //     poly.vertexlist = new int[poly.numberofvertices];
  //     size_t j = 0;
  //     for (auto v : at(f)) poly.vertexlist[j++] = v.i;
  //   }
  // }

  //----------------------------------------------------------------------------
  // void to_tetgen_io(tetgenio& in, const std::vector<face>& fs) const {
  //   std::map<vertex, int> vertex_mapping;
  //   in.numberofpoints = 0;
  //   for (auto f : fs) {
  //     for (auto v : vertices(f)) {
  //       if (vertex_mapping.find(v) == end(vertex_mapping)) {
  //         vertex_mapping[v] = in.numberofpoints++;
  //       }
  //     }
  //   }
  //   in.pointlist       = new tetgen::Real[in.numberofpoints * 3];
  //   in.pointmarkerlist = new int[in.numberofpoints];
  //   in.numberofpointattributes = 1;
  //   in.pointattributelist =
  //       new tetgen::real_type[in.numberofpoints * in.numberofpointattributes];
  //   for (const auto& [v, i] : vertex_mapping) {
  //     in.pointlist[i * 3]     = at(v)(0);
  //     in.pointlist[i * 3 + 1] = at(v)(1);
  //     in.pointlist[i * 3 + 2] = at(v)(2);
  //     in.pointmarkerlist[i] = -1;
  //     in.pointattributelist[i] = v.i;
  //   }
  //
  //   // Define input points
  //   in.numberoffacets = fs.size();
  //
  //   // copy faces
  //   in.facetlist = new tetgen::io::facet[in.numberoffacets];
  //   size_t i     = 0;
  //   for (auto f : fs) {
  //     auto& facet = in.facetlist[i++];
  //     facet.numberofholes    = 0;
  //     facet.holelist         = nullptr;
  //     facet.numberofpolygons = 1;
  //     facet.polygonlist      = new tetgen::io::polygon[1];
  //     auto& poly = facet.polygonlist[0];
  //     poly.numberofvertices = num_vertices(f);
  //     poly.vertexlist = new int[poly.numberofvertices];
  //     size_t j = 0;
  //     for (auto v : at(f)) {
  //       assert(vertex_mapping.find(v) != end(vertex_mapping));
  //       poly.vertexlist[j++] = vertex_mapping[v];
  //     }
  //   }
  // }

  //----------------------------------------------------------------------------
  template <typename T>
  auto& face_property(const std::string& name) {
    return *dynamic_cast<face_prop<T>*>(m_face_properties.at(name).get());
  }

  //----------------------------------------------------------------------------
  template <typename T>
  const auto& face_property(std::string const& name) const {
    return *dynamic_cast<face_prop<T>*>(m_face_properties.at(name).get());
  }

  //----------------------------------------------------------------------------
  template <typename T>
  auto& add_face_property(const std::string& name, const T& value = T{}) {
    auto [it, suc] = m_face_properties.insert(
        std::pair{name, std::make_unique<face_prop<T>>(value)});
    auto prop = dynamic_cast<face_prop<T>*>(it->second.get());
    prop->resize(m_faces.size());
    return *prop;
  }

  //----------------------------------------------------------------------------
  void write(const std::string& path) {
    auto ext = path.substr(path.find_last_of(".") + 1);
    if constexpr (N == 2 || N == 3) {
      if (ext == "vtk") {
        write_vtk(path);
      } else if (ext == "obj") {
        write_obj(path);
      }
    }
  }

  //----------------------------------------------------------------------------
  template <size_t _N = N, enable_if<_N == 2 || _N == 3> = true>
  void write_obj(const std::string& path) {
    std::ofstream fout(path);
    if (fout) {
      for (auto v : vertices())
        if constexpr (N == 2)
          fout << "v " << at(v)(0) << ' ' << at(v)(1) << " 0\n";
        else if constexpr (N == 3)
          fout << "v " << at(v)(0) << ' ' << at(v)(1) << " " << at(v)(2)
               << '\n';
      for (auto f : faces())
        fout << "f " << at(f)[0].i + 1 << ' ' << at(f)[1].i + 1 << ' '
             << at(f)[2].i + 1 << '\n';
      fout.close();
    }
  }

  //----------------------------------------------------------------------------
  template <size_t _N = N, enable_if<_N == 2 || _N == 3> = true>
  void write_vtk(const std::string& path,
                 const std::string& title = "tatooine mesh") const {
    vtk::legacy_file_writer writer(path, vtk::POLYDATA);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();

      // write points
      std::vector<std::array<Real, 3>> points;
      points.reserve(this->m_points.size());
      for (const auto& p : this->m_points) {
        if constexpr (N == 3) {
          points.push_back({p(0), p(1), p(2)});
        } else {
          points.push_back({p(0), p(1), 0});
        }
      }
      writer.write_points(points);

      // write faces
      std::vector<std::vector<size_t>> polygons;
      polygons.reserve(m_faces.size());
      for (const auto& f : faces()) {
        polygons.emplace_back();
        for (const auto v : at(f)) polygons.back().push_back(v.i);
      }
      writer.write_polygons(polygons);

      // write point data
      // TODO uncomment if vertices have edge and face properties
      if (this->m_vertex_properties.size() > 2) {
        writer.write_point_data(this->m_points.size());
        for (const auto& [name, prop] : this->m_vertex_properties) {
          if (name != "v:edges" && name != "v:faces") {
            std::vector<std::vector<Real>> data;
            data.reserve(this->m_points.size());

            if (prop->type() == typeid(vec<Real, 4>)) {
              for (const auto& v4 :
                   *dynamic_cast<const vertex_prop<vec<Real, 4>>*>(
                       prop.get()))
                data.push_back({v4(0), v4(1), v4(2), v4(3)});

            } else if (prop->type() == typeid(vec<Real, 3>)) {
              for (const auto& v3 :
                   *dynamic_cast<const vertex_prop<vec<Real, 3>>*>(
                       prop.get()))
                data.push_back({v3(0), v3(1), v3(2)});

            } else if (prop->type() == typeid(vec<Real, 2>)) {
              const auto& casted_prop =
                  *dynamic_cast<const vertex_prop<vec<Real, 2>>*>(prop.get());
              for (const auto& v2 : casted_prop) data.push_back({v2(0), v2(1)});

            } else if (prop->type() == typeid(Real)) {
              for (const auto& scalar :
                   *dynamic_cast<const vertex_prop<Real>*>(prop.get()))
                data.push_back({scalar});
            }
            if (!data.empty()) writer.write_scalars(name, data);
          }
        }
      }

      // write cell data
      if (m_face_properties.size() > 1) {
        writer.write_cell_data(m_faces.size());
        for (const auto& [name, prop] : m_face_properties) {
          if (name != "f:edges") {
            if (prop->type() == typeid(vec<Real, 4>)) {
              std::vector<std::vector<Real>> data;
              data.reserve(m_faces.size());
              for (const auto& v4 :
                   *dynamic_cast<const face_prop<vec<Real, 4>>*>(prop.get()))
                data.push_back({v4(0), v4(1), v4(2), v4(3)});
              writer.write_scalars(name, data);

            } else if (prop->type() == typeid(vec<Real, 3>)) {
              std::vector<std::vector<Real>> data;
              data.reserve(m_faces.size());
              for (const auto& v3 :
                   *dynamic_cast<const face_prop<vec<Real, 3>>*>(prop.get()))
                data.push_back({v3(0), v3(1), v3(2)});
              writer.write_scalars(name, data);

            } else if (prop->type() == typeid(vec<Real, 2>)) {
              std::vector<std::vector<Real>> data;
              data.reserve(m_faces.size());
              for (const auto& v2 :
                   *dynamic_cast<const face_prop<vec<Real, 2>>*>(prop.get()))
                data.push_back({v2(0), v2(1)});
              writer.write_scalars(name, data);

            } else if (prop->type() == typeid(double)) {
              std::vector<std::vector<double>> data;
              data.reserve(m_faces.size());
              for (const auto& scalar :
                   *dynamic_cast<const face_prop<double>*>(prop.get()))
                data.push_back({scalar});
              writer.write_scalars(name, data);

            } else if (prop->type() == typeid(float)) {
              std::vector<std::vector<float>> data;
              data.reserve(m_faces.size());
              for (const auto& scalar :
                   *dynamic_cast<const face_prop<float>*>(prop.get()))
                data.push_back({scalar});
              writer.write_scalars(name, data);
            }
          }
        }
      }
      writer.close();
    }
  }

  //! checks if two faces are neighbors
  bool are_neighbors(face f0, face f1) {
    for (auto v0 : vertices(f0)) {
      for (auto v1 : vertices(f1)) {
        if (v0 == v1) { return true; }
      }
    }
    return false;
  }

  //----------------------------------------------------------------------------
  template <typename face_cont_t>
  auto adjacent_faces(const face_cont_t& faces) {
    using groups_t    = std::list<std::vector<face>>;
    using groups_it_t = typename groups_t::iterator;
    groups_t groups;
    for (auto f : faces) {
      std::vector<groups_it_t> insertions;
      for (auto groups_it = groups.begin(); groups_it != groups.end();
           ++groups_it)
        for (auto gf : *groups_it)
          if (are_neighbors(f, gf)) {
            insertions.push_back(groups_it);
            break;
          }

      // no group was found -> create new group
      if (insertions.empty()) groups.emplace_back().push_back(f);
      //  exactly one match -> just insert face
      else if (insertions.size() == 1)
        insertions.front()->push_back(f);

      // multiple matches -> merge groups and insert face in first match
      else {
        insertions.front()->push_back(f);
        for (size_t i = 1; i < insertions.size(); ++i) {
          boost::copy(*insertions[i], std::back_inserter(*insertions.front()));
          groups.erase(insertions[i]);
        }
      }
    }

    return groups;
  }

  //----------------------------------------------------------------------------
  //! returns list of edges representing borders of list of faces.
  //! there might be more than one coherent border edge loop. use
  //! split_border_edges() for splitting coherent loops
  template <typename face_cont_t>
  auto border_edges(const face_cont_t& faces) const {
    std::map<edge, size_t> edge_counts;
    for (auto f : faces)
      for (auto e : edges(f)) ++edge_counts[e];

    std::set<edge> es;
    for (const auto& [e, cnt] : edge_counts)
      if (cnt == 1) es.insert(e);

    return es;
  }

  //----------------------------------------------------------------------------
  //! searches coherent border edge loops
  template <typename edge_cont_t>
  auto split_border_edges(const edge_cont_t& edges) {
    std::vector<std::set<edge>> splitted_edges;
    bool                        inserted = false;
    for (auto e : edges) {
      inserted = false;
      for (auto& cont : splitted_edges)
        for (auto inserted_edge : cont) {
          if (at(inserted_edge)[0] == at(e)[0] ||
              at(inserted_edge)[0] == at(e)[1] ||
              at(inserted_edge)[1] == at(e)[0] ||
              at(inserted_edge)[1] == at(e)[1]) {
            cont.insert(e);
            inserted = true;
            break;
          }
          if (inserted) break;
        }
      if (!inserted) splitted_edges.emplace_back().insert(e);
    }
    return splitted_edges;
  }

  //----------------------------------------------------------------------------
  //! converts list of border edges to ordered list of vertices.
  //! returned polygon has arbitrary rotation (clock-wise or counter-clockwise).
  //! if edges is no actual loop this function never returns.
  template <typename edge_cont_t>
  auto border_edges_to_vertices(const edge_cont_t& edges) {
    std::vector<vertex> polygon;
    // insert first edge
    polygon.push_back(at(*edges.begin())[0]);
    polygon.push_back(at(*edges.begin())[1]);

    bool   searching = true;
    size_t i         = 0;
    while (searching) {
      for (auto e_it = next(begin(edges)); e_it != end(edges); ++e_it) {
        auto e = *e_it;
        i      = 2;
        if (at(e)[0] == polygon.back() && at(e)[1] != *prev(end(polygon), 2))
          i = 1;

        else if (at(e)[1] == polygon.back() &&
                 at(e)[0] != *prev(end(polygon), 2))
          i = 0;

        // next edge found
        if (i < 2) {
          // insert vertex and check if other vertex of edge is beginning of
          // polygon
          if (polygon.front() == at(e)[i])
            searching = false;
          else {
            polygon.push_back(at(e)[i]);
          }
          break;
        }
      }
    }
    return polygon;
  }

  //----------------------------------------------------------------------------
  static bool is_left(const pos_type& a, const pos_type& b, const pos_type& c) {
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) >= 0;
  }

  //----------------------------------------------------------------------------
  template <size_t _N = N, enable_if_t<_N == 2> = true>
  bool polygon_is_counter_clockwise(const std::vector<vertex>& polygon) {
    size_t left_turns  = 0;
    size_t right_turns = 0;
    for (size_t i_0 = 0; i_0 < polygon.size(); ++i_0) {
      auto i_1 = (i_0 + 1) % polygon.size();
      auto i_2 = (i_0 + 2) % polygon.size();
      if (is_left(at(polygon[i_0]), at(polygon[i_1]), at(polygon[i_2])))
        ++left_turns;
      else
        ++right_turns;
    }
    return left_turns > right_turns;
  }

  //----------------------------------------------------------------------------
  template <typename face_cont_t>
  auto border_polygons(const face_cont_t& faces,
                       bool               check_counterclockwise = true) {
    auto border_loops = split_border_edges(border_edges(faces));
    std::vector<std::vector<vertex>> polygons;
    for (const auto& loop : border_loops) {
      polygons.push_back(border_edges_to_vertices(loop));
      if (check_counterclockwise &&
          !polygon_is_counter_clockwise(polygons.back()))
        reverse(begin(polygons.back()), end(polygons.back()));
    }
    return polygons;
  }

  //============================================================================
  struct rotation_independent_face_equal {
    static bool same_rotation(const std::vector<vertex>& lhs,
                              const std::vector<vertex>& rhs) {
      auto lit   = begin(lhs);
      auto rit   = begin(rhs);
      bool equal = true;
      for (; lit != end(lhs); ++lit, ++rit) {
        if (*lit != *rit) {
          equal = false;
          break;
        }
      }
      return equal;
    }

    //--------------------------------------------------------------------------
    static bool different_rotation(const std::vector<vertex>& lhs,
                                   const std::vector<vertex>& rhs) {
      auto lit   = begin(lhs);
      auto rit   = begin(rhs);
      bool equal = true;
      for (; lit != end(lhs); ++lit, --rit) {
        if (*lit != *rit) {
          equal = false;
          break;
        }
        if (rit == begin(rhs)) rit = end(rhs);
      }
      return equal;
    }

    //--------------------------------------------------------------------------
    bool operator()(const std::vector<vertex>& lhs,
                    const std::vector<vertex>& rhs) const {
      if (lhs.size() != rhs.size()) return false;
      if (same_rotation(lhs, rhs)) return true;
      return different_rotation(lhs, rhs);
    }
  };
};

#ifdef USE_TRIANGLE
//------------------------------------------------------------------------------
template <typename Real>
inline auto delaunay(const pointset<2, Real>& ps) {
  triangle::api api;
  api.behaviour().firstnumber = 0;
  api.mesh_create(ps.to_triangle_io());
  return mesh<2, Real>{api.mesh_copy()};
}

//------------------------------------------------------------------------------
template <typename Real>
inline auto delaunay(
    const pointset<2, Real>&                               ps,
    const std::vector<typename pointset<2, Real>::vertex>& vertices) {
  triangle::api api;
  api.behaviour().firstnumber = 0;
  api.mesh_create(ps.to_triangle_io(vertices));
  return mesh<2, Real>{api.mesh_copy()};
}

//------------------------------------------------------------------------------
template <typename Real>
inline auto constrained_delaunay(const edgeset<2, Real>& es) {
  triangle::api api;
  api.behaviour().firstnumber = 0;
  api.behaviour().poly        = true;
  api.behaviour().usesegments = true;
  auto contour                = es.to_triangle_io();
  contour.numberofsegments    = es.num_edges();
  size_t i;

  i                   = 0;
  contour.segmentlist = new int[contour.numberofsegments * 2];
  for (auto e : es.edges()) {
    contour.segmentlist[i]     = es[e][0].i;
    contour.segmentlist[i + 1] = es[e][1].i;
    i += 2;
  }

  api.mesh_create(contour);
  return mesh<2, Real>{api.mesh_copy()};
}

//------------------------------------------------------------------------------
template <typename Real>
inline auto constrained_delaunay(
    const pointset<2, Real>&                               ps,
    const std::vector<typename pointset<2, Real>::vertex>& polygon) {
  triangle::api api;
  api.behaviour().firstnumber = 0;
  api.behaviour().poly        = true;
  api.behaviour().usesegments = true;
  auto contour                = ps.to_triangle_io(polygon);
  contour.numberofsegments    = polygon.size();
  size_t i                    = 0;
  contour.segmentlist         = new int[contour.numberofsegments * 2];
  for (size_t j = 0; j < polygon.size(); ++j, i += 2) {
    contour.segmentlist[i]     = j;
    contour.segmentlist[i + 1] = (j + 1) % polygon.size();
  }

  api.mesh_create(contour);
  return mesh<2, Real>{api.mesh_copy()};
}

//------------------------------------------------------------------------------
template <typename Real>
inline auto conforming_delaunay(const edgeset<2, Real>& es,
                                triangle::real_type          minangle = 0,
                                triangle::real_type          maxarea  = 0) {
  triangle::api api;
  api.options("zpq" + std::to_string(minangle));
  if (maxarea > 0)
    api.options("zpq" + std::to_string(minangle) + "a" +
                std::to_string(maxarea));
  else
    api.options("zpq" + std::to_string(minangle));

  auto contour             = es.to_triangle_io();
  contour.numberofsegments = es.num_edges();
  size_t i                 = 0;
  contour.segmentlist      = new int[contour.numberofsegments * 2];
  for (auto e : es.edges()) {
    contour.segmentlist[i++] = es[e][0].i;
    contour.segmentlist[i++] = es[e][1].i;
  }

  api.mesh_create(contour);
  return mesh<2, Real>{api.mesh_copy()};
}
#endif

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif

